import os

os.environ["KMP_WARNINGS"] = "0"
import multiprocessing as mp

mp.set_start_method("fork", force=True)
import hashlib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import stim
import galois

from spiderstate.stim_utils import make_stim_circ_noisy
from spidercat.simulate import _layer_cnot_circuit
from spiderstate.cat_at_origin import row_optimized_cat_at_origin
from spiderstate.utils import load_qecc, FAO_simp_QECCS, hard_QECCS
from spiderstate.qubit_reuse import (
    build_circuit_dag,
    inject_qubit_reuse,
    apply_logical_qubit_merge_and_compress,
    dag_to_circuit,
    DepthPreservingStrategy, PureAggressiveStrategy
)
import json


from spiderstate.lut_decoder import LutDecoder

# Globals to inherit via OS fork (Zero-Copy)
_ESTIMATE_LER: bool = True
_G_CIRC_STR: str = None
_G_DECODER: LutDecoder = None
_G_H_X: np.ndarray = None
_G_L_X: np.ndarray = None


def _simulate_batch(batch_size):
    # Compiling per-batch mathematically guarantees independent PRNG streams
    # and takes negligible time (~11 microseconds)
    sampler = stim.Circuit(_G_CIRC_STR).compile_sampler()
    samples = sampler.sample(batch_size)

    _GF = galois.GF(2)
    g_H_x_T = _GF(_G_H_X.T)
    g_L_x_T = _GF(_G_L_X.T)

    # Flagged shots (any 1 in the flag measurements)
    is_flagged = np.any(samples[:, :-_G_H_X.shape[1]], axis=1)
    num_flagged = np.sum(is_flagged)

    filtered_samples = samples[~is_flagged]
    raw_measurements = filtered_samples[:, -_G_H_X.shape[1]:]

    # Fast GF(2) matrix multiplication for syndromes
    g_raw = _GF(raw_measurements.astype(np.int8))
    syndromes = g_raw @ g_H_x_T

    if not _ESTIMATE_LER:
        return batch_size, int(num_flagged), 0, None

    corrections, valid_mask = _G_DECODER.batch_decode_z(syndromes)

    valid_corrections = corrections[valid_mask]
    num_discarded = len(syndromes) - len(valid_corrections)

    valid_measurements = raw_measurements[valid_mask]
    # Fast bitwise XOR using NumPy
    corrected_measurements = valid_measurements ^ valid_corrections
    
    # Fast GF(2) matrix multiplication for logicals
    g_corrected = _GF(corrected_measurements.astype(np.int8))
    predicted_logicals = g_corrected @ g_L_x_T

    # Incorrect if any logical observable is flipped
    incorrect_predictions = np.any(predicted_logicals, axis=1)
    num_incorrect = np.sum(incorrect_predictions)

    return batch_size, int(num_flagged), int(num_discarded), int(num_incorrect)


def benchmark_CAO_state_prep(code: str, reuse_strategy, p=0.001, num_samples=100_000_000, estimate_ler=True):
    import random
    # Ensure deterministic circuit generation for this specific code
    # so the circuit hash matches across different script executions
    seed_val = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16)
    random.seed(seed_val)
    np.random.seed(seed_val)

    try:
        is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code, "FAO")
    except FileNotFoundError:
        is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code)
    if code in ("49_1_5", "95_1_7"):
        print(f"State: |+> (Code {code})")
        H_x, H_z = H_z, H_x
        L_x, L_z = L_z, L_x
    else:
        print(f"State: |0> (Code {code})")

    original_circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=25_000)

    n_data = H_x.shape[1]
    dag = build_circuit_dag(original_circ)
    mod_dag, _, _ = inject_qubit_reuse(dag, n_data, reuse_strategy)
    compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, n_data)
    circ_with_reuse, _ = dag_to_circuit(compressed_dag)

    noisy_circ, _ = make_stim_circ_noisy(circ_with_reuse, p, one_cnot_per_layer=True)
    # print(noisy_circ)

    noisy_circ.append("M", range(H_x.shape[1]))

    for i, H in enumerate(H_x):
        qubit_indices = np.where(H == 1)[0]
        record_targets = [stim.target_rec(i - H_x.shape[1]) for i in qubit_indices]
        noisy_circ.append("DETECTOR", record_targets)
    for i, L in enumerate(L_x):
        qubit_indices = np.where(L == 1)[0]
        record_targets = [stim.target_rec(i - H_x.shape[1]) for i in qubit_indices]
        noisy_circ.append("OBSERVABLE_INCLUDE", record_targets, i)

    # Compute circuit properties
    raw_cnots = [l for (name, l, _) in circ_with_reuse.flattened_operations() if name == "CX"]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    num_cx = len(cnots)
    num_flags = original_circ.num_detectors
    num_qubits = noisy_circ.num_qubits
    depth = len(_layer_cnot_circuit(cnots))
    num_sim_qubits = noisy_circ.num_qubits

    # Compute circuit hash and setup CSV caching
    circ_str = str(noisy_circ)
    circ_hash = hashlib.sha256(circ_str.encode()).hexdigest()[:16]

    os.makedirs("simulation_results", exist_ok=True)
    csv_file = f"simulation_results/{code}_{circ_hash}.csv"

    total_shots = 0
    total_flagged = 0
    total_discarded = 0
    total_incorrect = 0

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            total_shots = int(df['total_shots'].sum())
            total_flagged = int(df['num_flagged'].sum())
            total_discarded = int(df['num_discarded'].sum())
            total_incorrect = int(df['num_incorrect'].sum())

    remaining_samples = max(0, num_samples - total_shots)
    max_weight = None if bool(d % 2) else (d - 1) // 2

    # Build the LUT ONCE in the main thread
    decoder = estimate_ler and LutDecoder(H_x, max_decodable_weight=max_weight)

    global _G_DECODER, _G_CIRC_STR, _G_H_X, _G_L_X, _ESTIMATE_LER
    _ESTIMATE_LER = estimate_ler
    _G_DECODER = decoder
    _G_CIRC_STR = circ_str
    _G_H_X = H_x
    _G_L_X = L_x

    if remaining_samples > 0:
        batch_size = 1_000_000
        num_full_batches = remaining_samples // batch_size
        remainder = remaining_samples % batch_size
        batches = [batch_size] * num_full_batches
        if remainder > 0:
            batches.append(remainder)

        print(f"Running {remaining_samples} additional samples (Total existing: {total_shots})...")

        num_cores = max(1, mp.cpu_count() - 2)
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(_simulate_batch, b_size)
                for b_size in batches
            ]

            with tqdm(total=remaining_samples, desc=f"Simulating {code}") as pbar:
                for future in as_completed(futures):
                    b_size, n_flagged, n_discarded, n_incorrect = future.result()
                    total_shots += b_size
                    total_flagged += n_flagged
                    total_discarded += n_discarded
                    total_incorrect = (total_incorrect + n_incorrect) if estimate_ler else None

                    # Update CSV incrementally
                    df_new = pd.DataFrame([{
                        "total_shots": b_size,
                        "num_flagged": n_flagged,
                        "num_discarded": n_discarded,
                        "num_incorrect": n_incorrect
                    }])

                    if os.path.exists(csv_file):
                        df_new.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        df_new.to_csv(csv_file, index=False)

                    pbar.update(b_size)
    else:
        print(f"Using {total_shots} cached samples from {csv_file}")

    # Compute final metrics
    AR = 1.0 - (total_flagged / total_shots) if total_shots > 0 else 0.0
    total_valid_corrections = total_shots - total_flagged - total_discarded
    total_AR = total_valid_corrections / total_shots if total_shots > 0 else None

    print(f"Discarded {total_discarded} uncorrectable shots.")

    if estimate_ler:
        LER = total_incorrect / total_valid_corrections if total_valid_corrections > 0 else 0.0
    else:
        LER = None

    stats = {
        "code": code,
        "strategy": reuse_strategy.__class__.__name__,
        "p": p,
        "num_samples": total_shots,
        "total_flagged": total_flagged,
        "total_discarded": total_discarded,
        "total_incorrect": total_incorrect,
        "logical_error_rate": LER,
        "acceptance_rate": total_AR ,
        "raw_acceptance_rate": AR,
        "num_cx": num_cx,
        "num_flags": num_flags,
        "num_qubits_original": num_qubits,
        "num_sim_qubits": num_sim_qubits,
        "depth": depth,
        "circuit_volume": int(depth * num_sim_qubits),
        "expected_circuit_volume": int(depth * num_sim_qubits / total_AR) if total_AR is not None and total_AR > 0 else 0,
        "circuit_hash": circ_hash,
        "perfect_stim": str(circ_with_reuse),
        "noisy_circuit": circ_str,
    }

    json_file = f"simulation_results/{code}_{reuse_strategy.__class__.__name__}_{circ_hash}.json"
    with open(json_file, "w") as f:
        json.dump(stats, f, indent=4)

    return stats


def benchmark_with_lut(code_iterator):
    strategies = [
        PureAggressiveStrategy,
        DepthPreservingStrategy,
    ]
    for code in code_iterator():
        for StrategyClass in strategies:
            print(f"--- Benchmarking {code} with {StrategyClass.__name__} ---")
            stats = benchmark_CAO_state_prep(
                code, reuse_strategy=StrategyClass(), num_samples=1_000_000_000
            )
            print(f"Logical Error Rate = {stats['logical_error_rate']:.4e}", end=";\t ")
            print(f"Acceptance Rate = {stats['acceptance_rate']:.4f}", end=";\t ")
            print(f"CXs = {stats['num_cx']}", end=";\t ")
            print(f"Sim. Qubits = {stats['num_sim_qubits']}", end=";\t ")
            print(f"Flags = {stats['num_flags']}", end=";\t ")
            print(f"Depth = {stats['depth']}", end=";\t ")
            print(f"Expected Circuit Volume = {stats['expected_circuit_volume']}")
            print()


def benchmark_without_lut(code_iterator):
    strategies = [
        PureAggressiveStrategy,
        DepthPreservingStrategy,
    ]
    for code in code_iterator():
        for StrategyClass in strategies:
            print(f"--- Benchmarking {code} with {StrategyClass.__name__} ---")
            stats = benchmark_CAO_state_prep(
                code, reuse_strategy=StrategyClass(), num_samples=0, estimate_ler=False
            )
            if stats['logical_error_rate'] is not None:
                print(f"Logical Error Rate = {stats['logical_error_rate']:.4e}", end=";\t ")
            if stats['acceptance_rate'] is not None:
                print(f"Acceptance Rate = {stats['acceptance_rate']:.4f}", end=";\t ")
            print(f"CXs = {stats['num_cx']}", end=";\t ")
            print(f"Sim. Qubits = {stats['num_sim_qubits']}", end=";\t ")
            print(f"Flags = {stats['num_flags']}", end=";\t ")
            print(f"Depth = {stats['depth']}", end=";\t ")
            print(f"Expected Circuit Volume = {stats['expected_circuit_volume']}")
            print()


if __name__ == "__main__":
    benchmark_without_lut(hard_QECCS)
