import itertools
import os
os.environ["KMP_WARNINGS"] = "0"
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
from spiderstate.utils import load_qecc, MQT_simp_QECCS


class LutDecoder:
    def __init__(self, H, max_decodable_weight=None):
        self.H = H
        self.m, self.n = self.H.shape
        self.max_weight = max_decodable_weight
        if self.max_weight is None:
            self.max_weight = self.n  # Unbounded max weight
        self.powers_of_2 = 1 << np.arange(self.m)[::-1]
        self.lut_size = 1 << self.m
        self.lut = np.full((self.lut_size, self.n), -1, dtype=np.int8)
        self._build_table()

    def _syndrome_int(self, e):
        s = (e @ self.H.T) % 2
        return s @ self.powers_of_2

    def _build_table(self):
        e_zero = np.zeros(self.n, dtype=np.int8)
        self.lut[0] = e_zero
        if self.max_weight > 0:
            for w in range(1, self.max_weight + 1):
                for error_positions in itertools.combinations(range(self.n), w):
                    e = np.zeros(self.n, dtype=np.int8)
                    e[list(error_positions)] = 1
                    s_int = self._syndrome_int(e)
                    if self.lut[s_int, 0] == -1:
                        self.lut[s_int] = e

    def batch_decode_z(self, syndromes):
        s_ints = np.asarray(syndromes) @ self.powers_of_2
        return self.lut[s_ints]


# Global variables for the worker processes
_worker_sampler = None
_worker_decoder = None
_worker_H_x = None
_worker_L_x = None
_worker_g_H_x_T = None
_worker_g_L_x_T = None
_GF = None

def _init_worker(circ_str, H_x, L_x, max_weight):
    global _worker_sampler, _worker_decoder
    global _worker_H_x, _worker_L_x
    global _worker_g_H_x_T, _worker_g_L_x_T, _GF
    
    _GF = galois.GF(2)
    _worker_sampler = stim.Circuit(circ_str).compile_sampler()
    _worker_decoder = LutDecoder(H_x, max_weight)
    _worker_H_x = H_x
    _worker_L_x = L_x
    _worker_g_H_x_T = _GF(H_x.T)
    _worker_g_L_x_T = _GF(L_x.T)


def _simulate_batch(batch_size):
    samples = _worker_sampler.sample(batch_size)

    # Flagged shots (any 1 in the flag measurements)
    is_flagged = np.any(samples[:, :-_worker_H_x.shape[1]], axis=1)
    num_flagged = np.sum(is_flagged)

    filtered_samples = samples[~is_flagged]
    raw_measurements = filtered_samples[:, -_worker_H_x.shape[1]:]
    
    # Fast GF(2) matrix multiplication for syndromes
    g_raw = _GF(raw_measurements.astype(np.int8))
    syndromes = g_raw @ _worker_g_H_x_T
    
    corrections = _worker_decoder.batch_decode_z(syndromes)

    valid_mask = np.min(corrections, axis=1) != -1
    valid_corrections = corrections[valid_mask]
    num_discarded = len(syndromes) - len(valid_corrections)

    valid_measurements = raw_measurements[valid_mask]
    # Fast bitwise XOR using NumPy (which is extremely fast on int8)
    corrected_measurements = valid_measurements ^ valid_corrections
    
    # Fast GF(2) matrix multiplication for logicals
    g_corrected = _GF(corrected_measurements)
    predicted_logicals = g_corrected @ _worker_g_L_x_T

    # Incorrect if any logical observable is flipped
    incorrect_predictions = np.any(predicted_logicals, axis=1)
    num_incorrect = np.sum(incorrect_predictions)

    return batch_size, int(num_flagged), int(num_discarded), int(num_incorrect)


def benchmark_CAO_state_prep(code: str, p=0.001, num_samples=100_000_000):
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code)
    if code in ("49_1_5", "95_1_7"):
        print(f"State: |+> (Code {code})")
        H_x, H_z = H_z, H_x
        L_x, L_z = L_z, L_x
    else:
        print(f"State: |0> (Code {code})")
    
    circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=25_000)
    noisy_circ, _ = make_stim_circ_noisy(circ, p, one_cnot_per_layer=True)
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
    raw_cnots = [l for (name, l, _) in circ.flattened_operations() if name == "CX"]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    num_cx = len(cnots)
    num_flags = circ.num_qubits - H_x.shape[1]
    num_qubits = circ.num_qubits
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
    max_weight = None  # User requested unbound max weight
    max_weight = d // 2  # User requested unbound max weight

    if remaining_samples > 0:
        batch_size = 10_000_000
        num_full_batches = remaining_samples // batch_size
        remainder = remaining_samples % batch_size
        batches = [batch_size] * num_full_batches
        if remainder > 0:
            batches.append(remainder)
            
        print(f"Running {remaining_samples} additional samples (Total existing: {total_shots})...")
        
        with ProcessPoolExecutor(initializer=_init_worker, initargs=(circ_str, H_x, L_x, max_weight)) as executor:
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
                    total_incorrect += n_incorrect
                    
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
    total_AR = total_valid_corrections / total_shots if total_shots > 0 else 0.0
    
    print(f"Discarded {total_discarded} uncorrectable shots.")
    
    LER = total_incorrect / total_valid_corrections if total_valid_corrections > 0 else 0.0

    return LER, total_AR, num_cx, num_flags, num_qubits, num_sim_qubits, depth


if __name__ == "__main__":
    methods = {"MQT": MQT_simp_QECCS}
    for method_name, code_iterator in methods.items():
        for code in code_iterator():
            LER, AR, num_cx, num_flags, num_qubits, num_sim_qubits, depth = benchmark_CAO_state_prep(
                code, num_samples=10_000_000
            )
            # TODO: save circuit + circuit data in a json
            print(f"Logical Error Rate = {LER:.4e}", end=";\t ")
            print(f"Acceptance Rate = {AR:.4f}", end=";\t ")
            print(f"CXs = {num_cx}", end=";\t ")
            print(f"Sim. Qubits = {num_sim_qubits}", end=";\t ")
            print(f"Flags = {num_flags}", end=";\t ")
            print(f"Depth = {depth}", end=";\t ")
            print(f"Expected Circuit Volume = {int(depth * num_sim_qubits / AR) if AR > 0 else 0}")
            print()
