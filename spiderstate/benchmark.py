import concurrent.futures
import json

import numpy as np
import stim
import tesseract_decoder
from tesseract_decoder import tesseract

from spidercat.circuit_extraction import make_stim_circ_noisy
from spidercat.simulate import _layer_cnot_circuit
from spiderstate.cat_at_origin import row_optimized_cat_at_origin
from spiderstate.utils import load_qecc, FAO_QECCS, _expand_stim_operation_list, _layer_circuit_ops, \
    layered_ops_to_noisy_stim_circuit, apply_qubit_reuse, MQT_simp_QECCS

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import LutDecoder


def benchmark_CAO_state_prep(code: str, method: str, p=0.001, num_samples=100_000_000):
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code, method)
    if code in ("49_1_5", "95_1_7"):
        print("State: |+>")
        H_x, H_z = H_z, H_x
        L_x, L_z = L_z, L_x
    else:
        print("State: |0>")
    circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=25_000)
    noisy_circ = make_stim_circ_noisy(circ, p)

    noisy_circ.append("M", range(H_x.shape[1]))

    for i, H in enumerate(H_x):
        qubit_indices = np.where(H == 1)[0]
        record_targets = [stim.target_rec(i - H_x.shape[1]) for i in qubit_indices]
        noisy_circ.append("DETECTOR", record_targets)
    for i, L in enumerate(L_x):
        qubit_indices = np.where(L == 1)[0]
        record_targets = [stim.target_rec(i - H_x.shape[1]) for i in qubit_indices]
        noisy_circ.append("OBSERVABLE_INCLUDE", record_targets, i)

    # 3. Sample detectors and logicals
    samples = noisy_circ.compile_sampler().sample(num_samples)
    total_shots = len(samples)

    # 4. Post-selection: Identify flagged shots
    is_flagged = np.any(samples[:, :-H_x.shape[1]], axis=1)
    AR = 1.0 - np.average(is_flagged)

    filtered_samples = samples[~is_flagged]
    raw_measurements = filtered_samples[:, -H_x.shape[1]:]
    syndromes = raw_measurements @ H_x.T % 2
    code = CSSCode(Hx=H_x, Hz=H_z, distance=d)

    max_weight = (code.distance - 1) // 2
    decoder = LutDecoder(code, max_decodable_weight=max_weight if code.distance % 2 == 0 else None)
    corrections = decoder.batch_decode_z(syndromes)

    # 3. Post-selection: Find valid rows
    # Since valid correction arrays only contain 0s and 1s, checking if the
    # minimum value in the row is -1 instantly flags the sentinels.
    valid_mask = np.min(corrections, axis=1) != -1

    # 4. Filter the raw data
    valid_measurements = raw_measurements[valid_mask]
    valid_corrections = corrections[valid_mask]

    # 5. Apply corrections safely
    corrected_measurements = valid_measurements ^ valid_corrections
    predicted_logicals = corrected_measurements @ L_x.T % 2

    # Optional: Track your post-selection discard rate
    discarded_shots = len(syndromes) - len(valid_corrections)
    print(f"Discarded {discarded_shots} uncorrectable shots.")

    # If any logical observable failed to be corrected in a shot, that shot is a logical error
    incorrect_predictions = np.any(predicted_logicals, axis=1)
    LER = np.average(incorrect_predictions) if len(incorrect_predictions) > 0 else 0.0

    # Total Experimental Yield
    total_AR = len(valid_corrections) / total_shots

    raw_cnots = [l for (name, l, _) in circ.flattened_operations() if name == "CX"]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    num_cx = len(cnots)
    num_flags = circ.num_qubits - H_x.shape[1]
    num_qubits = circ.num_qubits
    depth = len(_layer_cnot_circuit(cnots))

    return LER, total_AR, num_cx, num_flags, num_qubits, noisy_circ.num_qubits, depth


if __name__ == "__main__":

    # LER, AR, num_cx, num_flags, num_qubits, depth = benchmark_CAO_state_prep("95_1_7", "FAO")
    methods = {"FAO": MQT_simp_QECCS}
    for method_name, code_iterator in methods.items():
        for code in code_iterator():
            [n, k, d] = code.split("_")
            if int(n) > 40:
                continue
            print(method_name, code)
            LER, AR, num_cx, num_flags, num_qubits, num_sim_qubits, depth = benchmark_CAO_state_prep(code, method_name, num_samples=100_000_000)
            print(f"Logical Error Rate = {LER:.4e}", end=";\t ")
            print(f"Acceptance Rate = {AR:.4f}", end=";\t ")
            print(f"CXs = {num_cx}", end=";\t ")
            print(f"Sim. Qubits = {num_sim_qubits}", end=";\t ")
            print(f"Flags = {num_flags}", end=";\t ")
            print(f"Depth = {depth}", end=";\t ")
            print(f"Expected Circuit Volume = {int(depth * num_sim_qubits / AR)}")
            print()
