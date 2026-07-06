import itertools

import numpy as np
import stim

from spidercat.circuit_extraction import make_stim_circ_noisy
from spidercat.simulate import _layer_cnot_circuit
from spiderstate.cat_at_origin import row_optimized_cat_at_origin
from spiderstate.utils import load_qecc, MQT_simp_QECCS


class LutDecoder:
    def __init__(self, H, max_decodable_weight=None):
        self.H = H
        self.m, self.n = self.H.shape
        self.max_weight = max_decodable_weight
        if self.max_weight is None:
            self.max_weight = 0
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
        s_ints = syndromes @ self.powers_of_2
        return self.lut[s_ints]


def benchmark_CAO_state_prep(code: str, p=0.001, num_samples=100_000_000):
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code)
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
    max_weight = (d - 1) // 2
    decoder = LutDecoder(H_x, max_decodable_weight=max_weight)
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
    methods = {"MQT": MQT_simp_QECCS}
    for method_name, code_iterator in methods.items():
        for code in code_iterator():
            LER, AR, num_cx, num_flags, num_qubits, num_sim_qubits, depth = benchmark_CAO_state_prep(
                code, num_samples=1_000_000
            )
            print(f"Logical Error Rate = {LER:.4e}", end=";\t ")
            print(f"Acceptance Rate = {AR:.4f}", end=";\t ")
            print(f"CXs = {num_cx}", end=";\t ")
            print(f"Sim. Qubits = {num_sim_qubits}", end=";\t ")
            print(f"Flags = {num_flags}", end=";\t ")
            print(f"Depth = {depth}", end=";\t ")
            print(f"Expected Circuit Volume = {int(depth * num_sim_qubits / AR)}")
            print()
