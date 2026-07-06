from collections import deque

import galois
import numpy as np
from tqdm import tqdm


class PureFaultSet:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.faults = np.zeros((0, num_qubits), dtype=np.int8)

    @classmethod
    def from_fault_array(cls, array: np.ndarray) -> 'PureFaultSet':
        if array.ndim != 2:
            raise ValueError("Input array must be 2-dimensional.")
        fault_set = cls(array.shape[1])
        fault_set.faults = np.unique(array, axis=0)
        return fault_set

    @classmethod
    def from_cnots(cls, cnots: list[tuple[int, int]], num_qubits: int, kind: str = "X") -> 'PureFaultSet':
        assert kind.capitalize() in {"X", "Z"}, "Kind must be either 'X' or 'Z'."
        qubit_faults = [[fault] for fault in np.eye(num_qubits, dtype=np.int8)]

        for control, target in reversed(cnots):
            ctrl, trgt = control, target
            if kind.capitalize() == "Z":
                ctrl, trgt = trgt, ctrl
            new_fault = qubit_faults[ctrl][-1] ^ qubit_faults[trgt][-1]
            qubit_faults[ctrl].append(new_fault)

        fs = cls.from_fault_array(np.array([fault for faults in qubit_faults for fault in faults], dtype=np.int8))
        return fs

    def remove_zero_rows(self) -> None:
        if self.faults.size > 0:
            self.faults = self.faults[np.any(self.faults, axis=1)]

    def remove_duplicates(self) -> None:
        if self.faults.size > 0:
            self.faults = np.unique(self.faults, axis=0)

    def add_faults(self, faults: np.ndarray) -> None:
        if faults.ndim != 2 or faults.shape[1] != self.num_qubits:
            raise ValueError(f"Faults array must be 2D with {self.num_qubits} columns.")
        self.faults = np.vstack((self.faults, faults))

    def __len__(self) -> int:
        return int(self.faults.shape[0])


class MWRCalculator:
    def __init__(self, stabs: np.ndarray):
        self.num_qubits = stabs.shape[1] if stabs.shape[0] > 0 else 0

        GF2 = galois.GF(2)
        if stabs.shape[0] == 0:
            self.H = np.zeros((0, self.num_qubits), dtype=np.int8)
        else:
            stabs_gf2 = GF2(stabs)
            self.H = np.array(stabs_gf2.null_space(), dtype=np.int8)

        self.lut = {}
        self.queue = deque([(np.zeros(self.num_qubits, dtype=np.int8), 0, 0)])

        if self.H.shape[0] > 0:
            zero_syn = tuple(np.zeros(self.H.shape[0], dtype=np.int8))
            self.lut[zero_syn] = np.zeros(self.num_qubits, dtype=np.int8)

    def get_mwr(self, fault: np.ndarray) -> np.ndarray:
        if self.H.shape[0] == 0:
            return np.zeros(self.num_qubits, dtype=np.int8)

        syn = tuple((self.H @ fault) % 2)

        while syn not in self.lut and self.queue:
            vec, wt, last_idx = self.queue.popleft()

            for i in range(last_idx, self.num_qubits):
                new_vec = vec.copy()
                new_vec[i] ^= 1
                new_syn = tuple((self.H @ new_vec) % 2)

                if new_syn not in self.lut:
                    self.lut[new_syn] = new_vec

                self.queue.append((new_vec, wt + 1, i + 1))

        if syn in self.lut:
            return self.lut[syn]

        raise RuntimeError(f"Syndrome {syn} not found in BFS space!")

    def get_mwr_batch(self, faults: np.ndarray) -> np.ndarray:
        if len(faults) == 0:
            return np.zeros_like(faults)
        reps = np.zeros_like(faults)
        for i in range(len(faults)):
            reps[i] = self.get_mwr(faults[i])
        return reps


def compute_minimum_weight_representatives(faults: np.ndarray, stabs: np.ndarray) -> np.ndarray:
    calc = MWRCalculator(stabs)
    return calc.get_mwr_batch(faults)


class MixedFaultSet:
    def __init__(
        self,
        single_faults: PureFaultSet,
        t: int,
        H_filter: np.ndarray,
        track_origins: bool = False
    ):
        self.t = t
        self.N = single_faults.num_qubits
        self.track_origins = track_origins

        self.mwr_calc = MWRCalculator(H_filter)

        self.active_errors, self.targets, self.W_effs, self.fault_meta = self._generate(single_faults, H_filter)

    def _generate(self, single_faults: PureFaultSet, H_filter: np.ndarray):
        unique_components = []
        seen = set()

        for f in single_faults.faults:
            key = (f.tobytes(), 0)
            if key not in seen:
                seen.add(key)
                unique_components.append((f, 0))

        I_N = np.eye(self.N, dtype=np.int8)
        for L in range(1, self.t):
            for q in range(self.N):
                f = I_N[q]
                key = (f.tobytes(), L)
                if key not in seen:
                    seen.add(key)
                    unique_components.append((f, L))

        null_active = np.zeros((self.t, self.N), dtype=np.int8)

        current_states = {
            null_active.tobytes(): {
                "active": null_active,
                "weight": 0,
                "origins": [[]] if self.track_origins else []
            }
        }

        for comp_idx, comp in enumerate(tqdm(unique_components, desc="Generating mixed faults")):
            f, L_c = comp

            states_to_expand = list(current_states.values())
            for state in states_to_expand:
                if state["weight"] < self.t:
                    new_weight = state["weight"] + 1
                    new_active = state["active"].copy()
                    new_active[L_c:] ^= f

                    final_data = new_active[-1]
                    mwr = self.mwr_calc.get_mwr(final_data)
                    S = final_data ^ mwr
                    new_active_norm = new_active ^ S

                    key = new_active_norm.tobytes()

                    existing = current_states.get(key)

                    if existing is None:
                        current_states[key] = {
                            "active": new_active_norm,
                            "weight": new_weight,
                            "origins": [o + [comp_idx] for o in state["origins"]] if self.track_origins else []
                        }
                    else:
                        if new_weight < existing["weight"]:
                            existing["weight"] = new_weight
                            existing["active"] = new_active_norm
                            if self.track_origins:
                                existing["origins"] = [o + [comp_idx] for o in state["origins"]]
                        elif new_weight == existing["weight"]:
                            if self.track_origins:
                                existing["origins"].extend([o + [comp_idx] for o in state["origins"]])

        valid_active_errors = []
        valid_targets = []
        valid_Weffs = []
        valid_meta = []

        for state in current_states.values():
            if state["weight"] == 0:
                continue

            k = state["weight"]
            active = state["active"]
            final_data = active[-1]

            mwr = self.mwr_calc.get_mwr(final_data)
            W_eff = int(np.sum(mwr))
            req_det = min(W_eff - k, self.t - k + 1)

            if req_det > 0:
                valid_active_errors.append(active)
                valid_targets.append(req_det)
                valid_Weffs.append(W_eff)

                meta = {
                    "weight": k,
                    "final_data": final_data
                }
                if self.track_origins:
                    meta["origins"] = state["origins"]
                valid_meta.append(meta)

        if not valid_active_errors:
            return np.empty((0, self.t, self.N), dtype=np.int8), np.array([]), np.array([]), []

        return np.array(valid_active_errors), np.array(valid_targets), np.array(valid_Weffs), valid_meta
