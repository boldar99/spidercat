from collections import deque

import galois
import numpy as np
from tqdm import tqdm
import itertools
from spiderstate.cnot_scheduler import DangerousFault


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

        self.active_errors, self.targets, self.W_effs, self.fault_meta = self._generate(single_faults)

    def _generate(self, single_faults: PureFaultSet):
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

    def precompute_T_E_Q(self) -> dict:
        unique_final_data_keys = {}
        num_mixed_faults = len(self.active_errors)
        for i in range(num_mixed_faults):
            T_E = self.targets[i]
            if T_E <= 0:
                continue
            k = self.fault_meta[i]["weight"]
            final_data = self.fault_meta[i]["final_data"]
            key = (tuple(final_data), k, T_E)
            if key not in unique_final_data_keys:
                unique_final_data_keys[key] = []
            unique_final_data_keys[key].append(i)
            
        precomputed_T_E_Q = {}
        all_combs_list = []
        key_to_slices = {}
        current_idx = 0
        
        for key in unique_final_data_keys:
            final_data_tuple, k, T_E = key
            final_data = np.array(final_data_tuple, dtype=np.int8)
            T_E_Q = {(): int(T_E)}
            
            key_combs = []
            combs_meta = []
            for size in range(1, self.t - k + 1):
                combs = list(itertools.combinations(range(self.N), size))
                if not combs:
                    continue
                combs_array = np.zeros((len(combs), self.N), dtype=np.int8)
                for c_idx, comb in enumerate(combs):
                    combs_array[c_idx] = final_data
                    for q in comb:
                        combs_array[c_idx, q] ^= 1
                key_combs.append(combs_array)
                combs_meta.append((size, combs))
                
            if key_combs:
                stacked_combs = np.vstack(key_combs)
                n_combs = len(stacked_combs)
                all_combs_list.append(stacked_combs)
                key_to_slices[key] = (current_idx, current_idx + n_combs, combs_meta)
                current_idx += n_combs
            else:
                precomputed_T_E_Q[key] = T_E_Q
                
        if all_combs_list:
            giant_combs_array = np.vstack(all_combs_list)
            giant_reps = self.mwr_calc.get_mwr_batch(giant_combs_array)
            giant_weffs = np.sum(giant_reps, axis=1)
            
            for key in key_to_slices:
                final_data_tuple, k, T_E = key
                start_idx, end_idx, combs_meta = key_to_slices[key]
                key_weffs = giant_weffs[start_idx:end_idx]
                
                T_E_Q = {(): int(T_E)}
                offset = 0
                for size, combs in combs_meta:
                    n_c = len(combs)
                    weff_combs = key_weffs[offset : offset + n_c]
                    t_e_combs = np.minimum(weff_combs - (k + size), self.t - (k + size) + 1)
                    for c_idx, comb in enumerate(combs):
                        T_E_Q[comb] = int(t_e_combs[c_idx])
                    offset += n_c
                precomputed_T_E_Q[key] = T_E_Q

        return precomputed_T_E_Q

    def find_dangerous_faults(self, chosen_layers: list[list[np.ndarray]], precomputed_T_E_Q: dict) -> list[DangerousFault]:
        num_mixed_faults = len(self.active_errors)
        if num_mixed_faults == 0:
            return []
            
        GF2 = galois.GF(2)
        active_errors_gf2 = GF2(self.active_errors)
        
        D_E_all = [set() for _ in range(num_mixed_faults)]
        D_eq_L_E_counts_all = np.zeros((num_mixed_faults, self.t), dtype=int)
        
        for l_idx, layer in enumerate(chosen_layers):
            if len(layer) == 0:
                continue
            layer_matrix = np.vstack(layer).astype(np.int8)
            syn_all = np.array(active_errors_gf2[:, l_idx, :] @ GF2(layer_matrix).T, dtype=np.int8)
            
            fault_indices, stab_indices = np.nonzero(syn_all)
            for f_idx, s_idx in zip(fault_indices, stab_indices):
                D_E_all[f_idx].add((l_idx, s_idx))
                D_eq_L_E_counts_all[f_idx, l_idx] += 1
                
        dangerous_faults = []
        for i in range(num_mixed_faults):
            T_E = self.targets[i]
            if T_E <= 0:
                continue
                
            D_E = D_E_all[i]
            D_eq_L_E_counts = D_eq_L_E_counts_all[i]
                        
            key = (tuple(self.fault_meta[i]["final_data"]), self.fault_meta[i]["weight"], T_E)
            T_E_Q = precomputed_T_E_Q[key]
            
            is_dangerous = False
            for L in range(len(chosen_layers)):
                abs_D_eq_L = D_eq_L_E_counts[L]
                abs_D_gt_L = sum(D_eq_L_E_counts[L+1:])
                
                for Q, req in T_E_Q.items():
                    syn_gt_L_Q = set()
                    for q in Q:
                        for l_idx in range(L + 1, len(chosen_layers)):
                            for s_idx, st in enumerate(chosen_layers[l_idx]):
                                if q in np.nonzero(st)[0]:
                                    if (l_idx, s_idx) in syn_gt_L_Q:
                                        syn_gt_L_Q.remove((l_idx, s_idx))
                                    else:
                                        syn_gt_L_Q.add((l_idx, s_idx))
                                        
                    D_gt_L = { (l_idx, s_idx) for (l_idx, s_idx) in D_E if l_idx > L }
                    xor_set = D_gt_L.symmetric_difference(syn_gt_L_Q)
                    M_E_Q = len(D_E) - abs_D_gt_L + len(xor_set) - req
                    
                    if M_E_Q < abs_D_eq_L:
                        is_dangerous = True
                        break
                if is_dangerous:
                    break
                    
            if is_dangerous:
                dangerous_faults.append(DangerousFault(frozenset(D_E), T_E_Q))
                    
        df_dict = {}
        for df in dangerous_faults:
            if df.D_E not in df_dict:
                df_dict[df.D_E] = df.T_E_Q.copy()
            else:
                for Q in df.T_E_Q:
                    if Q not in df_dict[df.D_E]:
                        df_dict[df.D_E][Q] = df.T_E_Q[Q]
                    else:
                        df_dict[df.D_E][Q] = max(df_dict[df.D_E][Q], df.T_E_Q[Q])
                    
        return [DangerousFault(D_E, T_E_Q) for D_E, T_E_Q in df_dict.items()]
