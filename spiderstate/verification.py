import logging
from itertools import combinations

import numpy as np
from mqt.qecc.circuit_synthesis.faults import PureFaultSet, product_fault_set
from spiderstate.fast_verification import fast_greedy_set_cover
from mqt.qecc.circuit_synthesis import CNOTCircuit
from spidercat.syndrome_measurement import cnot_cost

logger = logging.getLogger(__name__)

def compute_unitary_fault_set_1(cnots: list[tuple[int, int]], num_qubits: int, kind: str = "X"):
    circ = CNOTCircuit()
    seen = set()
    for (c, n) in cnots:
        if c not in seen:
            seen.add(c)
            circ.initialize_qubit(c, "X")
        if n not in seen:
            seen.add(n)
            circ.initialize_qubit(n, "Z")
    for rem in set(range(num_qubits)) - seen:
        circ.initialize_qubit(rem, "Z")
    circ.add_cnots(cnots)
    single_faults = PureFaultSet.from_cnot_circuit(circ, kind=kind)
    single_faults.remove_zero_rows()
    single_faults.remove_duplicates()
    return single_faults


def compute_bare_injected_faults(stabs_layers: list[list[np.ndarray]], num_qubits: int) -> PureFaultSet:
    injected_faults = []
    for layer in stabs_layers:
        for stab in layer:
            qubits = np.where(stab)[0].tolist()
            # For a bare SE circuit, a fault on the ancilla propagates to a suffix of the data qubits
            # The suffixes correspond to the faults injected between the sequential CNOTs
            for k in range(len(qubits)):
                f = np.zeros(num_qubits, dtype=np.int8)
                f[qubits[k:]] = 1
                injected_faults.append(f)
                
    if injected_faults:
        fs = PureFaultSet.from_fault_array(np.array(injected_faults, dtype=np.int8))
    else:
        fs = PureFaultSet.from_fault_array(np.zeros((0, num_qubits), dtype=np.int8))
        
    fs.remove_zero_rows()
    fs.remove_duplicates()
    return fs


def _generate_candidate_stabilizers(stabs: np.ndarray, max_combinations: int) -> np.ndarray:
    k = stabs.shape[0]
    candidate_stabs = []
    
    if k <= 8:
        for i in range(1, 1 << k):
            comb = [j for j in range(k) if (i >> j) & 1]
            s = np.bitwise_xor.reduce(stabs[comb], axis=0)
            if np.any(s):
                candidate_stabs.append(s)
    else:
        for r in range(1, max_combinations + 1):
            for comb in combinations(range(k), r):
                s = np.bitwise_xor.reduce(stabs[list(comb)], axis=0)
                if np.any(s):
                    candidate_stabs.append(s)
                    
    if not candidate_stabs:
        return np.empty((0, stabs.shape[1]), dtype=np.int8)
        
    candidate_stabs = np.unique(np.array(candidate_stabs, dtype=np.int8), axis=0)
    return candidate_stabs

def _greedy_set_cover(coverage: np.ndarray, weights: np.ndarray, candidate_stabs: np.ndarray) -> list[np.ndarray]:
    num_candidates, num_faults = coverage.shape
    uncovered = set(range(num_faults))
    selected_idx = []
    
    subsets = [set(np.where(coverage[i])[0]) for i in range(num_candidates)]
    costs = weights * 10 + 1
    
    while uncovered:
        best_idx = None
        best_ratio = float('inf')
        
        for i in range(num_candidates):
            covered_by_this = subsets[i] & uncovered
            if not covered_by_this:
                continue
            ratio = costs[i] / len(covered_by_this)
            if ratio < best_ratio:
                best_ratio = ratio
                best_idx = i
                
        if best_idx is None:
            break
            
        selected_idx.append(best_idx)
        uncovered -= subsets[best_idx]
        
    return [candidate_stabs[i] for i in selected_idx]

def _solve_top_n_weighted_set_covers(
    faults: np.ndarray, stabs: np.ndarray, t: int, max_combinations: int, max_time_sec: int, top_n: int, target_coverage: np.ndarray | int = 1
) -> list[list[np.ndarray]]:
    faults = faults[np.any(faults, axis=1)]
    if len(faults) == 0:
        return [[]]
        
    candidate_stabs = _generate_candidate_stabilizers(stabs, max_combinations)
    if candidate_stabs.shape[0] == 0:
        return []
        
    weights = np.sum(candidate_stabs, axis=1)

    costs = np.array([cnot_cost(w, t) for w in weights])
    
    coverage = ((candidate_stabs @ faults.T) % 2).astype(bool)
    
    num_candidates = candidate_stabs.shape[0]
    
    coverable = np.any(coverage, axis=0)
    valid_cov = coverage[:, coverable]
    
    if valid_cov.shape[1] == 0:
        return []

    if isinstance(target_coverage, int):
        uncovered_start = np.full(valid_cov.shape[1], target_coverage, dtype=int)
    else:
        uncovered_start = target_coverage[coverable].astype(int)
        
    covered_counts = np.sum(valid_cov, axis=1)
    valid = covered_counts > 0
    if not valid.any():
        return []
        
    ratios = np.full(num_candidates, np.inf)
    ratios[valid] = costs[valid] / covered_counts[valid]
    
    best_starts = np.argsort(ratios)[:top_n * 2]
    
    unique_covers = []
    seen_signatures = set()
    
    for start_idx in best_starts:
        if ratios[start_idx] == np.inf:
            break
            
        uncovered = uncovered_start.copy()
        uncovered[valid_cov[start_idx]] -= 1
        
        selected_idx = [start_idx]
        if np.any(uncovered > 0):
            from spiderstate.fast_verification import fast_greedy_set_cover
            rest_idx = fast_greedy_set_cover(valid_cov, costs, uncovered)

            selected_idx.extend(rest_idx)
            
        selected_idx.sort()
        sig = tuple(selected_idx)
        
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_covers.append([candidate_stabs[i] for i in selected_idx])
            
        if len(unique_covers) >= top_n:
            break
            
    return unique_covers

def compute_effective_weights(faults: np.ndarray, stabs: np.ndarray) -> np.ndarray:
    current_faults = faults.copy()
    weights = np.sum(current_faults, axis=1)
    improved = True
    while improved:
        improved = False
        for stab in stabs:
            candidate = (current_faults + stab) % 2
            cand_weights = np.sum(candidate, axis=1)
            mask = cand_weights < weights
            if np.any(mask):
                current_faults[mask] = candidate[mask]
                weights[mask] = cand_weights[mask]
                improved = True
    return weights

def find_low_weight_verification_stabilizers(fault_sets: list[PureFaultSet], stabs: np.ndarray, max_combinations: int = 4, max_time_sec: int = 60) -> list[list[np.ndarray]]:
    logger.info("Finding low-weight verification stabilizers using Z3 ILP")
    n_layers = len(fault_sets)
    layers: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for num_errors in range(n_layers):
        faults_obj = fault_sets[num_errors]
        if len(faults_obj) == 0:
            layers[num_errors] = []
            continue
        covers = _solve_top_n_weighted_set_covers(faults_obj.faults, stabs, n_layers, max_combinations, max_time_sec, top_n=1)
        if covers:
            layers[num_errors] = covers[0]
    return layers

def _generate_raw_fault_sets(single_faults: PureFaultSet, t: int, H_filter: np.ndarray) -> list[PureFaultSet]:
    """Generates the raw unfiltered unitary fault sets U_1, ..., U_t."""
    fault_sets = []
    
    current = single_faults.copy()
    current.filter_by_weight_at_least(2, H_filter)
    fault_sets.append(current)
    
    for k in range(2, t + 1):
        if len(fault_sets[-1]) == 0:
            empty_fs = fault_sets[-1].copy()
            fault_sets.append(empty_fs)
            continue
            
        next_raw = product_fault_set(fault_sets[-1], single_faults)
        # Include lower weight faults as well
        next_raw.faults = np.concatenate((fault_sets[-1].faults, next_raw.faults))
        next_raw.faults = np.unique(next_raw.faults, axis=0)
        next_raw.remove_zero_rows()
        next_raw.filter_by_weight_at_least(k + 1, H_filter)
        fault_sets.append(next_raw)
        
    return fault_sets

def find_lookahead_verification_stabilizers(
    single_faults: PureFaultSet,
    stabs: np.ndarray,
    H_filter: np.ndarray,
    t: int,
    max_combinations: int = 4,
    top_n: int = 50,
    beam_width: int = 5,
    max_time_sec: int = 60,
    verbose: bool = False
) -> list[list[np.ndarray]]:
    """
    Finds verification stabilizers for t layers using mathematical lookahead.
    It minimizes the size of the next layer's raw fault set after filtering.
    """
    candidate_stabs = _generate_candidate_stabilizers(stabs, max_combinations)
    costs = np.array([cnot_cost(w, t) for w in np.sum(candidate_stabs, axis=1)])
    
    raw_fault_sets = _generate_raw_fault_sets(single_faults, t, H_filter)
    
    beam = [(0.0, [], [])]
    
    for layer_idx in range(t):
        raw_current = raw_fault_sets[layer_idx]
        next_beam_candidates = []
        
        if verbose:
            print(f"  [Layer {layer_idx + 1}] Expanding beam of size {len(beam)}:")
            
        for state_idx, (realized_cost, layers, accumulated_stabs) in enumerate(beam):
            if accumulated_stabs:
                current_stabs_arr = np.vstack(accumulated_stabs).astype(np.int8)
                ## TODO: This is the wrong bit
                surviving_faults = raw_current.get_undetectable_faults(current_stabs_arr)
            else:
                surviving_faults = raw_current.faults
                
            if len(surviving_faults) == 0:
                next_beam_candidates.append((realized_cost, realized_cost, layers + [[]], accumulated_stabs))
                continue
                
            # Compute dynamic target coverage
            W_eff = compute_effective_weights(surviving_faults, H_filter)
            # layer_idx = number of faults - 1. So f_count = layer_idx + 1.
            f_count = layer_idx + 1
            target_coverage = np.clip(W_eff - f_count, 1, t)
                
            candidate_covers = _solve_top_n_weighted_set_covers(surviving_faults, stabs, t, max_combinations, max_time_sec, top_n, target_coverage=target_coverage)
            
            if not candidate_covers:
                next_beam_candidates.append((realized_cost, realized_cost, layers + [[]], accumulated_stabs))
                continue
                
            if layer_idx == t - 1:
                # Last layer: no lookahead needed
                best_last = candidate_covers[0]
                best_last_score = sum(cnot_cost(w, t) for w in np.sum(best_last, axis=1))
                final_cost = realized_cost + best_last_score
                next_beam_candidates.append((final_cost, final_cost, layers + [best_last], accumulated_stabs + best_last))
                continue
                
            for cover_idx, cover in enumerate(candidate_covers):
                test_stabs_list = accumulated_stabs + cover
                test_stabs_arr = np.vstack(test_stabs_list).astype(np.int8)
                
                raw_next = raw_fault_sets[layer_idx + 1]
                next_surviving = raw_next.get_undetectable_faults(test_stabs_arr)
                fs_size = len(next_surviving)
                
                next_cost = 0
                if fs_size > 0:
                    next_cov = ((candidate_stabs @ next_surviving.T) % 2).astype(bool)
                    coverable = np.any(next_cov, axis=0)
                    valid_cov = next_cov[:, coverable]
                    if valid_cov.shape[1] > 0:
                        chosen = fast_greedy_set_cover(valid_cov, costs)
                        next_cost = np.sum(costs[chosen])
                    uncoverable = fs_size - valid_cov.shape[1]
                    next_cost += uncoverable * 1000
                    
                cover_cost = sum(cnot_cost(w, t) for w in np.sum(cover, axis=1))
                new_realized_cost = realized_cost + cover_cost
                lookahead_score = new_realized_cost + next_cost
                
                next_beam_candidates.append((new_realized_cost, lookahead_score, layers + [cover], accumulated_stabs + cover))
                
        # Sort candidates by lookahead_score
        next_beam_candidates.sort(key=lambda x: x[1])
        
        # Deduplicate and form new beam
        unique_beam = []
        seen_sigs = set()
        for r_cost, l_score, lyrs, acc_stabs in next_beam_candidates:
            if not acc_stabs:
                sig = ()
            else:
                arr = np.vstack(acc_stabs)
                sig = hash(arr.tobytes())
                
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique_beam.append((r_cost, lyrs, acc_stabs))
                if len(unique_beam) >= beam_width:
                    break
                    
        beam = unique_beam
        
        if verbose:
            print(f"  -> Kept top {len(beam)} states. Best Lookahead Score: {next_beam_candidates[0][1] if layer_idx < t - 1 else beam[0][0]}")

    print(f"  [Chosen Stabilizers]")
    for i, layer in enumerate(beam[0][1]):
        print(f"    -> Layer {i + 1}: {["".join(map(str, stab.tolist())) for stab in layer]}")

    return beam[0][1]

def compute_bare_injected_faults(layers: list[list[np.ndarray]], num_qubits: int) -> PureFaultSet:
    from spiderstate.cat_at_origin import bare_se_circuit
    faults = []
    for layer in layers:
        for stab in layer:
            qubits = np.where(stab)[0].tolist()
            # SE circuit has CNOTs from ancilla to qubits
            for j in range(1, len(qubits)):
                err = np.zeros(num_qubits, dtype=np.int8)
                err[qubits[j:]] = 1
                faults.append(err)
    fs = PureFaultSet(num_qubits)
    if not faults:
        return fs
    faults_arr = np.array(faults, dtype=np.int8)
    fs.faults = np.unique(faults_arr, axis=0)
    return fs
