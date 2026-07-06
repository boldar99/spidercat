import logging
from pprint import pprint

import galois
from itertools import combinations
from typing import Callable
from scipy.spatial.distance import cdist

import numpy as np
from mqt.qecc.circuit_synthesis.faults import product_fault_set, PureFaultSet
from tqdm import tqdm

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
    
    GF2 = galois.GF(2)
    coverage = np.array(GF2(candidate_stabs) @ GF2(faults).T, dtype=bool)
    
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
            rest_idx = fast_greedy_set_cover(
                valid_cov, 
                costs, 
                uncovered,
                candidate_stabs=candidate_stabs,
                selected_stabs_indices=[start_idx]
            )
            
            # Accept partial covers because the beam search can finish them in later layers
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
    reps = compute_minimum_weight_representatives(faults, stabs)
    return np.sum(reps, axis=1)

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

class TrackedFaultSet:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.faults = np.empty((0, num_qubits), dtype=np.int8)
        self.original_reps = [] 
        self.ways_to_form = [] 
        self.fault_ids = []

    def __len__(self):
        return len(self.faults)

def compute_minimum_weight_representatives(faults: np.ndarray, stabs: np.ndarray) -> np.ndarray:
    if len(faults) == 0:
        return faults.copy()
    k = stabs.shape[0]
    if k == 0:
        return faults.copy()
        
    num_qubits = stabs.shape[1]

    print(f"{k=} and {faults.shape=}")

    # TODO: This is temporaraly 10 from 15 because the other method is faster,
    if k <= 10:
        all_stabs = []
        for i in range(1 << k):
            comb = [j for j in range(k) if (i >> j) & 1]
            if comb:
                s = np.bitwise_xor.reduce(stabs[comb], axis=0)
            else:
                s = np.zeros(num_qubits, dtype=np.int8)
            all_stabs.append(s)
        all_stabs = np.array(all_stabs, dtype=np.int8)

        batch_size = 2000
        reps = np.zeros_like(faults)
        for i in tqdm(range(0, len(faults), batch_size), desc="Finding minimum weight representatives of faults (using LUT)"):
            batch = faults[i:i+batch_size]

            # TODO: why do we need cdist? We have a complete list of stabilizers
            # We can turn the stabs to an into and index the list.
            dist = cdist(batch, all_stabs, metric='cityblock')
            min_idx = np.argmin(dist, axis=1)
            reps[i:i+batch_size] = (batch + all_stabs[min_idx]) % 2

        return reps

    GF2 = galois.GF(2)
    stabs_gf2 = GF2(stabs)
    H_gf2 = stabs_gf2.null_space()
    H = np.array(H_gf2, dtype=np.int8)
    
    if H.shape[0] == 0:
        return np.zeros_like(faults)

    syndromes = (faults @ H.T) % 2
    unique_syndromes = np.unique(syndromes, axis=0)
    target_syndromes = set(tuple(s) for s in unique_syndromes)
    
    found = {}
    from collections import deque
    queue = deque([(np.zeros(num_qubits, dtype=np.int8), 0, 0)])
    pbar = tqdm(total=len(target_syndromes), desc="Finding minimum weight representatives of faults (using galois)")
    
    while queue and len(found) < len(target_syndromes):
        vec, wt, last_idx = queue.popleft()
        syn = tuple((H @ vec) % 2)
        if syn in target_syndromes and syn not in found:
            found[syn] = vec
            pbar.update(1)
            
        for i in range(last_idx, num_qubits):
            new_vec = vec.copy()
            new_vec[i] ^= 1
            queue.append((new_vec, wt + 1, i + 1))
            
    reps = np.zeros_like(faults)
    for i, syn in enumerate(syndromes):
        reps[i] = found[tuple(syn)]
        
    return reps

def _generate_raw_fault_sets(single_faults: PureFaultSet, t: int, H_filter: np.ndarray) -> list[TrackedFaultSet]:
    """Generates the raw unfiltered unitary fault sets U_1, ..., U_t with full generation tracking."""
    fault_sets = []
    
    base_faults = single_faults.faults
    min_weight_reps = compute_minimum_weight_representatives(base_faults, H_filter)
    
    base_id_to_rep = {} 
    base_id_to_orig = {} 
    rep_to_base_id = {} 
    
    next_base_id = 0
    for i in range(len(base_faults)):
        rep = min_weight_reps[i]
        rep_bytes = rep.tobytes()
        if rep_bytes not in rep_to_base_id:
            rep_to_base_id[rep_bytes] = next_base_id
            base_id_to_rep[next_base_id] = rep
            base_id_to_orig[next_base_id] = base_faults[i]
            next_base_id += 1
            
    return fault_sets

def _generate_unified_active_errors(
    single_faults: PureFaultSet, 
    t: int, 
    H_filter: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    import itertools
    
    N = single_faults.num_qubits
    
    unique_components = []
    seen = set()
    
    for f in single_faults.faults:
        key = (f.tobytes(), 0)
        if key not in seen:
            seen.add(key)
            unique_components.append((f, 0))
            
    I_N = np.eye(N, dtype=np.int8)
    for L in range(1, t):
        for q in range(N):
            f = I_N[q]
            key = (f.tobytes(), L)
            if key not in seen:
                seen.add(key)
                unique_components.append((f, L))
                
    combinations_list = []
    final_data_list = []
    weights_list = []
    
    for k in range(1, t + 1):
        # TODO: Faster fault combinations and smaller footprint
        #
        # The current implementation uses itertools.combinations to generate any
        # combination of unique faults. For larger distances (>= 9) and large fault
        # sets very slow. It would be better to instead generate the combination of
        # k-faults from k-1 fault combined with 1 fault. Because we will have to find
        # the minimum weight of all faults, we also actually find their minimum weight
        # representative (MWR) afterwards. Therefore, it would be more efficient to
        # calculate the MWR for each fault as they are generated, and filter out
        # duplicates.
        for combo in tqdm(itertools.combinations(unique_components, k)):
            final_data = np.zeros(N, dtype=np.int8)
            for f, L in combo:
                final_data ^= f
            combinations_list.append(combo)
            final_data_list.append(final_data)
            weights_list.append(k)
            
    if len(combinations_list) == 0:
        return np.empty((0, t, N), dtype=np.int8), np.array([]), np.array([]), np.empty((0, N), dtype=np.int8), []
        
    final_data_array = np.array(final_data_list, dtype=np.int8)
    reps = compute_minimum_weight_representatives(final_data_array, H_filter)
    W_effs = np.sum(reps, axis=1)
    
    valid_active_errors = []
    valid_targets = []
    valid_Weffs = []
    valid_meta = []
    valid_final_data = []
    
    for i in range(len(combinations_list)):
        k = weights_list[i]
        W_eff = W_effs[i]
        req_det = min(W_eff - k, t - k + 1)
        if req_det <= 0:
            continue
            
        combo = combinations_list[i]
        active = np.zeros((t, N), dtype=np.int8)
        for f, L in combo:
            active[L:] ^= f
            
        valid_active_errors.append(active)
        valid_targets.append(req_det)
        valid_Weffs.append(W_eff)
        valid_meta.append({
            "components": combo,
            "weight": k,
            "final_data": final_data_list[i]
        })
        valid_final_data.append(final_data_list[i])
        
    if not valid_active_errors:
        return np.empty((0, t, N), dtype=np.int8), np.array([]), np.array([]), None, []
        
    return np.array(valid_active_errors), np.array(valid_targets), np.array(valid_Weffs), None, valid_meta

def find_lookahead_verification_stabilizers(
    single_faults: PureFaultSet,
    stabs: np.ndarray,
    H_filter: np.ndarray,
    t: int,
    max_combinations: int = 4,
    top_n: int = 50,
    beam_width: int = 5,
    max_time_sec: int = 60,
    verbose: bool = False,
    cost_fn: Callable[[int, int], float] | None = None
) -> list[list[np.ndarray]]:
    """
    Finds verification stabilizers for t layers using mathematical lookahead.
    Tracks exact fault detections across layers using a 2D signature matrix.
    """
    if cost_fn is None:
        cost_fn = cnot_cost

    candidate_stabs = _generate_candidate_stabilizers(stabs, max_combinations)
    costs = np.array([cost_fn(w, t) for w in np.sum(candidate_stabs, axis=1)])
    print("Found {} stabilizers".format(len(candidate_stabs)))

    # TODO: Cleaner fault management
    #
    # Because we have so many details we need to keep track of related to faults,
    # and also quite a lot of methods to deal with these, we should have a custom
    # class (FaultSet) that wraps all these together. This class should be in a
    # saparate file.
    active_errors, targets, W_effs, T_E_q_matrix, fault_meta = _generate_unified_active_errors(single_faults, t, H_filter)
    num_mixed_faults = len(active_errors)
    print("Found {} active errors".format(num_mixed_faults))
    
    GF2 = galois.GF(2)
    candidate_stabs_gf2 = GF2(candidate_stabs)
    active_errors_gf2 = GF2(active_errors)
    
    # beam state: (realized_cost, layers, accumulated_stabs, detection_counts)
    # detection_counts is a 1D array of shape (num_mixed_faults,)
    initial_detections = np.zeros(num_mixed_faults, dtype=int)
    beam = [(0.0, [], [], initial_detections)]

    # TODO: Somehow the first layer is always quite slow and then everythign else
    # is super speedy. Why is that? Can we do better? Is the whole computation
    # dependant on the first greedy cover?
    for layer_idx in range(t):
        next_beam_candidates = []
        
        if verbose:
            print(f"  [Layer {layer_idx + 1}] Expanding beam of size {len(beam)}:")
            
        for state_idx, (realized_cost, layers, accumulated_stabs, det_counts) in enumerate(beam):
            unsat_mask = det_counts < targets
            
            if not np.any(unsat_mask):
                next_beam_candidates.append((realized_cost, realized_cost, layers + [[]], accumulated_stabs, det_counts))
                continue
                
            # For evaluating candidate covers, we need the active faults in THIS layer
            # which is active_errors[:, layer_idx, :]
            active_threat_faults = active_errors[unsat_mask, layer_idx, :]
            
            # If the active faults are all zero, then these threats cannot be covered by THIS layer!
            nonzero_fault_mask = np.any(active_threat_faults, axis=1)
            
            if not np.all(nonzero_fault_mask):
                # There is an unsatisfied target, but its fault hasn't "occurred" yet or is zero in this layer.
                # Actually, if it's 0, we can just skip covering it in THIS layer.
                pass
                
            surviving_faults = active_threat_faults[nonzero_fault_mask]
            threat_targets = targets[unsat_mask][nonzero_fault_mask] - det_counts[unsat_mask][nonzero_fault_mask]
            capped_targets = np.minimum(threat_targets, 1)
            
            if len(surviving_faults) == 0:
                next_beam_candidates.append((realized_cost, realized_cost, layers + [[]], accumulated_stabs, det_counts))
                continue
                
            cov_matrix = np.array(candidate_stabs_gf2 @ GF2(surviving_faults).T, dtype=bool)
            num_uncoverable = np.sum(~np.any(cov_matrix, axis=0))
            realized_cost += num_uncoverable * 1000

            candidate_covers = _solve_top_n_weighted_set_covers(
                surviving_faults, stabs, t, max_combinations, max_time_sec, top_n, target_coverage=capped_targets
            )

            if not candidate_covers:
                penalty = 1000 * len(surviving_faults)
                next_beam_candidates.append((realized_cost + penalty, realized_cost + penalty, layers + [[]], accumulated_stabs, det_counts))
                continue
                
            if layer_idx == t - 1:
                # Last layer: no lookahead needed
                best_last = candidate_covers[0]
                best_last_score = sum(cost_fn(w, t) for w in np.sum(best_last, axis=1))
                if len(best_last) > 1:
                    N_q = np.sum(best_last, axis=0)
                    overlaps = np.sum(N_q > 1)
                    print(H_filter)
                    best_last_score += 2 * np.max(overlaps - H_filter.shape[1] // 10, 0)
                final_cost = realized_cost + best_last_score
                
                # Update detections
                layer_matrix = np.vstack(best_last).astype(np.int8)
                new_sigs = np.sum(np.array(active_errors_gf2[:, layer_idx, :] @ GF2(layer_matrix).T, dtype=np.int8), axis=1)
                new_det_counts = det_counts + new_sigs
                print()
                
                next_beam_candidates.append((final_cost, final_cost, layers + [best_last], accumulated_stabs + best_last, new_det_counts))
                continue

            print()
                
            for cover_idx, cover in enumerate(candidate_covers):
                cover_matrix = np.vstack(cover).astype(np.int8)
                
                new_sigs = np.sum(np.array(active_errors_gf2[:, layer_idx, :] @ GF2(cover_matrix).T, dtype=np.int8), axis=1)
                new_det_counts = det_counts + new_sigs
                
                # Lookahead scoring
                unsat_mask = new_det_counts < targets
                next_layer_idx = min(layer_idx + 1, t - 1)
                active_next = active_errors[unsat_mask, next_layer_idx, :]
                
                nonzero_fault_mask = np.any(active_next, axis=1)
                if np.any(nonzero_fault_mask):
                    next_surviving = active_next[nonzero_fault_mask]
                else:
                    num_qubits = active_errors.shape[2] if len(active_errors) > 0 else 0
                    next_surviving = np.zeros((0, num_qubits), dtype=np.int8)
                
                fs_size = len(next_surviving)
                
                next_cost = 0
                if fs_size > 0:
                    next_cov = np.array(candidate_stabs_gf2 @ GF2(next_surviving).T, dtype=bool)
                    coverable = np.any(next_cov, axis=0)
                    valid_cov = next_cov[:, coverable]
                    if valid_cov.shape[1] > 0:
                        chosen = fast_greedy_set_cover(valid_cov, costs, candidate_stabs=candidate_stabs)
                        next_cost = np.sum(costs[chosen])
                        
                        uncovered_after = np.ones(valid_cov.shape[1], dtype=int)
                        for idx in chosen:
                            uncovered_after[valid_cov[idx]] -= 1
                        missed_in_valid = np.sum(uncovered_after > 0)
                    else:
                        missed_in_valid = 0
                        
                        
                    uncoverable = fs_size - valid_cov.shape[1] + missed_in_valid
                    next_cost += uncoverable * 1000  # Severely penalize uncoverable faults
                    
                cover_cost = sum(cost_fn(w, t) for w in np.sum(cover, axis=1))
                if len(cover) > 1:
                    N_q = np.sum(cover, axis=0)
                    overlaps = np.sum(N_q > 1)
                    cover_cost += 2 * overlaps
                new_realized_cost = realized_cost + cover_cost
                lookahead_score = new_realized_cost + next_cost
                
                next_beam_candidates.append((new_realized_cost, lookahead_score, layers + [cover], accumulated_stabs + cover, new_det_counts))
                
        # Sort candidates by lookahead_score
        next_beam_candidates.sort(key=lambda x: x[1])
        
        # Deduplicate and form new beam
        unique_beam = []
        seen_sigs = set()
        for r_cost, l_score, lyrs, acc_stabs, det_counts in tqdm(next_beam_candidates):
            if not acc_stabs:
                sig = ()
            else:
                arr = np.vstack(acc_stabs)
                sig = hash(arr.tobytes())
                
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique_beam.append((r_cost, lyrs, acc_stabs, det_counts))
                if len(unique_beam) >= beam_width:
                    break
                    
        beam = unique_beam
        
        if verbose:
            print(f"  -> Kept top {len(beam)} states. Best Lookahead Score: {next_beam_candidates[0][1] if layer_idx < t - 1 else beam[0][0]}")

    # SCORING & PRUNING
    valid_beam = []
    from spiderstate.cnot_scheduler import DangerousFault, schedule_all_verification_layers
    num_qubits = active_errors.shape[2] if len(active_errors) > 0 else 0

    # Precompute T_E_Q for unique final_data and k combinations ONCE before scoring candidates
    if verbose:
        print("Computing T_E_Q...")
    unique_final_data_keys = {}
    for i in range(num_mixed_faults):
        T_E = targets[i]
        if T_E <= 0:
            continue
        k = fault_meta[i]["weight"]
        final_data = fault_meta[i]["final_data"]
        key = (tuple(final_data), k, T_E)
        if key not in unique_final_data_keys:
            unique_final_data_keys[key] = []
        unique_final_data_keys[key].append(i)
        
    precomputed_T_E_Q = {}
    import itertools
    
    all_combs_list = []
    key_to_slices = {}
    current_idx = 0
    
    for key in unique_final_data_keys:
        final_data_tuple, k, T_E = key
        final_data = np.array(final_data_tuple, dtype=np.int8)
        T_E_Q = {(): int(T_E)}
        
        key_combs = []
        combs_meta = []
        for size in range(1, t - k + 1):
            combs = list(itertools.combinations(range(num_qubits), size))
            if not combs:
                continue
            combs_array = np.zeros((len(combs), num_qubits), dtype=np.int8)
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
        giant_reps = compute_minimum_weight_representatives(giant_combs_array, H_filter)
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
                t_e_combs = np.minimum(weff_combs - (k + size), t - (k + size) + 1)
                for c_idx, comb in enumerate(combs):
                    T_E_Q[comb] = int(t_e_combs[c_idx])
                offset += n_c
            precomputed_T_E_Q[key] = T_E_Q

    if verbose:
        print("Finding the best circuits in the beam...")
    for cand in beam:
        det_counts = cand[3]
        chosen_layers = cand[1]
        
        target_coverage_violations = []
        dangerous_faults = []
        
        unsat_mask = det_counts < targets
        if np.any(unsat_mask):
            for idx in np.where(unsat_mask)[0]:
                q_to_flag = np.nonzero(fault_meta[idx]["final_data"])[0]
                if len(q_to_flag) > 0:
                    target_coverage_violations.append({
                        "layer": 0, 
                        "q": int(q_to_flag[0]), 
                        "s_j": -1, 
                        "violation_amount": int(targets[idx] - det_counts[idx]),
                        "type": "target_coverage"
                    })
            
        # Precompute syndromes for ALL faults across ALL chosen layers in a vectorized manner
        D_E_all = [set() for _ in range(num_mixed_faults)]
        D_eq_L_E_counts_all = np.zeros((num_mixed_faults, t), dtype=int)
        
        for l_idx, layer in enumerate(chosen_layers):
            if len(layer) == 0:
                continue
            layer_matrix = np.vstack(layer).astype(np.int8)
            syn_all = np.array(active_errors_gf2[:, l_idx, :] @ GF2(layer_matrix).T, dtype=np.int8)
            
            fault_indices, stab_indices = np.nonzero(syn_all)
            for f_idx, s_idx in zip(fault_indices, stab_indices):
                D_E_all[f_idx].add((l_idx, s_idx))
                D_eq_L_E_counts_all[f_idx, l_idx] += 1
                
        for i in range(num_mixed_faults):
            T_E = targets[i]
            if T_E <= 0:
                continue
                
            D_E = D_E_all[i]
            D_eq_L_E_counts = D_eq_L_E_counts_all[i]
                        
            key = (tuple(fault_meta[i]["final_data"]), fault_meta[i]["weight"], T_E)
            T_E_Q = precomputed_T_E_Q[key]
            
            # Filter: we only need to pass this fault if there is SOME layer L and SOME Q 
            # where M_E_Q < |D_{=L}(E)|.
            # M_E_Q = |D_E| - |D_{>L}(E)| + |syn_{>L}(E \oplus Q)| - T_E_Q
            # |syn_{>L}(E \oplus Q)| = |D_{>L}(E) \oplus syn_{>L}(Q)|
            
            is_dangerous = False
            for L in range(len(chosen_layers)):
                abs_D_eq_L = D_eq_L_E_counts[L]
                abs_D_gt_L = sum(D_eq_L_E_counts[L+1:])
                
                # Check if ANY Q violates M_E_Q >= |D_{=L}|
                for Q, req in T_E_Q.items():
                    # Calculate syn_{>L}(Q)
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
                    
        unique_dfs = [DangerousFault(D_E, T_E_Q) for D_E, T_E_Q in df_dict.items()]
        
        layers_qubits = []
        for layer in chosen_layers:
            stabs_qubits = []
            for stab in layer:
                stabs_qubits.append(np.nonzero(stab)[0].tolist())
            layers_qubits.append(stabs_qubits)
            
        try:
            ticks, sched_violations = schedule_all_verification_layers(layers_qubits, unique_dfs)
            all_violations = target_coverage_violations + sched_violations
            valid_beam.append({
                "cand": cand,
                "chosen_layers": chosen_layers,
                "det_counts": det_counts,
                "unique_dfs": unique_dfs,
                "ticks": ticks,
                "violations": all_violations
            })
        except RuntimeError:
            pass # Failed to schedule, try next state
            
    if not valid_beam:
        raise RuntimeError("No states in the beam could be scheduled safely! Try increasing max_combinations or top_n.")
        
    # Sort valid beam by total score: realized_cost + 1000 * number of flags
    # print([x["cand"] for x in valid_beam])
    valid_beam.sort(key=lambda x: x["cand"][0] + 1000 * len(x["violations"]))
    best_state = valid_beam[0]
    
    if verbose:
        print(f"  [Chosen Stabilizers] (Violations: {len(best_state['violations'])})")
        for i, layer in enumerate(best_state["chosen_layers"]):
            print(f"    -> Layer {i + 1}: {[''.join(map(str, stab.tolist())) for stab in layer]}")
            
    return best_state["chosen_layers"], best_state["unique_dfs"], best_state["ticks"], best_state["violations"]

def compute_bare_injected_faults(layers: list[list[np.ndarray]], ticks_layers: list[list[list[tuple[int, int]]]], num_qubits: int) -> PureFaultSet:
    faults = []
    for layer, ticks in zip(layers, ticks_layers):
        # Build the ordered list of data qubits for each stabilizer
        ordered_qubits_by_stab = [[] for _ in range(len(layer))]
        for tick_ops in ticks:
            for stab_idx, q in tick_ops:
                ordered_qubits_by_stab[stab_idx].append(q)
                
        # Generate the suffix faults for each stabilizer
        for stab_idx, ordered_qubits in enumerate(ordered_qubits_by_stab):
            for j in range(1, len(ordered_qubits)):
                err = np.zeros(num_qubits, dtype=np.int8)
                err[ordered_qubits[j:]] = 1
                faults.append(err)
                
    fs = PureFaultSet(num_qubits)
    if not faults:
        return fs
    faults_arr = np.array(faults, dtype=np.int8)
    fs.faults = np.unique(faults_arr, axis=0)
    return fs
