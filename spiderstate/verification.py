import logging
from itertools import combinations
from typing import Callable

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

    if k <= 4:
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
        for i in range(0, len(faults), batch_size):
            batch = faults[i:i+batch_size]
            candidate = (batch[:, None, :] + all_stabs[None, :, :]) % 2
            cand_weights = np.sum(candidate, axis=2)
            min_idx = np.argmin(cand_weights, axis=1)
            reps[i:i+batch_size] = candidate[np.arange(len(batch)), min_idx]
            
        return reps

    import galois
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
    
    while queue and len(found) < len(target_syndromes):
        vec, wt, last_idx = queue.popleft()
        syn = tuple((H @ vec) % 2)
        if syn in target_syndromes and syn not in found:
            found[syn] = vec
            
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
        for combo in itertools.combinations(unique_components, k):
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
        return np.empty((0, t, N), dtype=np.int8), np.array([]), np.array([]), np.empty((0, N), dtype=np.int8), []
        
    valid_final_data_array = np.array(valid_final_data, dtype=np.int8)
    M = len(valid_active_errors)
    E_oplus_q = np.zeros((M, N, N), dtype=np.int8)
    for q in range(N):
        E_oplus_q[:, q, :] = valid_final_data_array
        E_oplus_q[:, q, q] ^= 1
        
    E_oplus_q_flat = E_oplus_q.reshape(-1, N)
    reps_q = compute_minimum_weight_representatives(E_oplus_q_flat, H_filter)
    W_eff_q_flat = np.sum(reps_q, axis=1)
    W_eff_matrix = W_eff_q_flat.reshape(M, N)
    
    valid_weights = np.array([m["weight"] for m in valid_meta])
    T_E_q_matrix = np.minimum(W_eff_matrix - (valid_weights[:, None] + 1), t - (valid_weights[:, None] + 1) + 1)
    
    return np.array(valid_active_errors), np.array(valid_targets), np.array(valid_Weffs), T_E_q_matrix, valid_meta

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
    
    active_errors, targets, W_effs, T_E_q_matrix, fault_meta = _generate_unified_active_errors(single_faults, t, H_filter)
    num_mixed_faults = len(active_errors)
    
    # beam state: (realized_cost, layers, accumulated_stabs, detection_counts)
    # detection_counts is a 1D array of shape (num_mixed_faults,)
    initial_detections = np.zeros(num_mixed_faults, dtype=int)
    beam = [(0.0, [], [], initial_detections)]
    
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
                
            cov_matrix = ((candidate_stabs @ surviving_faults.T) % 2).astype(bool)
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
                final_cost = realized_cost + best_last_score
                
                # Update detections
                layer_matrix = np.vstack(best_last).astype(np.int8)
                new_sigs = np.sum((active_errors[:, layer_idx, :] @ layer_matrix.T) % 2, axis=1)
                new_det_counts = det_counts + new_sigs
                
                next_beam_candidates.append((final_cost, final_cost, layers + [best_last], accumulated_stabs + best_last, new_det_counts))
                continue
                
            for cover_idx, cover in enumerate(candidate_covers):
                cover_matrix = np.vstack(cover).astype(np.int8)
                
                new_sigs = np.sum((active_errors[:, layer_idx, :] @ cover_matrix.T) % 2, axis=1)
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
                    next_cov = ((candidate_stabs @ next_surviving.T) % 2).astype(bool)
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
                    next_cost += uncoverable * 2  # Penalize uncoverable faults by 2 CNOTs (1 flag)
                    
                cover_cost = sum(cost_fn(w, t) for w in np.sum(cover, axis=1))
                new_realized_cost = realized_cost + cover_cost
                lookahead_score = new_realized_cost + next_cost
                
                next_beam_candidates.append((new_realized_cost, lookahead_score, layers + [cover], accumulated_stabs + cover, new_det_counts))
                
        # Sort candidates by lookahead_score
        next_beam_candidates.sort(key=lambda x: x[1])
        
        # Deduplicate and form new beam
        unique_beam = []
        seen_sigs = set()
        for r_cost, l_score, lyrs, acc_stabs, det_counts in next_beam_candidates:
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
                    
        for i in range(num_mixed_faults):
            T_E = targets[i]
            if T_E <= 0:
                continue
                
            D_E = set()
            for l_idx, layer in enumerate(chosen_layers):
                if len(layer) == 0:
                    continue
                layer_matrix = np.vstack(layer).astype(np.int8)
                syn = (active_errors[i, l_idx, :] @ layer_matrix.T) % 2
                for s_idx, flipped in enumerate(syn):
                    if flipped:
                        D_E.add((l_idx, s_idx))
                        
            if len(D_E) > 0:
                T_E_q = {q: T_E_q_matrix[i, q] for q in range(num_qubits)}
                dangerous_faults.append(DangerousFault(frozenset(D_E), T_E_q))
                    
        df_dict = {}
        for df in dangerous_faults:
            if df.D_E not in df_dict:
                df_dict[df.D_E] = df.T_E_q.copy()
            else:
                for q in df_dict[df.D_E]:
                    df_dict[df.D_E][q] = max(df_dict[df.D_E][q], df.T_E_q[q])
                    
        unique_dfs = [DangerousFault(D_E, T_E_q) for D_E, T_E_q in df_dict.items()]
        
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
        
    # Sort valid beam by total score: realized_cost + 2 * number of flags
    valid_beam.sort(key=lambda x: x["cand"][0] + 2 * len(x["violations"]))
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
