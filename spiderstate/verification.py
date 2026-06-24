import logging
from itertools import combinations

import numpy as np
from mqt.qecc.circuit_synthesis.faults import product_fault_set, PureFaultSet
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
            
            # Check if we fully covered the faults. If not, the overlap constraint made it impossible.
            uncovered_after = uncovered.copy()
            for idx in rest_idx:
                uncovered_after[valid_cov[idx]] -= 1
            if np.any(uncovered_after > 0):
                continue

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
    k = stabs.shape[0]
    if k == 0:
        return np.sum(faults, axis=1)
        
    all_stabs = []
    for i in range(1 << k):
        comb = [j for j in range(k) if (i >> j) & 1]
        if comb:
            s = np.bitwise_xor.reduce(stabs[comb], axis=0)
        else:
            s = np.zeros(stabs.shape[1], dtype=np.int8)
        all_stabs.append(s)
    all_stabs = np.array(all_stabs, dtype=np.int8)
    
    batch_size = 2000
    weights = np.zeros(len(faults), dtype=int)
    for i in range(0, len(faults), batch_size):
        batch = faults[i:i+batch_size]
        candidate = (batch[:, None, :] + all_stabs[None, :, :]) % 2
        cand_weights = np.sum(candidate, axis=2)
        weights[i:i+batch_size] = np.min(cand_weights, axis=1)
        
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
    Tracks exact fault detections across layers using a 2D signature matrix.
    """
    candidate_stabs = _generate_candidate_stabilizers(stabs, max_combinations)
    costs = np.array([cnot_cost(w, t) for w in np.sum(candidate_stabs, axis=1)])
    
    raw_fault_sets = _generate_raw_fault_sets(single_faults, t, H_filter)
    
    # Precompute target coverage for each R_k pool
    targets = []
    for layer_idx, fs in enumerate(raw_fault_sets):
        if len(fs.faults) == 0:
            targets.append(np.array([]))
            continue
        W_eff = compute_effective_weights(fs.faults, H_filter)
        f_count = layer_idx + 1
        target_coverage = np.minimum(W_eff - f_count, t - f_count + 1)
        targets.append(target_coverage)
    
    # beam state: (realized_cost, layers, accumulated_stabs, detection_counts)
    # detection_counts is a list of 2D arrays, one for each R_k
    initial_detections = [np.zeros((len(fs.faults), 0), dtype=np.int8) for fs in raw_fault_sets]
    beam = [(0.0, [], [], initial_detections)]
    
    for layer_idx in range(t):
        raw_current = raw_fault_sets[layer_idx]
        target_current = targets[layer_idx]
        
        next_beam_candidates = []
        
        if verbose:
            print(f"  [Layer {layer_idx + 1}] Expanding beam of size {len(beam)}:")
            
        for state_idx, (realized_cost, layers, accumulated_stabs, det_counts) in enumerate(beam):
            det_current = det_counts[layer_idx]
            
            if len(raw_current.faults) == 0:
                next_beam_candidates.append((realized_cost, realized_cost, layers + [[]], accumulated_stabs, det_counts))
                continue
                
            total_detections = np.sum(det_current, axis=1)
            unsatisfied_mask = total_detections < target_current
            surviving_faults = raw_current.faults[unsatisfied_mask]
            
            if len(surviving_faults) == 0:
                next_beam_candidates.append((realized_cost, realized_cost, layers + [[]], accumulated_stabs, det_counts))
                continue
                
            threat_targets = target_current[unsatisfied_mask] - total_detections[unsatisfied_mask]
            
            cov_matrix = ((candidate_stabs @ surviving_faults.T) % 2).astype(bool)
            num_uncoverable = np.sum(~np.any(cov_matrix, axis=0))
            realized_cost += num_uncoverable * 1000
                
            candidate_covers = _solve_top_n_weighted_set_covers(
                surviving_faults, stabs, t, max_combinations, max_time_sec, top_n, target_coverage=threat_targets
            )
            
            if not candidate_covers:
                penalty = 1000 * len(surviving_faults)
                next_beam_candidates.append((realized_cost + penalty, realized_cost + penalty, layers + [[]], accumulated_stabs, det_counts))
                continue
                
            if layer_idx == t - 1:
                # Last layer: no lookahead needed
                best_last = candidate_covers[0]
                best_last_score = sum(cnot_cost(w, t) for w in np.sum(best_last, axis=1))
                final_cost = realized_cost + best_last_score
                
                # Update detections
                layer_matrix = np.vstack(best_last).astype(np.int8)
                new_det_counts = []
                for k, det in enumerate(det_counts):
                    new_sigs = ((layer_matrix @ raw_fault_sets[k].faults.T) % 2).T.astype(np.int8)
                    new_det_counts.append(np.hstack((det, new_sigs)))
                
                next_beam_candidates.append((final_cost, final_cost, layers + [best_last], accumulated_stabs + best_last, new_det_counts))
                continue
                
            for cover_idx, cover in enumerate(candidate_covers):
                cover_matrix = np.vstack(cover).astype(np.int8)
                
                new_det_counts = []
                for k, det in enumerate(det_counts):
                    new_sigs = ((cover_matrix @ raw_fault_sets[k].faults.T) % 2).T.astype(np.int8)
                    new_det_counts.append(np.hstack((det, new_sigs)))
                
                # Lookahead scoring
                raw_next = raw_fault_sets[layer_idx + 1]
                target_next = targets[layer_idx + 1]
                
                if len(raw_next.faults) > 0:
                    det_next = new_det_counts[layer_idx + 1]
                    total_next = np.sum(det_next, axis=1)
                    unsat_next_mask = total_next < target_next
                    next_surviving = raw_next.faults[unsat_next_mask]
                else:
                    next_surviving = np.zeros((0, raw_next.num_qubits), dtype=np.int8)
                
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
                    next_cost += uncoverable * 1000
                    
                cover_cost = sum(cnot_cost(w, t) for w in np.sum(cover, axis=1))
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

    # STRICT PRUNING
    valid_beam = []
    from spiderstate.cnot_scheduler import DangerousFault, schedule_all_verification_layers
    num_qubits = raw_fault_sets[0].num_qubits if raw_fault_sets else 0
    I_N = np.eye(num_qubits, dtype=np.int8)
    
    for cand in beam:
        det_counts = cand[3]
        chosen_layers = cand[1]
        
        is_valid = True
        for layer_idx, target_coverage in enumerate(targets):
            if len(target_coverage) == 0: continue
            total_det = np.sum(det_counts[layer_idx], axis=1)
            if np.any(total_det < target_coverage):
                is_valid = False
                break
        if not is_valid:
            continue
            
        dangerous_faults = []
        for layer_idx, fs in enumerate(raw_fault_sets):
            k = layer_idx + 1
            target_coverage = targets[layer_idx]
            if len(target_coverage) == 0:
                continue
                
            E_oplus_q_flat = (fs.faults[:, None, :] + I_N[None, :, :]) % 2
            E_oplus_q_flat = E_oplus_q_flat.reshape(-1, num_qubits)
            W_eff_flat = compute_effective_weights(E_oplus_q_flat, H_filter)
            W_eff_matrix = W_eff_flat.reshape(len(fs.faults), num_qubits)
            T_E_q_matrix = np.minimum(W_eff_matrix - (k + 1), t - (k + 1) + 1)
            
            for i, f in enumerate(fs.faults):
                T_E = target_coverage[i]
                if T_E <= 0:
                    continue
                    
                D_E = set()
                col_idx = 0
                for l_idx, layer in enumerate(chosen_layers):
                    for s_idx in range(len(layer)):
                        if det_counts[layer_idx][i, col_idx] == 1:
                            D_E.add((l_idx, s_idx))
                        col_idx += 1
                        
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
            ticks, violations = schedule_all_verification_layers(layers_qubits, unique_dfs)
            valid_beam.append({
                "cand": cand,
                "chosen_layers": chosen_layers,
                "det_counts": det_counts,
                "unique_dfs": unique_dfs,
                "ticks": ticks,
                "violations": violations
            })
        except RuntimeError:
            pass # Failed to schedule, try next state
            
    if not valid_beam:
        raise RuntimeError("Strict Pruning: No states in the beam satisfied the target coverage AND could be scheduled safely! Try increasing max_combinations or top_n.")
        
    # Sort valid beam by number of violations (minimize), then by lookahead score (maximize, negative because we sort ascending)
    valid_beam.sort(key=lambda x: (len(x["violations"]), -x["cand"][0]))
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
