import logging
from itertools import combinations

import numpy as np
import z3
from mqt.qecc.circuit_synthesis.faults import PureFaultSet, product_fault_set

logger = logging.getLogger(__name__)

def _generate_candidate_stabilizers(stabs: np.ndarray, max_combinations: int) -> np.ndarray:
    k = stabs.shape[0]
    candidate_stabs = []
    
    if k <= 12:
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

def _solve_top_n_weighted_set_covers(faults: np.ndarray, stabs: np.ndarray, max_combinations: int, max_time_sec: int, top_n: int = 5) -> list[list[np.ndarray]]:
    faults = faults[np.any(faults, axis=1)]
    if len(faults) == 0:
        return []
        
    candidate_stabs = _generate_candidate_stabilizers(stabs, max_combinations)
    if len(candidate_stabs) == 0:
        return []
        
    coverage = (candidate_stabs @ faults.T) % 2
    weights = np.sum(candidate_stabs, axis=1)
    
    num_candidates = candidate_stabs.shape[0]
    num_faults = faults.shape[0]
    
    coverable = np.any(coverage, axis=0)
    coverable_idx = np.where(coverable)[0]
    
    if len(coverable_idx) == 0:
        return []
        
    opt = z3.Optimize()
    opt.set("timeout", max_time_sec * 1000)
    
    x = [z3.Int(f"x_{i}") for i in range(num_candidates)]
    for i in range(num_candidates):
        opt.add(z3.Or(x[i] == 0, x[i] == 1))
        
    for j in coverable_idx:
        covering_candidates = np.where(coverage[:, j])[0]
        opt.add(z3.Sum([x[i] for i in covering_candidates]) >= 1)
        
    cost_expr = z3.Sum([x[i] * (int(weights[i]) * 10 + 1) for i in range(num_candidates)])
    opt.minimize(cost_expr)
    
    results = []
    
    for _ in range(top_n):
        res = opt.check()
        if res == z3.sat:
            model = opt.model()
            selected_idx = [i for i in range(num_candidates) if model.evaluate(x[i]).as_long() == 1]
            results.append([candidate_stabs[i] for i in selected_idx])
            
            # Block this exact solution from being found again
            block_cond = []
            for i in range(num_candidates):
                val = 1 if i in selected_idx else 0
                block_cond.append(x[i] != val)
            opt.add(z3.Or(block_cond))
        else:
            break
            
    if not results:
        logger.warning("Z3 optimizer failed or timed out. Falling back to greedy set cover.")
        greedy_res = _greedy_set_cover(coverage[:, coverable_idx], weights, candidate_stabs)
        if greedy_res:
            results.append(greedy_res)
            
    return results

def find_low_weight_verification_stabilizers(fault_sets: list[PureFaultSet], stabs: np.ndarray, max_combinations: int = 4, max_time_sec: int = 60) -> list[list[np.ndarray]]:
    logger.info("Finding low-weight verification stabilizers using Z3 ILP")
    n_layers = len(fault_sets)
    layers: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for num_errors in range(n_layers):
        faults_obj = fault_sets[num_errors]
        if len(faults_obj) == 0:
            layers[num_errors] = []
            continue
        covers = _solve_top_n_weighted_set_covers(faults_obj.faults, stabs, max_combinations, max_time_sec, top_n=1)
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
    top_n: int = 10,
    max_time_sec: int = 60,
    verbose: bool = False
) -> list[list[np.ndarray]]:
    """
    Finds verification stabilizers for t layers using mathematical lookahead.
    It minimizes the size of the next layer's raw fault set after filtering.
    """
    layers = []
    accumulated_stabs = []
    
    # Precompute raw U_1, ..., U_t
    raw_fault_sets = _generate_raw_fault_sets(single_faults, t, H_filter)
    
    for layer_idx in range(t):
        raw_current = raw_fault_sets[layer_idx]
        
        # Filter raw_current against all accumulated stabilizers so far
        if accumulated_stabs:
            current_stabs_arr = np.vstack(accumulated_stabs).astype(np.int8)
            surviving_faults = raw_current.get_undetectable_faults(current_stabs_arr)
        else:
            surviving_faults = raw_current.faults
            
        if len(surviving_faults) == 0:
            layers.append([])
            continue
            
        # Get top N covers for the surviving current faults
        candidate_covers = _solve_top_n_weighted_set_covers(surviving_faults, stabs, max_combinations, max_time_sec, top_n)
        
        if not candidate_covers:
            layers.append([])
            continue
            
        if layer_idx == t - 1:
            # Last layer: no lookahead needed, just take the minimum weight cover
            layers.append(candidate_covers[0])
            break
            
        best_cover = None
        min_next_fault_size = float('inf')
        
        if verbose:
            print(f"  [Layer {layer_idx + 1}] Evaluating {len(candidate_covers)} candidate covers:")
            
        for cover_idx, cover in enumerate(candidate_covers):
            # Evaluate how this cover (combined with previous) reduces U_{k+1}
            test_stabs_list = accumulated_stabs + cover
            test_stabs_arr = np.vstack(test_stabs_list).astype(np.int8)
            
            raw_next = raw_fault_sets[layer_idx + 1]
            next_surviving = raw_next.get_undetectable_faults(test_stabs_arr)
            fs_size = len(next_surviving)
            
            if verbose:
                w_sum = sum(np.sum(s) for s in cover)
                if fs_size > 0:
                    f_weights = np.sum(next_surviving, axis=1)
                    max_w = int(np.max(f_weights))
                    counts = {w: int(np.sum(f_weights == w)) for w in range(1, max_w + 1) if np.sum(f_weights == w) > 0}
                    print(f"    - Cover {cover_idx} (Total Weight: {w_sum}): Next Fault Set Size = {fs_size}, Fault Weights: {counts}")
                else:
                    print(f"    - Cover {cover_idx} (Total Weight: {w_sum}): Next Fault Set Size = 0")
            
            if fs_size < min_next_fault_size:
                min_next_fault_size = fs_size
                best_cover = cover
                
        if verbose:
            best_idx = next(i for i, c in enumerate(candidate_covers) if c is best_cover)
            print(f"  -> Selected Cover {best_idx} with Next Fault Set Size {min_next_fault_size}")
            
        layers.append(best_cover)
        accumulated_stabs.extend(best_cover)
        
    return layers
