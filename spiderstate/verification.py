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

def compute_next_unitary_fault_set(single_faults: PureFaultSet, last_faults: PureFaultSet, measured_stabs: list[np.ndarray], weight: int, H_filter: np.ndarray) -> PureFaultSet:
    next_faults = product_fault_set(last_faults, single_faults)
    double_last = product_fault_set(last_faults, last_faults)
    
    combined_faults = np.concatenate((double_last.faults, next_faults.faults))
    next_faults.faults = np.unique(combined_faults, axis=0)
    
    if len(measured_stabs) > 0:
        stabs_arr = np.array(measured_stabs, dtype=np.int8)
        # We want to keep faults that are NOT detected by the measured stabilizers.
        # A fault is undetected if it commutes with all measured stabilizers.
        undetected_faults = next_faults.get_undetectable_faults(stabs_arr)
        next_faults.faults = undetected_faults
        
    next_faults.faults = np.concatenate((last_faults.faults, next_faults.faults))
    next_faults.remove_zero_rows()
    next_faults.remove_duplicates()
    next_faults.filter_by_weight_at_least(weight, H_filter)
    return next_faults

def find_lookahead_verification_stabilizers(
    single_faults: PureFaultSet,
    stabs: np.ndarray,
    H_filter: np.ndarray,
    t: int,
    max_combinations: int = 4,
    top_n: int = 25,
    max_time_sec: int = 1,
    verbose: bool = False
) -> list[list[np.ndarray]]:
    """
    Finds verification stabilizers for t layers. At each layer, it evaluates the top N covers
    and chooses the one that minimizes the size of the *next* layer's fault set.
    """
    layers = []
    
    # Layer 1 baseline faults
    current_faults = single_faults.copy()
    current_faults.filter_by_weight_at_least(2, H_filter)
    last_layer_faults = current_faults
    
    for layer_idx in range(1, t + 1):
        if len(current_faults) == 0:
            layers.append([])
            continue
            
        # 1. Get top N covers for current_faults
        candidate_covers = _solve_top_n_weighted_set_covers(current_faults.faults, stabs, max_combinations, max_time_sec, top_n)
        
        if not candidate_covers:
            layers.append([])
            continue
            
        if layer_idx == t:
            # For the last layer, lookahead doesn't matter, just take the minimum cost one (the first one)
            layers.append(candidate_covers[0])
            break
            
        best_cover = None
        min_next_fault_size = float('inf')
        
        if verbose:
            print(f"  [Layer {layer_idx}] Evaluating {len(candidate_covers)} candidate covers:")
            
        for cover_idx, cover in enumerate(candidate_covers):
            # 2. Simulate next fault set
            weight_threshold = layer_idx + 2
            next_fs = compute_next_unitary_fault_set(single_faults, current_faults, cover, weight_threshold, H_filter)
            fs_size = len(next_fs)
            
            if verbose:
                w_sum = sum(np.sum(s) for s in cover)
                print(f"    - Cover {cover_idx} (Total Weight: {w_sum}): Next Fault Set Size = {fs_size}")
            
            if fs_size < min_next_fault_size:
                min_next_fault_size = fs_size
                best_cover = cover
                
        if verbose:
            best_idx = next(i for i, c in enumerate(candidate_covers) if c is best_cover)
            print(f"  -> Selected Cover {best_idx} with Next Fault Set Size {min_next_fault_size}")
            
        layers.append(best_cover)
        
        # Advance to the next layer using the best cover
        current_faults = compute_next_unitary_fault_set(single_faults, current_faults, best_cover, layer_idx + 2, H_filter)
        
    return layers
