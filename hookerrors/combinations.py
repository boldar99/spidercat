import itertools
import numpy as np

def check_combination(splits, n, t, k, strategy, L_cosets):
    E_comb = np.zeros(n, dtype=int)
    for s in splits:
        E_comb[list(s)] ^= 1
        
    # Condition 1: w_min* <= k (Trivially Safe)
    # The internal faults consumed <= k distance.
    if strategy.check_tier1(E_comb, threshold=k):
        return True
        
    # Condition 2: min_logical >= 2*t - k + 1 (Decoder-Benign)
    # The decoder will not mistakenly apply a logical error.
    if len(L_cosets) == 0:
        return True
    if strategy.check_tier3(E_comb, L_cosets, threshold=2*t - k + 1):
        return True
        
    return False

def find_globally_safe_assignment(gens, safe_candidates_per_gen, n, t, strategy, L_cosets):
    """
    Finds a globally safe assignment of MULTI-SPLITS (chains of safe splits) 
    that maximizes the total number of internal edges.
    """
    best_assignment = {}
    
    # Precompute the maximal multisplit for each generator
    max_chains = {}
    for g in gens:
        chain = find_maximal_multisplit(None, safe_candidates_per_gen[g])
        if chain:
            max_chains[g] = chain
            
    # Greedily build the global assignment
    current_splits = []
    
    # Sort generators by chain length (descending) to prioritize leanest decompositions
    sorted_gens = sorted(max_chains.keys(), key=lambda g: len(max_chains[g]), reverse=True)
    
    for g in sorted_gens:
        candidate_chain = max_chains[g]
        is_compatible = True
        
        # Test if adding this entire chain is safe with current_splits
        new_splits = current_splits + candidate_chain
        
        # We need to check all combinations of size k <= t in new_splits
        # to ensure they are jointly safe.
        for k in range(2, t + 1):
            if len(new_splits) < k:
                continue
            
            # To optimize, we only check combinations that include at least one element from candidate_chain
            # (since combinations purely from current_splits are already verified)
            for subset in itertools.combinations(new_splits, k):
                if any(x in candidate_chain for x in subset):
                    if not check_combination(subset, n, t, k, strategy, L_cosets):
                        is_compatible = False
                        break
            if not is_compatible:
                break
                
        if is_compatible:
            best_assignment[g] = candidate_chain
            current_splits.extend(candidate_chain)
            
    return best_assignment

def find_maximal_multisplit(support, safe_splits_list):
    """
    Given a generator's support and a list of its safe splits (tuples of ints),
    finds the longest chain of nested safe splits (S_1 subset S_2 subset ... subset S_k).
    Returns the chain as a list of tuples.
    """
    if not safe_splits_list:
        return []
        
    # Sort splits by length to ensure topological ordering
    splits = sorted([set(s) for s in safe_splits_list], key=len)
    n = len(splits)
    
    dp = [1] * n
    prev = [-1] * n
    
    for i in range(n):
        for j in range(i):
            # If splits[j] is a strict subset of splits[i]
            if splits[j].issubset(splits[i]) and len(splits[j]) < len(splits[i]):
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j
                    
    max_len = 0
    best_end = -1
    for i in range(n):
        if dp[i] > max_len:
            max_len = dp[i]
            best_end = i
            
    chain = []
    curr = best_end
    while curr != -1:
        # Convert back to sorted tuples
        chain.append(tuple(sorted(list(splits[curr]))))
        curr = prev[curr]
        
    return chain[::-1]
