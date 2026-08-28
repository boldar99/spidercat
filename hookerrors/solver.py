import numpy as np
import itertools
import sys
import os
import argparse
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spiderstate.utils import load_qecc
from hookerrors.filters import MILPStrategy, LookupStrategy, TieredStrategy, HeuristicOnlyStrategy
from hookerrors.searchers import ExhaustiveSearcher, EarlyExitSearcher
from hookerrors.combinations import find_globally_safe_assignment

def analyze_hook_errors(H_x, H_z, L_x, L_z, d, prep_basis='Z', method='tiered', searcher_type='exhaustive', max_splits=1, max_weight=None, find_assignment=False):
    n = H_x.shape[1]
    k_x = L_x.shape[0] if len(L_x) > 0 else 0
    k_z = L_z.shape[0] if len(L_z) > 0 else 0
    t = (d - 1) // 2
    
    if prep_basis == 'Z':
        Mx_prep = H_x
        Mz_prep = np.vstack([H_z, L_z]) if len(L_z) > 0 else H_z
        L_check_x = L_x
        L_check_z = np.zeros((0, n), dtype=int)
    elif prep_basis == 'X':
        Mx_prep = np.vstack([H_x, L_x]) if len(L_x) > 0 else H_x
        Mz_prep = H_z
        L_check_x = np.zeros((0, n), dtype=int)
        L_check_z = L_z
    else:
        raise ValueError("prep_basis must be 'X' or 'Z'")
        
    def generate_cosets(L_check):
        k = L_check.shape[0] if len(L_check) > 0 else 0
        cosets = []
        for c in itertools.product([0, 1], repeat=k):
            if sum(c) == 0:
                continue
            L_c = np.zeros(n, dtype=int)
            for j in range(k):
                if c[j] == 1:
                    L_c ^= L_check[j]
            cosets.append(L_c)
        return cosets
        
    L_cosets_x = generate_cosets(L_check_x)
    L_cosets_z = generate_cosets(L_check_z)
    
    # 1. Initialize the Strategy (Filter)
    if method == 'milp':
        strategy_x = MILPStrategy(Mx_prep, t)
        strategy_z = MILPStrategy(Mz_prep, t)
    elif method == 'tiered':
        strategy_x = TieredStrategy(Mx_prep, t)
        strategy_z = TieredStrategy(Mz_prep, t)
    elif method == 'lookup':
        strategy_x = LookupStrategy(Mx_prep, t)
        strategy_z = LookupStrategy(Mz_prep, t)
    elif method == 'heuristic':
        strategy_x = HeuristicOnlyStrategy(Mx_prep, t)
        strategy_z = HeuristicOnlyStrategy(Mz_prep, t)
    else:
        raise ValueError("method must be 'milp', 'tiered', 'lookup', or 'heuristic'")
        
    # 2. Initialize the Searcher (Pruning)
    if searcher_type == 'exhaustive':
        searcher = ExhaustiveSearcher(max_split_size=max_weight)
    elif searcher_type == 'early_exit':
        searcher = EarlyExitSearcher(max_splits=max_splits, max_split_size=max_weight)
    else:
        raise ValueError("searcher_type must be 'exhaustive' or 'early_exit'")
    
    safe_splits = {}
    raw_safe_candidates = {"X": {}, "Z": {}}
    gens_x = []
    gens_z = []
    
    # Define evaluators for X and Z hook errors
    def evaluate_x_hook(subset):
        Ex = np.zeros(n, dtype=int)
        Ex[list(subset)] = 1
        if strategy_x.check_tier1(Ex):
            return True
        elif len(L_cosets_x) > 0 and strategy_x.check_tier3(Ex, L_cosets_x):
            return True
        elif len(L_cosets_x) == 0:
            return True
        return False

    def evaluate_z_hook(subset):
        Ez = np.zeros(n, dtype=int)
        Ez[list(subset)] = 1
        if strategy_z.check_tier1(Ez):
            return True
        elif len(L_cosets_z) > 0 and strategy_z.check_tier3(Ez, L_cosets_z):
            return True
        elif len(L_cosets_z) == 0:
            return True
        return False

    # 1. Analyze X generators (produce X-type hook errors)
    for row_idx, gen in enumerate(H_x):
        support = np.where(gen == 1)[0]
        gen_str = f"X({', '.join(map(str, support))})"
        gens_x.append(gen_str)
        
        valid_subsets = searcher.search(support, evaluate_x_hook)
        raw_safe_candidates["X"][gen_str] = valid_subsets
        safe_splits[gen_str] = [f"X({', '.join(map(str, s))})" for s in valid_subsets]

    # 2. Analyze Z generators (produce Z-type hook errors)
    for row_idx, gen in enumerate(H_z):
        support = np.where(gen == 1)[0]
        gen_str = f"Z({', '.join(map(str, support))})"
        gens_z.append(gen_str)
        
        valid_subsets = searcher.search(support, evaluate_z_hook)
        raw_safe_candidates["Z"][gen_str] = valid_subsets
        safe_splits[gen_str] = [f"Z({', '.join(map(str, s))})" for s in valid_subsets]

    global_assignment = None
    if find_assignment:
        assignment_x = find_globally_safe_assignment(gens_x, raw_safe_candidates["X"], n, t, strategy_x, L_cosets_x)
        assignment_z = find_globally_safe_assignment(gens_z, raw_safe_candidates["Z"], n, t, strategy_z, L_cosets_z)
        if assignment_x is not None and assignment_z is not None:
            global_assignment = {}
            for g, chain in assignment_x.items():
                formatted_chain = ", ".join(f"({', '.join(map(str, s))})" for s in chain)
                global_assignment[g] = f"X({formatted_chain})"
            for g, chain in assignment_z.items():
                formatted_chain = ", ".join(f"({', '.join(map(str, s))})" for s in chain)
                global_assignment[g] = f"Z({formatted_chain})"
                
    return safe_splits, global_assignment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze safe hook errors for a given QECC.")
    parser.add_argument("--code", type=str, nargs='*', default=["9_1_3"],
                        help="The name of the QECC to load (e.g., 7_1_3, 9_1_3, 17_1_5)")
    parser.add_argument("--basis", type=str, choices=['X', 'Z'], default='Z',
                        help="The logical state being prepared (X or Z). Defaults to Z.")
    parser.add_argument("--method", type=str, choices=['milp', 'tiered', 'lookup', 'heuristic'], default='lookup',
                        help="Solver method: milp (exact only), tiered (fast heuristics + exact), lookup (BFS dict), heuristic (BP-OSD only).")
    parser.add_argument("--searcher", type=str, choices=['exhaustive', 'early_exit'], default='exhaustive',
                        help="Search pruning method: exhaustive (all combinations), early_exit (stop after finding max-splits).")
    parser.add_argument("--max-splits", type=int, default=1,
                        help="Number of safe splits to find before stopping (if using early_exit).")
    parser.add_argument("--max-weight", type=int, default=None,
                        help="Maximum weight of the split to consider (e.g., 2 or 3).")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print the full mapping of all generators and their safe splits.")
    parser.add_argument("--find-assignment", action="store_true", default=True,
                        help="Find a globally safe assignment where all combinations of k<=t hooks are safe.")
    args = parser.parse_args()
    
    for code_name in args.code:
        print(f"--- Testing {code_name} (Basis: {args.basis}, Method: {args.method}, Searcher: {args.searcher}) ---")
        try:
            is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code_name)
        except Exception as e:
            print(f"Failed to load {code_name}: {e}")
            continue
            
        t0 = time.time()
        safe_splits, global_assignment = analyze_hook_errors(
            H_x, H_z, L_x, L_z, d, 
            prep_basis=args.basis, 
            method=args.method,
            searcher_type=args.searcher,
            max_splits=args.max_splits,
            max_weight=args.max_weight,
            find_assignment=args.find_assignment
        )
        t1 = time.time()
        
        num_x_safe = sum(len(s) for g, s in safe_splits.items() if g.startswith("X"))
        num_z_safe = sum(len(s) for g, s in safe_splits.items() if g.startswith("Z"))
                
        print(f"Code {code_name} (d={d}, t={(d-1)//2}):")
        print(f"  Time taken: {t1 - t0:.4f} seconds")
        print(f"  X generators: {len(H_x)}, safe X-type hook errors: {num_x_safe}")
        print(f"  Z generators: {len(H_z)}, safe Z-type hook errors: {num_z_safe}")
        
        if args.verbose:
            print("\nGenerators and their Safe Splits Mapping:")
            print(json.dumps(safe_splits, indent=2))
            
        if args.find_assignment:
            print("\nGlobally Safe Assignment:")
            if global_assignment:
                print(json.dumps(global_assignment, indent=2))
            else:
                print("Could not find a globally safe assignment from the candidates.")
        print()
