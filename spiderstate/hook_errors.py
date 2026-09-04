from pprint import pprint

import numpy as np
import galois
import argparse
import time
import sys
import os
import json

from spiderstate.optimize_parity_matrix import optimize_fault_tolerant_matrix, row_optimize_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spiderstate.utils import load_qecc

def find_safe_splits(support, M_prep):
    """
    Finds all strictly safe splits for a given generator support using the GF(2) null-space method.
    A split is safe if it is equivalent to a weight <= 1 fault up to M_prep.
    Returns a list of safe splits (each is a tuple of qubit indices).
    """
    n = M_prep.shape[1]
    I = [i for i in range(n) if i not in support]
    safe_splits = set()

    # Empty and full splits are trivial
    safe_splits.add(())
    safe_splits.add(tuple(sorted(support)))

    if len(I) == 0:
        null_basis = np.eye(M_prep.shape[0], dtype=int)
    else:
        M_I = M_prep[:, I]
        null_basis = np.array(galois.GF(2)(M_I).T.null_space(), dtype=int)

    if null_basis.size > 0:
        base_stabs = (null_basis @ M_prep) % 2
        # Include the zero vector (S=0)
        valid_S = [np.zeros(n, dtype=int)]

        # We only generate a small number of combinations to avoid exponential blowup
        import itertools
        max_combinations = min(10, base_stabs.shape[0])
        for r in range(1, max_combinations + 1):
            for combo in itertools.combinations(range(base_stabs.shape[0]), r):
                S_combo = np.zeros(n, dtype=int)
                for idx in combo:
                    S_combo = (S_combo + base_stabs[idx]) % 2
                valid_S.append(S_combo)
                if len(valid_S) > 1000:
                    break
            if len(valid_S) > 1000:
                break

        # E1 = 0
        for row in valid_S:
            supp = tuple(sorted(np.where(row == 1)[0].tolist()))
            safe_splits.add(supp)

        # E1 inside G
        for q in support:
            e_q = np.zeros(n, dtype=int)
            e_q[q] = 1
            for row in valid_S:
                supp = tuple(sorted(np.where((row + e_q) % 2 == 1)[0].tolist()))
                safe_splits.add(supp)
                
    # E1 outside G (q in I)
    if len(I) > 0:
        gf_M_I_T = galois.GF(2)(M_I.T)
        for q in I:
            e_q_I = np.zeros(len(I), dtype=int)
            e_q_I[I.index(q)] = 1

            # Augmented matrix [M_I^T | e_q_I^T]
            Aug = np.column_stack((gf_M_I_T, e_q_I))
            rref = Aug.row_reduce()

            # Check if there is a pivot in the last column
            has_solution = True
            for i in range(rref.shape[0]):
                if np.count_nonzero(rref[i, :-1]) == 0 and rref[i, -1] != 0:
                    has_solution = False
                    break

            if has_solution:
                x_part = np.zeros(gf_M_I_T.shape[1], dtype=int)
                for i in range(rref.shape[0]):
                    row = rref[i]
                    nonzero = np.nonzero(row[:-1])[0]
                    if len(nonzero) > 0:
                        x_part[nonzero[0]] = int(row[-1])

                e_q = np.zeros(n, dtype=int)
                e_q[q] = 1
                base_sol = (x_part @ M_prep + e_q) % 2
                supp = tuple(sorted(np.where(base_sol == 1)[0].tolist()))
                safe_splits.add(supp)

                # Add homogeneous solutions
                if null_basis.size > 0:
                    for row in valid_S:
                        supp_hom = tuple(sorted(np.where((base_sol + row) % 2 == 1)[0].tolist()))
                        safe_splits.add(supp_hom)
                
    # Filter to ensure they are strictly subsets of G (they should be by math, but just to be safe)
    G_set = set(support)
    valid_splits = [s for s in safe_splits if set(s).issubset(G_set)]

    # Cap to avoid O(N^2) explosion in find_longest_chain
    if len(valid_splits) > 1000:
        valid_splits = valid_splits[:1000]

    return valid_splits

def find_longest_chain(support, safe_splits):
    """
    Given a list of safe splits, finds the longest chain of nested splits: S1 c S2 c ... c G.
    Returns the chain as a tuple of tuples.
    """
    splits = sorted([set(s) for s in safe_splits], key=len)
    if not splits:
        return ()

    n = len(splits)
    dp = [1] * n
    prev = [-1] * n

    for i in range(n):
        for j in range(i):
            if splits[j].issubset(splits[i]):
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j

    max_idx = np.argmax(dp)
    chain = []
    curr = max_idx
    while curr != -1:
        chain.append(tuple(sorted(splits[curr])))
        curr = prev[curr]

    chain.reverse()

    # Remove empty and full splits for display purposes
    G_tuple = tuple(sorted(support))
    clean_chain = [c for c in chain if c and c != G_tuple]

    return tuple(clean_chain)

def analyze_hook_errors(H_z, L_z):
    Mz_prep = np.vstack([H_z, L_z]) if len(L_z) > 0 else H_z

    global_assignment = {}
    num_z_safe = 0

    for gen in H_z:
        support = tuple(np.where(gen == 1)[0].tolist())
        splits = find_safe_splits(support, Mz_prep)
        chain = find_longest_chain(support, splits)
        if chain:
            # 1. Compute physical pieces (differences between consecutive chain elements)
            pieces = []
            prev = set()
            for s in chain:
                pieces.append(set(s) - prev)
                prev = set(s)
            pieces.append(set(support) - prev)

            # 2. Merge pieces of size 1 into adjacent pieces
            merged_pieces = []
            current_piece = set()
            for p in pieces:
                current_piece.update(p)
                if len(current_piece) > 1:
                    merged_pieces.append(sorted(list(current_piece)))
                    current_piece = set()

            if len(current_piece) > 0:
                if len(merged_pieces) > 0:
                    merged_pieces[-1] = sorted(list(set(merged_pieces[-1]) | current_piece))
                else:
                    merged_pieces.append(sorted(list(current_piece)))

            # 3. Format output
            global_assignment[support] = merged_pieces

    return global_assignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze safe hook errors for a given QECC using the GF(2) null-space method.")
    parser.add_argument("--code", type=str, nargs='*', default=["7_1_3", "32_20_4"],
                        help="The names of the QECCs to test")
    args = parser.parse_args()
    
    for code in args.code:
        _, H_x, H_z, L_x, L_z, d = load_qecc(code)
        if code in ("49_1_5", "95_1_7"):
            H_x, H_z = H_z, H_x
            L_x, L_z = L_z, L_x
        _, H_z = row_optimize_matrix(H_z, d // 2, 1_000)

        t0 = time.time()
        print(f"  Code: {code}")
        global_assignment = analyze_hook_errors(H_z, L_z)
        t1 = time.time()
        print(f"  Time taken: {t1 - t0:.4f} seconds")
        print("\nGlobally Safe Assignment:")
        pprint(global_assignment)
        print()
