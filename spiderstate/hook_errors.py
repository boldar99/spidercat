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
from spiderstate.utils import load_qecc, get_conj_M, FAO_simp_QECCS, misc_QECCS, MQT_QECCS, FAO_QECCS


def analyze_hook_errors_(H_z, L_z):
    return characterize_stabilizer_splits(np.vstack([H_z, L_z]))

def generate_safe_splits_poly(support, M_prep):
    """
    Generates the exact set of safe splits in polynomial scaling time, avoiding full-space combinatorics.
    Uses affine coset geometry (V_0 + U) on the sub-support G.
    """
    n = M_prep.shape[1]
    I = [i for i in range(n) if i not in support]
    G = list(support)
    
    if len(I) == 0:
        import itertools
        return set(tuple(s) for r in range(len(G)+1) for s in itertools.combinations(G, r))
        
    M_I = M_prep[:, I]
    M_G = M_prep[:, G]
    
    null_basis_I = np.array(galois.GF(2)(M_I).T.null_space(), dtype=int)
    if null_basis_I.size > 0:
        V_basis = (null_basis_I @ M_G) % 2
        V_basis = np.array(galois.GF(2)(V_basis).row_reduce(), dtype=int)
        V_basis = V_basis[~np.all(V_basis == 0, axis=1)]
    else:
        V_basis = np.zeros((0, len(G)), dtype=int)
        
    R = [np.zeros(len(G), dtype=int)]
    for i in range(len(G)):
        e = np.zeros(len(G), dtype=int)
        e[i] = 1
        R.append(e)
        
    gf_M_I_T = galois.GF(2)(M_I.T)
    for q in range(len(I)):
        e_q_I = np.zeros(len(I), dtype=int)
        e_q_I[q] = 1
        Aug = np.column_stack((gf_M_I_T, e_q_I))
        rref = Aug.row_reduce()
        
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
            base_sol_G = (x_part @ M_G) % 2
            R.append(base_sol_G)
            
    V_0 = [np.zeros(len(G), dtype=int)]
    for row in V_basis:
        new_V_0 = []
        for v in V_0:
            new_V_0.append(v)
            new_V_0.append((v + row) % 2)
        V_0 = new_V_0
        
    safe_splits = set()
    for r in R:
        for v in V_0:
            combined = (r + v) % 2
            split = tuple(sorted(G[i] for i in range(len(G)) if combined[i] == 1))
            safe_splits.add(split)
            
    return safe_splits


def characterize_stabilizer_splits(Mz_prep):
    """
    Evaluates all stabilizers and classifies their optimal partition shape in polynomial time.
    Returns a dict mapping support to a dictionary containing universal/partial lists.
    Filters out stabilizers that can only be split trivially (i.e. partitions containing 1).
    """
    results = {}
    import math
    for gen in Mz_prep:
        support = tuple(np.where(gen == 1)[0].tolist())
        N = len(support)
        n_qubits = Mz_prep.shape[1]
        I = [i for i in range(n_qubits) if i not in support]
        
        if len(I) == 0:
            def get_all_parts(n, current_prefix, current_partition):
                if current_prefix == n: return [current_partition]
                res = []
                for size in range(2, n - current_prefix + 1):
                    res.extend(get_all_parts(n, current_prefix + size, current_partition + [size]))
                return res
            univ = get_all_parts(N, 0, [])
            univ = [p for p in univ if len(p) > 1]
            univ.sort(key=lambda p: (min(p), -len(p)), reverse=True)
            if univ:
                results[support] = {"universal": univ, "partial": []}
            continue
            
        safe_splits = generate_safe_splits_poly(support, Mz_prep)
        
        splits_by_size = {}
        for s in safe_splits:
            L = len(s)
            if 2 <= L <= N - 2:
                if L not in splits_by_size:
                    splits_by_size[L] = []
                splits_by_size[L].append(set(s))
                
        universal_sizes = set()
        for L, splits in splits_by_size.items():
            if len(splits) == math.comb(N, L):
                universal_sizes.add(L)
                
        def get_partitions_from_sizes(n, current_prefix, current_partition, valid_sizes):
            if current_prefix == n: return [current_partition]
            res = []
            for size in range(2, n - current_prefix + 1):
                next_prefix = current_prefix + size
                if next_prefix in valid_sizes or next_prefix == n:
                    res.extend(get_partitions_from_sizes(n, next_prefix, current_partition + [size], valid_sizes))
            return res
            
        univ_parts = get_partitions_from_sizes(N, 0, [], universal_sizes)
        non_trivial_univ = [p for p in univ_parts if len(p) > 1]
        non_trivial_univ.sort(key=lambda p: (min(p), -len(p)), reverse=True)
        
        memo = {}
        def dfs(current_split, current_size):
            if current_size == N: return [[]]
            t_split = tuple(sorted(current_split))
            if t_split in memo: return memo[t_split]
                
            valid_suffixes = []
            
            if N - current_size >= 2:
                valid_suffixes.append([N - current_size])
                
            for L in sorted(splits_by_size.keys()):
                if L - current_size >= 2:
                    for next_split in splits_by_size[L]:
                        if current_split.issubset(next_split):
                            suffixes = dfs(next_split, L)
                            for suf in suffixes:
                                valid_suffixes.append([L - current_size] + suf)
            
            unique_suf = []
            seen = set()
            for suf in valid_suffixes:
                t = tuple(suf)
                if t not in seen:
                    seen.add(t)
                    unique_suf.append(suf)
            memo[t_split] = unique_suf
            return unique_suf
            
        exist_parts = dfs(set(), 0)
        non_trivial_exist = [p for p in exist_parts if len(p) > 1]
        
        univ_tuples = set(tuple(p) for p in non_trivial_univ)
        non_trivial_exist = [p for p in non_trivial_exist if tuple(p) not in univ_tuples]
        non_trivial_exist.sort(key=lambda p: (min(p), -len(p)), reverse=True)
        
        if non_trivial_univ or non_trivial_exist:
            results[support] = {
                "universal": non_trivial_univ,
                "partial": non_trivial_exist,
                "splits_by_size": splits_by_size
            }
            
    return results


def explain_safe_splits(results, max_print_splits=25):
    """
    Pretty prints the safe splits analysis, showing partition shapes and their safety status.
    """
    import math
    for support, data in results.items():
        N = len(support)
        print(f"Stabilizer Support: {support} (Weight {N})")
        
        all_parts = []
        for p in data["universal"]:
            all_parts.append((p, "Universally Safe"))
        for p in data["partial"]:
            all_parts.append((p, "Partially Safe"))
            
        if not all_parts:
            print("  No non-trivial safe partitions found.")
            print("-" * 50)
            continue
            
        print("  Prefix Size Safety Ratios:")
        shapes_to_print = []
        
        for p, status in all_parts:
            if len(p) > 2:
                prefix_ratios = []
                current = 0
                min_sc = float('inf')
                for chunk in p[:-1]:
                    current += chunk
                    if "splits_by_size" in data:
                        sc = len(data["splits_by_size"].get(current, []))
                    else:
                        sc = math.comb(N, current)
                    tc = math.comb(N, current)
                    prefix_ratios.append(f"{sc}/{tc}")
                    min_sc = min(min_sc, sc)
                if status == "Partially Safe" and min_sc <= max_print_splits:
                    shapes_to_print.append(p)
                ratio_str = " | ".join(prefix_ratios)
                print(f"    {p}: {ratio_str} ({status})")
            else:
                L = p[0]
                if "splits_by_size" in data:
                    sc = len(data["splits_by_size"].get(L, []))
                else:
                    sc = math.comb(N, L)
                tc = math.comb(N, L)
                if status == "Partially Safe" and sc <= max_print_splits:
                    shapes_to_print.append(p)
                print(f"    {p}: {sc:5d} / {tc:5d} ({status})")
                
        if shapes_to_print:
            print("  Exact Partially Safe Splits:")
            support_set = set(support)
            splits_by_size = data.get("splits_by_size", {})
            
            for p in shapes_to_print:
                prefix_sizes = []
                current = 0
                for chunk in p[:-1]:
                    current += chunk
                    prefix_sizes.append(current)
                prefix_sizes.append(N)
                
                def backtrack(prefix_idx, current_split):
                    if prefix_idx == len(prefix_sizes):
                        return [[]]
                    L = prefix_sizes[prefix_idx]
                    if L == N:
                        return [[tuple(sorted(support_set - current_split))]]
                        
                    valid_chains = []
                    for next_split in splits_by_size.get(L, []):
                        if current_split.issubset(next_split):
                            chunk = tuple(sorted(next_split - current_split))
                            suffixes = backtrack(prefix_idx + 1, next_split)
                            for suf in suffixes:
                                valid_chains.append([chunk] + suf)
                    return valid_chains
                    
                chains = backtrack(0, set())
                chains.sort()
                for chain in chains:
                    print(f"    {p}: {tuple(chain)}")
                    
        print("-" * 50)


def get_exact_partial_splits(support, shape, splits_by_size, required_last_element=None):
    """
    Reconstructs an exact valid chain of subsets for a partially safe partition shape.
    """
    support_set = set(support)
    N = len(support)
    prefix_sizes = []
    current = 0
    for chunk in shape[:-1]:
        current += chunk
        prefix_sizes.append(current)
    prefix_sizes.append(N)
    
    def backtrack(prefix_idx, current_split):
        if prefix_idx == len(prefix_sizes):
            return [[]]
        L = prefix_sizes[prefix_idx]
        if L == N:
            return [[tuple(sorted(support_set - current_split))]]
            
        valid_chains = []
        for next_split in splits_by_size.get(L, []):
            if current_split.issubset(next_split):
                chunk = tuple(sorted(next_split - current_split))
                suffixes = backtrack(prefix_idx + 1, next_split)
                for suf in suffixes:
                    valid_chains.append([chunk] + suf)
        return valid_chains
        
    chains = backtrack(0, set())
    if required_last_element is not None:
        valid_chains = [chain for chain in chains if required_last_element in chain[-1]]
        if not valid_chains:
            return None
        return valid_chains[0]
    return chains[0] if chains else None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze safe hook errors for a given QECC using the GF(2) null-space method.")
    parser.add_argument("--code", type=str, nargs='*', default=FAO_QECCS(),
                        help="The names of the QECCs to test")
    args = parser.parse_args()
    
    for code in args.code:
        _, H_x, H_z, L_x, L_z, d = load_qecc(code)
        if code in ("49_1_5", "95_1_7"):
            H_x, H_z = H_z, H_x
            L_x, L_z = L_z, L_x
        t = d // 2

        cost, row_M = row_optimize_matrix(H_x, t=t, max_basis_tries=1_000)
        # print(list(find_safe_logical_hook_errors(get_conj_M(row_M)).values()))
        print(f"\nEvaluating code: {code}")
        explain_safe_splits(characterize_stabilizer_splits(get_conj_M(row_M)))
