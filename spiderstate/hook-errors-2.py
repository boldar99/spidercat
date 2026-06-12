import numpy as np
from itertools import combinations
from typing import Dict, Tuple, List, Set
from dataclasses import dataclass
import time
from z3 import Optimize, Bool, If, Sum, Or, Not, sat


@dataclass
class HookErrorItem:
    E: np.ndarray
    cost: int
    indices: Set[int]
    stabilizer_index: int
    s_H: int
    l_actual: int
    global_index: int = -1


def prep_integer_matrices(H_check: np.ndarray, L_check: np.ndarray) -> Tuple[List[int], int, int]:
    if L_check is None or L_check.size == 0:
        M = H_check.astype(int)
        k_logicals = 0
    else:
        M = np.vstack((H_check, L_check)).astype(int)
        k_logicals = L_check.shape[0]

    k_total = M.shape[0]
    if k_total == 0:
        return [0] * H_check.shape[1], 0, 0

    powers = 1 << np.arange(k_total - 1, -1, -1, dtype=object)
    col_ints = [int((M[:, i] % 2).dot(powers)) for i in range(M.shape[1])]
    return col_ints, k_total, k_logicals


def build_fast_lut(col_ints: List[int], max_weight: int) -> Dict[int, int]:
    print(f"Building Integer BFS LUT up to weight {max_weight}...")
    start = time.time()
    lut = {0: 0}
    current_level = {0}

    for w in range(1, max_weight + 1):
        next_level = set()
        for s in current_level:
            for c in col_ints:
                new_s = s ^ c
                if new_s not in lut:
                    lut[new_s] = w
                    next_level.add(new_s)
        current_level = next_level
    print(f"LUT built in {time.time() - start:.3f} seconds.")
    return lut


def is_error_fine_fast(s_H: int, l_actual: int, total_cost: int, lut: Dict[int, int], k_logicals: int, d: int) -> bool:
    """
    Evaluates safety by ensuring the adversary cannot bridge the gap to ANY OTHER logical branch.
    """
    required_dist = d - total_cost - 1
    if required_dist < 0:
        return False

    # If there are no logicals (e.g., Z-errors on |0>_L), no logical failure is possible.
    if k_logicals == 0:
        return True

    # Check the minimum weight needed to reach ALL OTHER logical branches
    for l_other in range(1 << k_logicals):
        if l_other == l_actual:
            continue
        key_other = (s_H << k_logicals) | l_other

        # If the adversary can supply an error of weight <= required_dist to reach this branch, it's fatal.
        if lut.get(key_other, float('inf')) <= required_dist:
            return False

    return True


def generate_candidates(
    H_measure: np.ndarray,
    col_ints: List[int],
    lut: Dict[int, int],
    k_logicals: int,
    d: int,
    strict_coset_filter: bool
) -> Tuple[List[HookErrorItem], Dict[int, List[int]]]:
    n = H_measure.shape[1]
    candidates = []
    stabs_to_cands = {i: [] for i in range(H_measure.shape[0])}

    global_idx = 0
    for i in range(H_measure.shape[0]):
        support = np.where(H_measure[i] == 1)[0]

        # GEOMETRIC COMPLEMENT RULE
        max_w = len(support) // 2

        for w in range(2, max_w + 1):
            for indices in combinations(support, w):
                key_E = 0
                for idx in indices:
                    key_E ^= col_ints[idx]

                # Bitwise extraction
                if k_logicals > 0:
                    l_actual = key_E & ((1 << k_logicals) - 1)
                    s_H = key_E >> k_logicals
                else:
                    l_actual = 0
                    s_H = key_E

                # STRICT COSET FILTER (Toggleable)
                if strict_coset_filter:
                    if lut.get(key_E, float('inf')) < w:
                        continue

                        # The Corrected Physics Oracle
                if not is_error_fine_fast(s_H, l_actual, 1, lut, k_logicals, d):
                    continue

                E = np.zeros(n, dtype=int)
                E[list(indices)] = 1
                item = HookErrorItem(E, 1, set(indices), i, s_H, l_actual, global_idx)
                candidates.append(item)
                stabs_to_cands[i].append(global_idx)
                global_idx += 1

    return candidates, stabs_to_cands


def find_optimal_hooks_cegar(
    H_measure: np.ndarray,
    H_check: np.ndarray,
    L_check: np.ndarray,
    d: int,
    max_cost: int = None,
    strict_coset_filter: bool = True
) -> Dict[int, List[HookErrorItem]]:
    if max_cost is None:
        max_cost = d // 2

    if max_cost < 1:
        return {}

    col_ints, k_total, k_logicals = prep_integer_matrices(H_check, L_check)
    max_degree = int(np.max(np.sum(H_measure, axis=1)))
    max_lut_weight = max(max_degree, d - 2)
    lut = build_fast_lut(col_ints, max_lut_weight)

    candidates, stabs_to_cands = generate_candidates(
        H_measure, col_ints, lut, k_logicals, d, strict_coset_filter
    )
    N = len(candidates)

    if N == 0:
        return {}

    opt = Optimize()
    X = [Bool(f"x_{i}") for i in range(N)]

    stab_has_hook = []
    for stab_idx, cand_indices in stabs_to_cands.items():
        if cand_indices:
            has_hook = Bool(f"has_hook_{stab_idx}")
            opt.add(has_hook == Or([X[i] for i in cand_indices]))
            stab_has_hook.append(If(has_hook, 1, 0))

    opt.maximize(Sum(stab_has_hook))
    opt.maximize(Sum([If(X[i], 1, 0) for i in range(N)]))

    while True:
        if opt.check() != sat:
            raise RuntimeError("Z3 Optimizer UNSAT.")

        model = opt.model()
        proposed_indices = [i for i in range(N) if model.evaluate(X[i], model_completion=True)]
        conflict_found = False

        if max_cost >= 2 and k_logicals > 0:
            for cost_comb in range(2, max_cost + 1):
                if conflict_found: break
                for sub_indices in combinations(proposed_indices, cost_comb):
                    subset = [candidates[i] for i in sub_indices]

                    stab_indices = [c.stabilizer_index for c in subset]
                    if len(stab_indices) != len(set(stab_indices)):
                        continue

                    s_H_comb = 0
                    l_actual_comb = 0
                    for c in subset:
                        s_H_comb ^= c.s_H
                        l_actual_comb ^= c.l_actual

                    total_cost = sum(c.cost for c in subset)

                    if not is_error_fine_fast(s_H_comb, l_actual_comb, total_cost, lut, k_logicals, d):
                        conflict_clause = Or([Not(X[i]) for i in sub_indices])
                        opt.add(conflict_clause)
                        conflict_found = True
                        break

        if not conflict_found:
            final_assignment = {i: [] for i in range(H_measure.shape[0])}
            for i in proposed_indices:
                item = candidates[i]
                final_assignment[item.stabilizer_index].append(item)
            return final_assignment


def analyze_0_state_prep(H_x, H_z, L_x, L_z, d, strict_coset_filter=True):
    """
    Analyzes BOTH X and Z measurement hooks during the preparation of |0>_L.
    X-errors are evaluated against L_z.
    Z-errors are evaluated with no logicals, as L_z acts as a stabilizer.
    """
    empty_L = np.empty((0, H_x.shape[1]))

    print("\n--- Analyzing X-Hooks for |0>_L State Preparation ---")
    x_hooks = find_optimal_hooks_cegar(H_x, H_z, L_z, d, strict_coset_filter=strict_coset_filter)

    print("\n--- Analyzing Z-Hooks for |0>_L State Preparation ---")
    z_hooks = find_optimal_hooks_cegar(H_z, H_x, empty_L, d, strict_coset_filter=strict_coset_filter)

    return x_hooks, z_hooks


def print_hook_analysis(optimal_hooks: Dict[int, List[HookErrorItem]], H_measure: np.ndarray, title: str):
    num_stabs = H_measure.shape[0]
    simplified_stabs = sum(1 for hooks in optimal_hooks.values() if hooks)

    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")
    print(f"Successfully found allowed hooks for {simplified_stabs} out of {num_stabs} stabilizers:\n")

    for stab_idx in range(num_stabs):
        support = np.where(H_measure[stab_idx] == 1)[0]
        hooks = optimal_hooks.get(stab_idx, [])
        if hooks:
            print(f"  [+] Stab {stab_idx:02d} (Qubits {list(support)}):")
            for err in hooks:
                indices = [idx for idx, val in enumerate(err.E) if val == 1]
                print(f"      -> Allowed Hook: {indices}")
        else:
            print(f"  [-] Stab {stab_idx:02d} (Qubits {list(support)}): NO SAFE HOOK.")


def run():
    try:
        from spiderstate.utils import load_qecc
        name = "19_1_5"  # Steane code
        is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(name, "MQT")
    except ImportError:
        return

    print(f"Testing |0>_L State Preparation Mapping on {name} (n={H_x.shape[1]}, d={d})...")

    # We set strict_coset_filter=False because Steane is a perfect code
    x_hooks, z_hooks = analyze_0_state_prep(H_x, H_z, L_x, L_z, d, strict_coset_filter=False)

    print_hook_analysis(x_hooks, H_x, title="X-Hooks (Measured against L_z)")
    print_hook_analysis(z_hooks, H_z, title="Z-Hooks (Protected by L_z stabilizer)")


if __name__ == "__main__":
    run()