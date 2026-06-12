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

    E_eff_indices: List[int] = None
    used_stabs: List[int] = None
    used_logicals: List[int] = None


def decompose_gf2(basis: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not np.any(target):
        return np.zeros(basis.shape[0], dtype=int)

    A = basis.T.copy()
    b = target.copy().reshape(-1, 1)
    Aug = np.hstack((A, b))
    n_rows, n_cols = Aug.shape

    lead = 0
    for r in range(n_rows):
        if lead >= n_cols - 1:
            break
        i = r
        while Aug[i, lead] == 0:
            i += 1
            if i == n_rows:
                i = r
                lead += 1
                if lead == n_cols - 1:
                    break
        if lead == n_cols - 1:
            break

        Aug[[i, r]] = Aug[[r, i]]
        for i in range(n_rows):
            if i != r and Aug[i, lead] == 1:
                Aug[i] = (Aug[i] + Aug[r]) % 2
        lead += 1

    c = np.zeros(basis.shape[0], dtype=int)
    for i in range(basis.shape[0]):
        ones = np.where(Aug[:, i] == 1)[0]
        if len(ones) > 0:
            c[i] = Aug[ones[0], -1]

    if np.array_equal((c @ basis) % 2, target):
        return c
    return None


def prep_integer_matrices(H_check: np.ndarray, L_check: np.ndarray) -> Tuple[List[int], int, int]:
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
    current_level = [(0, 0)]

    for w in range(1, max_weight + 1):
        next_level = []
        for E_int, s_H in current_level:
            for i, c in enumerate(col_ints):
                new_E_int = E_int ^ (1 << i)
                new_s_H = s_H ^ c
                if new_s_H not in lut:
                    lut[new_s_H] = new_E_int
                    next_level.append((new_E_int, new_s_H))
        current_level = next_level
    print(f"LUT built in {time.time() - start:.3f} seconds.")
    return lut


def is_error_fine_fast(s_H: int, l_actual: int, total_cost: int, lut: Dict[int, int], k_logicals: int, d: int) -> bool:
    required_dist = d - total_cost - 1
    if required_dist < 0:
        return False

    for l_other in range(1 << k_logicals):
        if l_other == l_actual:
            continue
        key_other = (s_H << k_logicals) | l_other

        E_int_other = lut.get(key_other, None)
        if E_int_other is not None:
            wt_other = bin(E_int_other).count('1')
            if wt_other <= required_dist:
                return False

    return True


def generate_candidates(
    H_measure: np.ndarray,
    col_ints: List[int],
    lut: Dict[int, int],
    k_logicals: int,
    d: int,
    H_decompose: np.ndarray,
    L_decompose: np.ndarray
) -> Tuple[List[HookErrorItem], Dict[int, List[int]]]:
    n = H_measure.shape[1]
    candidates = []
    stabs_to_cands = {i: [] for i in range(H_measure.shape[0])}

    basis = np.vstack((H_decompose, L_decompose)) if L_decompose.size else H_decompose
    k_stabs = H_decompose.shape[0]

    global_idx = 0
    for i in range(H_measure.shape[0]):
        support = np.where(H_measure[i] == 1)[0]
        max_w = len(support) // 2

        for w in range(2, max_w + 1):
            for indices in combinations(support, w):
                key_E = 0
                for idx in indices:
                    key_E ^= col_ints[idx]

                l_actual = key_E & ((1 << k_logicals) - 1)
                s_H = key_E >> k_logicals

                E_eff_int = lut.get(key_E, None)
                if E_eff_int is None:
                    continue

                # The Unified Logical Safety Oracle (Applies equally to X and Z)
                if not is_error_fine_fast(s_H, l_actual, 1, lut, k_logicals, d):
                    continue

                E_actual = np.zeros(n, dtype=int)
                E_actual[list(indices)] = 1

                E_eff_list = [bit for bit in range(n) if (E_eff_int & (1 << bit))]
                E_eff_arr = np.zeros(n, dtype=int)
                E_eff_arr[E_eff_list] = 1

                E_diff = (E_actual + E_eff_arr) % 2

                c = decompose_gf2(basis, E_diff)
                used_stabs = [idx for idx in range(k_stabs) if c[idx] == 1] if c is not None else []
                used_logicals = [idx - k_stabs for idx in range(len(c)) if
                                 idx >= k_stabs and c[idx] == 1] if c is not None else []

                item = HookErrorItem(
                    E=E_actual, cost=1, indices=set(indices), stabilizer_index=i,
                    s_H=s_H, l_actual=l_actual, global_index=global_idx,
                    E_eff_indices=E_eff_list, used_stabs=used_stabs, used_logicals=used_logicals
                )

                candidates.append(item)
                stabs_to_cands[i].append(global_idx)
                global_idx += 1

    return candidates, stabs_to_cands


def find_optimal_hooks_cegar(
    H_measure: np.ndarray, H_check: np.ndarray, L_check: np.ndarray, d: int,
    H_decompose: np.ndarray, L_decompose: np.ndarray,
    max_cost: int = None
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
        H_measure, col_ints, lut, k_logicals, d,
        H_decompose, L_decompose
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

                    if not is_error_fine_fast(s_H_comb, l_actual_comb, sum(c.cost for c in subset), lut, k_logicals, d):
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


def analyze_ancilla_prep(H_x, H_z, L_x, L_z, d):
    # Both bases are now evaluated against the lethal Data block logicals
    print("\n--- Analyzing X-Hooks for Ancilla Prep (Evaluated against L_z) ---")
    x_hooks = find_optimal_hooks_cegar(
        H_x, H_z, L_z, d,
        H_decompose=H_x, L_decompose=L_x
    )

    print("\n--- Analyzing Z-Hooks for Ancilla Prep (Evaluated against L_x) ---")
    z_hooks = find_optimal_hooks_cegar(
        H_z, H_x, L_x, d,
        H_decompose=H_z, L_decompose=L_x
    )
    return x_hooks, z_hooks


def print_hook_analysis(optimal_hooks: Dict[int, List[HookErrorItem]], H_measure: np.ndarray, title: str, d: int):
    num_stabs = H_measure.shape[0]
    simplified_stabs = sum(1 for hooks in optimal_hooks.values() if hooks)

    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(f"Successfully found allowed hooks for {simplified_stabs} out of {num_stabs} stabilizers:\n")

    for stab_idx in range(num_stabs):
        support = np.where(H_measure[stab_idx] == 1)[0]
        hooks = optimal_hooks.get(stab_idx, [])
        if hooks:
            print(f"  [+] Stab {stab_idx:02d} (Qubits {list(support)}):")

            # Sorted by efficiency: Cost / Physical Weight ascending
            hooks.sort(key=lambda x: x.cost / len(x.indices))

            for err in hooks:
                indices = [idx for idx, val in enumerate(err.E) if val == 1]
                physical_weight = len(indices)
                ratio = err.cost / physical_weight

                stabs_str = f"Stabs {[f'{i:02d}' for i in err.used_stabs]}" if err.used_stabs else "No Stabs"
                logs_str = f" + Logicals {err.used_logicals}" if err.used_logicals else ""
                proof_str = f"{stabs_str}{logs_str}"

                print(
                    f"      -> Hook: {indices}  ==>  Physical Wt: {physical_weight} | Fault Cost: {err.cost} | Ratio (Cost/Wt): {ratio:.2f}")
                print(
                    f"         Status: SAFE. Requires > {d - err.cost - 1} additional faults on Data block to break code.")
        else:
            print(f"  [-] Stab {stab_idx:02d} (Qubits {list(support)}): NO SAFE HOOK.")


def run():
    try:
        from spiderstate.utils import load_qecc
        name = "7_1_3"
        is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(name, "FAO")
    except ImportError:
        return

    print(f"Testing Transversal Ancilla Preparation Mapping on {name} (n={H_x.shape[1]}, d={d})...")

    x_hooks, z_hooks = analyze_ancilla_prep(H_x, H_z, L_x, L_z, d)

    print_hook_analysis(x_hooks, H_x, title="X-Hooks (Evaluated by Fault Cost vs L_z)", d=d)
    print_hook_analysis(z_hooks, H_z, title="Z-Hooks (Evaluated by Fault Cost vs L_x)", d=d)


if __name__ == "__main__":
    run()