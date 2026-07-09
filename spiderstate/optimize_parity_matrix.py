import galois
from pprint import pprint

import numpy as np
import random
from functools import lru_cache
import warnings
import math

from spiderstate.utils import load_qecc
from spiderstate.verification import _generate_candidate_stabilizers
from spiderstate.fast_verification import TrueBackwardTracker
from spidercat.syndrome_measurement import cnot_cost as se_cnot_cost


# --- USER'S ORIGINAL COST FUNCTIONS ---
def density_lower_bound(t):
    with warnings.catch_warnings(action="ignore"):
        return np.where(t == 1, np.inf,
                        (np.ceil((t + 3) / 2) * np.floor((t + 3) / 2)) /
                        (np.ceil((t + 3) / 2) * np.floor((t + 3) / 2) +
                         np.ceil((t - 3) / 2) * np.floor((t + 3) / 2) +
                         np.floor((t - 3) / 2) * np.ceil((t + 3) / 2))
                        )


def minimum_E_and_V(n, t):
    density = density_lower_bound(t)
    E_nec = np.ceil(n / density).astype(int)
    remainder = E_nec % 3
    adjustment = (3 - remainder) % 3
    E_final = E_nec + adjustment
    V_final = (2 * E_final) // 3
    return E_final, V_final


@lru_cache
def minimum_number_of_flags(n, t):
    t_alt = np.floor(n / 2) - 1
    t = np.where(t < t_alt, t, t_alt)
    E, N = minimum_E_and_V(n, t)
    return (np.ceil(E - N + 2).astype(int) - 1).tolist()


def minimum_number_of_cnots(n, t):
    return n - 1 + 2 * minimum_number_of_flags(n, t)


def cnot_cost(M: np.ndarray, t: int) -> int:
    row_sums = np.sum(M, axis=1)
    column_sums = np.sum(M, axis=0)
    cost = 0
    for n in column_sums:
        if n > 1:
            cost += 2 * minimum_number_of_flags(n + 1, t)
    for n in row_sums:
        cost += n - 1 + 2 * minimum_number_of_flags(n, t)
    return cost


def ancilla_cost(M: np.ndarray, t: int) -> int:
    row_sums = np.sum(M, axis=1)
    column_sums = np.sum(M, axis=0)
    cost = 0
    for n in column_sums:
        if n > 1:
            cost += minimum_number_of_flags(n + 1, t)
    for n in row_sums:
        cost += minimum_number_of_flags(n, t)
    return cost


# --- OPTIMIZATION FUNCTIONS ---
def has_unique_ones_property(M: np.ndarray) -> bool:
    """Checks if each row has a '1' that is the unique '1' in its column."""
    col_sums = np.sum(M, axis=0)
    cols_with_one = np.where(col_sums == 1)[0]
    if len(cols_with_one) < M.shape[0]: return False
    found_rows = np.unique(np.argmax(M[:, cols_with_one], axis=0))
    return len(found_rows) == M.shape[0]


def row_optimize_matrix(M: np.ndarray, t: int, max_basis_tries: int = 1_000) -> tuple[float, np.ndarray]:
    r, c = M.shape
    
    GF2 = galois.GF(2)
    A = GF2(M).row_reduce()
    k = np.sum(np.any(A, axis=1))

    # --- PHASE 1: Row Operations (Find best basis) ---
    best_row_op_M = None
    best_row_op_cost = float('inf')
    if has_unique_ones_property(M):
        best_row_op_M = M
        best_row_op_cost = cnot_cost(M, t)

    cols_arr = np.arange(c)

    for _ in range(max_basis_tries):
        np.random.shuffle(cols_arr)
        
        A_perm = GF2(M[:, cols_arr]).row_reduce()
        A_k_perm = np.array(A_perm[:k])
        
        inv_cols = np.empty_like(cols_arr)
        inv_cols[cols_arr] = np.arange(c)
        M_new = A_k_perm[:, inv_cols]

        cost = cnot_cost(M_new, t)
        if cost < best_row_op_cost:
            best_row_op_cost = cost
            best_row_op_M = M_new.copy()

    if best_row_op_M is None:
        raise ValueError("Could not find any valid matrix representation.")

    matrix_after_row_ops = best_row_op_M.copy()
    return best_row_op_cost, matrix_after_row_ops

def optimize_fault_tolerant_matrix(
    M: np.ndarray,
    t: int,
    max_col_ops: int = 10,
    H_x: np.ndarray = None,
    H_z: np.ndarray = None,
    max_basis_tries: int = 5000,
    beam_width: int = 5,
    return_portfolio: bool = False,
    stabs_X: np.ndarray = None,
    stabs_Z: np.ndarray = None,
    H_reduce_X: np.ndarray = None,
    H_reduce_Z: np.ndarray = None,
    heuristic: str = "overlap",
    patience: int = 10
):
    """
    Returns:
    - matrix_after_row_ops
    - final_matrix_after_col_ops
    - col_ops_performed (list of tuples: (target, source))
    If return_portfolio is True, returns (matrix_after_row_ops, portfolio) where portfolio is a list of (M, col_ops)
    """
    best_row_op_cost, matrix_after_row_ops = row_optimize_matrix(M, t, max_basis_tries)
    c = M.shape[1]

    if stabs_X is None and H_x is not None:
        stabs_X = H_x
    if stabs_Z is None and H_z is not None:
        stabs_Z = H_z
        
    candidate_stabs_X = None
    candidate_stabs_Z = None
    if stabs_X is not None:
        candidate_stabs_X = _generate_candidate_stabilizers(stabs_X, 0)
    if stabs_Z is not None:
        candidate_stabs_Z = _generate_candidate_stabilizers(stabs_Z, 0)
        
    costs_X = np.array([se_cnot_cost(w, t) for w in np.sum(candidate_stabs_X, axis=1)]) if candidate_stabs_X is not None else None
    costs_Z = np.array([se_cnot_cost(w, t) for w in np.sum(candidate_stabs_Z, axis=1)]) if candidate_stabs_Z is not None else None

    base_tracker = TrueBackwardTracker(
        t, candidate_stabs_X, candidate_stabs_Z, H_reduce_X, H_reduce_Z, costs_X, costs_Z, heuristic=heuristic
    )

    current_M = matrix_after_row_ops.copy()
    initial_score = cnot_cost(current_M, t) + base_tracker.evaluate_cost()
    beam = [(initial_score, current_M, base_tracker, [])]
    global_best = beam[0]
    
    global_seen = {tuple(current_M.flatten())}
    
    ops_since_improvement = 0

    for op_num in range(1, max_col_ops):
        next_beam_fast = []
        for score, curr_M, curr_tracker, curr_ops in beam:
            if ancilla_cost(curr_M, t) == 0:
                # Reached the end: 0 ancillas needed!
                return matrix_after_row_ops, curr_M, curr_ops[::-1]
                
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    
                    # Prevent trivial backtracking (undoing the exact same CNOT)
                    if curr_ops and curr_ops[-1] == (j, i):
                        continue

                    test_M = curr_M.copy()
                    test_M[:, i] = (test_M[:, i] + test_M[:, j]) % 2

                    if has_unique_ones_property(test_M):
                        m_tup = tuple(test_M.flatten())
                        if m_tup in global_seen:
                            continue

                        # Fast heuristic: just the CNOT cost
                        fast_heur = cnot_cost(test_M, t) + op_num
                        next_beam_fast.append((fast_heur, test_M, curr_tracker, curr_ops, j, i, m_tup))
        
        if not next_beam_fast:
            break
            
        # Pre-sort by fast heuristic
        next_beam_fast.sort(key=lambda x: x[0])
        
        # Only fully evaluate the top candidates (margin of safety: beam_width * 40)
        fully_evaluated = []
        for cand in next_beam_fast[:beam_width * 40]:
            fast_heur, test_M, curr_tracker, curr_ops, j, i, m_tup = cand
            
            test_tracker = curr_tracker.copy()
            test_tracker.update_cnot(j, i)
            heur = fast_heur + test_tracker.evaluate_cost()
            fully_evaluated.append((heur, test_M, test_tracker, curr_ops + [(j, i)], m_tup))
            
        fully_evaluated.sort(key=lambda x: x[0])
        
        beam = []
        layer_seen = set()
        
        improved_in_this_layer = False
        for cand in fully_evaluated:
            if cand[0] < global_best[0]:
                global_best = cand[:4]
                improved_in_this_layer = True
                
            m_tup = cand[4]
            if m_tup not in layer_seen:
                layer_seen.add(m_tup)
                global_seen.add(m_tup)
                beam.append(cand[:4])
            if len(beam) >= beam_width:
                break
                
        if improved_in_this_layer:
            ops_since_improvement = 0
        else:
            ops_since_improvement += 1
            if ops_since_improvement >= patience:
                # Early stopping: No improvement in the last `patience` operations
                break

    best_score, best_M, best_tracker, best_ops = global_best
    return matrix_after_row_ops, best_M, best_ops[::-1]


# Example Execution
if __name__ == "__main__":
    is_self_dual, H_z, H_x, L_x, L_z, d = load_qecc("49_1_7")
    t = d // 2

    # We want H_reduce_X to reduce X faults, so it must be X-type stabilizers.
    # We want H_reduce_Z to reduce Z faults, so it must be Z-type stabilizers.
    H_reduce_X = H_x
    H_reduce_Z = H_z

    _, row_M = row_optimize_matrix(H_x, t=t, max_basis_tries=10_000)
    # row_M = pivot_optimize_parity_matrix(H_x, t=t, max_basis_tries=100_000)


    print(f"Original matrix:")
    print(f"Original CNOT cost (t={t}): {cnot_cost(H_x, t)}")
    print("np.array([")
    for row in H_x:
        print("  [", end="")
        for r in row[:-1]:
            print(f"{r}, ", end="")
        print(f"{row[-1]}],")
    print("])")
    print()

    print(f"After Row Operations:")
    print(f"CNOT cost (t={t}): {cnot_cost(row_M, t)}")
    print("np.array([")
    for row in row_M:
        print("  [", end="")
        for r in row[:-1]:
            print(f"{r}, ", end="")
        print(f"{row[-1]}],")
    print("])")

    # print(f"After Row & Column Operations:")
    # print(f"Column Operations Applied (target, source): {col_ops}")
    # print(f"CNOT cost (t={t}): {cnot_cost(final_M, t)}")
    # print("np.array([")
    # for row in final_M:
    #     print("  [", end="")
    #     for r in row[:-1]:
    #         print(f"{r}, ", end="")
    #     print(f"{row[-1]}],")
    # print("])")
