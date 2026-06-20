from pprint import pprint

import numpy as np
import random
from functools import lru_cache
import warnings
import math

from spiderstate.utils import load_qecc
from spiderstate.verification import _generate_candidate_stabilizers
from spiderstate.fast_verification import DynamicCoverageTracker


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

def invert_mod2(matrix: np.ndarray) -> np.ndarray:
    """Inverts a square matrix over GF(2) using Gauss-Jordan elimination. Returns None if singular."""
    n = matrix.shape[0]
    A = np.hstack([matrix, np.eye(n, dtype=int)]) % 2
    for i in range(n):
        pivot = -1
        for j in range(i, n):
            if A[j, i] == 1:
                pivot = j
                break
        if pivot == -1:
            return None
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
        for j in range(n):
            if i != j and A[j, i] == 1:
                A[j] = (A[j] + A[i]) % 2
    return A[:, n:]


def has_unique_ones_property(M: np.ndarray) -> bool:
    """Checks if each row has a '1' that is the unique '1' in its column."""
    r, c = M.shape
    found_rows = set()
    for j in range(c):
        col = M[:, j]
        if np.sum(col) == 1:
            found_rows.add(np.argmax(col))
    return len(found_rows) == r


def row_optimize_matrix(M: np.ndarray, t: int, max_basis_tries: int = 1_000) -> np.ndarray:
    r, c = M.shape

    # --- PHASE 1: Row Operations (Find best basis) ---
    best_row_op_M = None
    best_row_op_cost = float('inf')

    valid_bases = 0
    attempts = 0

    while valid_bases < max_basis_tries and attempts < max_basis_tries * 5:
        attempts += 1
        cols = random.sample(range(c), r)
        submatrix = M[:, cols]

        inv_sub = invert_mod2(submatrix)
        if inv_sub is not None:
            valid_bases += 1
            M_new = (inv_sub @ M) % 2

            if has_unique_ones_property(M_new):
                cost = cnot_cost(M_new, t)
                if cost < best_row_op_cost:
                    best_row_op_cost = cost
                    best_row_op_M = M_new.copy()

    if best_row_op_M is None:
        raise ValueError("Could not find a full-rank submatrix.")

    matrix_after_row_ops = best_row_op_M.copy()
    return best_row_op_cost, matrix_after_row_ops


# --- SIMULATED ANNEALING FOR PHASE 2 ---
def apply_ops(base_M: np.ndarray, ops: list) -> np.ndarray:
    """Applies a sequence of column operations to a matrix."""
    M = base_M.copy()
    for target, source in ops:
        M[:, target] = (M[:, target] + M[:, source]) % 2
    return M


def simulated_annealing_phase2(base_M: np.ndarray, t: int, max_col_ops: int,
                               initial_temp: float = 10.0, cooling_rate: float = 0.99, max_iter: int = 1500):
    """
    Optimizes column operations using Simulated Annealing.
    State is defined as a specific sequence of operations.
    """
    c = base_M.shape[1]

    # Current state: an empty sequence of operations
    current_ops = []
    current_M = base_M.copy()
    current_cost = cnot_cost(current_M, t)  # len(ops) is 0 here

    # Track the global best seen
    best_ops = []
    best_M = current_M.copy()
    best_cost = current_cost

    temp = initial_temp

    for iteration in range(max_iter):
        # 1. Generate a neighbor sequence
        neighbor_ops = current_ops.copy()

        # Decide how to mutate the sequence
        mutation_type = random.choice(['add', 'remove', 'modify'])

        if mutation_type == 'add' and len(neighbor_ops) < max_col_ops:
            target = random.randint(0, c - 1)
            source = random.choice([x for x in range(c) if x != target])
            neighbor_ops.append((target, source))

        elif mutation_type == 'remove' and len(neighbor_ops) > 0:
            neighbor_ops.pop(random.randrange(len(neighbor_ops)))

        elif mutation_type == 'modify' and len(neighbor_ops) > 0:
            idx = random.randrange(len(neighbor_ops))
            target = random.randint(0, c - 1)
            source = random.choice([x for x in range(c) if x != target])
            neighbor_ops[idx] = (target, source)
        else:
            continue  # Skip if mutation wasn't possible (e.g., remove on empty list)

        # 2. Evaluate neighbor
        neighbor_M = apply_ops(base_M, neighbor_ops)

        # Hard constraint: Must maintain the unique 1s property
        if not has_unique_ones_property(neighbor_M):
            continue

            # Energy = CNOT cost + sequence length (since each op costs 1)
        neighbor_cost = cnot_cost(neighbor_M, t) + len(neighbor_ops)

        # 3. Acceptance criteria
        delta_E = neighbor_cost - (cnot_cost(current_M, t) + len(current_ops))

        if delta_E < 0 or random.random() < math.exp(-delta_E / temp):
            current_ops = neighbor_ops
            current_M = neighbor_M

            # Update global best if this is the absolute lowest we've seen
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                best_M = neighbor_M.copy()
                best_ops = neighbor_ops.copy()

        # Cool down
        temp *= cooling_rate

    return best_M, best_ops, best_cost


def optimize_fault_tolerant_matrix(M: np.ndarray, t: int, max_col_ops: int, H_x: np.ndarray = None, H_z: np.ndarray = None, max_basis_tries: int = 5000, return_portfolio: bool = False, beam_width: int = 5, portfolio_size: int = 20, stabs_X: np.ndarray = None, stabs_Z: np.ndarray = None, H_filter_X: np.ndarray = None, H_filter_Z: np.ndarray = None):
    """
    Returns:
    - matrix_after_row_ops
    - final_matrix_after_col_ops
    - col_ops_performed (list of tuples: (target, source))
    If return_portfolio is True, returns (matrix_after_row_ops, portfolio) where portfolio is a list of (M, col_ops)
    """
    r, c = M.shape
    best_row_op_cost, matrix_after_row_ops = row_optimize_matrix(M, t, max_basis_tries)

    # --- PHASE 2: Column Operations ---
    current_M = matrix_after_row_ops.copy()
    current_F_X = np.eye(c, dtype=int)
    current_F_Z = np.eye(c, dtype=int)

    candidate_stabs_X = None
    candidate_stabs_Z = None

    if stabs_X is None and H_x is not None:
        stabs_X = H_x
    if stabs_Z is None and H_z is not None:
        stabs_Z = H_z

    if stabs_X is not None and stabs_Z is not None:
        candidate_stabs_X = _generate_candidate_stabilizers(stabs_X, 3)
        candidate_stabs_Z = _generate_candidate_stabilizers(stabs_Z, 3)
        
    base_tracker = DynamicCoverageTracker(
        t,
        current_F_X, current_F_Z, 
        candidate_stabs_X, candidate_stabs_Z, 
        H_filter_X=H_filter_X,
        H_filter_Z=H_filter_Z,
    )

    current_M = matrix_after_row_ops.copy()
    initial_score = cnot_cost(current_M, t) + base_tracker.evaluate_cost()
    beam = [(initial_score, current_M, base_tracker, [])]
    global_best = beam[0]
    
    all_candidates = [(initial_score, current_M, [])]
    
    for op_num in range(max_col_ops):
        next_beam = []
        for score, curr_M, curr_tracker, curr_ops in beam:
            if ancilla_cost(curr_M, t) == 0:
                # Reached the end: 0 ancillas needed!
                return matrix_after_row_ops, curr_M, curr_ops[::-1]
                
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    test_M = curr_M.copy()
                    test_M[:, i] = (test_M[:, i] + test_M[:, j]) % 2

                    if has_unique_ones_property(test_M):
                        test_tracker = curr_tracker.copy()
                        test_tracker.update_cnot(j, i)
                        heur = cnot_cost(test_M, t) + op_num + test_tracker.evaluate_cost()
                        next_beam.append((heur, test_M, test_tracker, curr_ops + [(j, i)]))
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x[0])
        seen = set()
        beam = []
        for cand in next_beam:
            if cand[0] < global_best[0]:
                global_best = cand
            m_tup = tuple(cand[1].flatten())
            if m_tup not in seen:
                seen.add(m_tup)
                beam.append(cand)
            if len(beam) >= beam_width:
                break

    best_score, best_M, best_tracker, best_ops = global_best
    return matrix_after_row_ops, best_M, best_ops[::-1]


# Example Execution
if __name__ == "__main__":
    is_self_dual, H_z, H_x, L_x, L_z, d = load_qecc("12_2_4")
    t = d // 2

    row_M, final_M, col_ops = optimize_fault_tolerant_matrix(H_x, t=t, max_col_ops=0, max_basis_tries=10_000)
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

    print(f"After Row & Column Operations:")
    print(f"Column Operations Applied (target, source): {col_ops}")
    print(f"CNOT cost (t={t}): {cnot_cost(final_M, t)}")
    print("np.array([")
    for row in final_M:
        print("  [", end="")
        for r in row[:-1]:
            print(f"{r}, ", end="")
        print(f"{row[-1]}],")
    print("])")
