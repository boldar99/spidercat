import numpy as np
import stim
from collections import deque

from spiderstate.utils import strings_to_H_T


def find_circuit(stabs: np.ndarray, basis: str) -> stim.Circuit:
    """
    Finds a CNOT circuit to measure a given set of stabilizers.
    Constraint: At most 1 data->ancilla CNOT per data qubit.

    stabs: np.ndarray representing the target parity check matrix H.
    basis: 'X' or 'Z'.

    Returns a stim.Circuit where data qubits are 0 to num_data_qubits-1
    and ancilla qubits are num_data_qubits to num_data_qubits + len(stabs) - 1.
    """
    assert basis in ('X', 'Z'), "basis must be 'X' or 'Z'"
    m = stabs.shape[0]
    n = stabs.shape[1]

    # Check if all stabs have correct length
    assert all(len(s) == n for s in stabs), "All stabilizers must have length num_data_qubits"

    # Convert columns to integers (each integer represents a column of the parity check matrix)
    cols = []
    for i in range(n):
        val = 0
        for j in range(m):
            if stabs[j, i] == 1:
                val |= (1 << j)
        cols.append(val)

    unit_vectors = {1 << j for j in range(m)}

    initial_unsolved = frozenset(c for c in cols if c != 0 and c not in unit_vectors)

    queue = deque([(initial_unsolved, [])])
    visited = {initial_unsolved}
    solution_path = None

    while queue:
        unsolved, path = queue.popleft()

        if not unsolved:
            solution_path = path
            break

        for r in range(m):
            for s in range(m):
                if r == s:
                    continue

                # Operation: add row r to row s
                # In terms of column vector v: v_s <- v_s ^ v_r
                # This corresponds to bit s in v being XORed with bit r
                new_unsolved = set()
                for v in unsolved:
                    bit_r = (v >> r) & 1
                    if bit_r:
                        new_v = v ^ (1 << s)
                    else:
                        new_v = v

                    if new_v not in unit_vectors:
                        new_unsolved.add(new_v)

                new_unsolved = frozenset(new_unsolved)
                if new_unsolved not in visited:
                    visited.add(new_unsolved)
                    queue.append((new_unsolved, path + [(r, s)]))

    if solution_path is None:
        raise ValueError("No circuit found")

    # Reconstruct the circuit
    K = len(solution_path)
    forward_ops = solution_path[::-1]

    col_injection_times = {}  # column value -> (step k, unit_vector_j)

    for val in set(cols):
        if val == 0:
            continue
        current_v = val
        found = False
        if current_v in unit_vectors:
            for j in range(m):
                if current_v == (1 << j):
                    col_injection_times[val] = (0, j)
                    found = True
                    break
        else:
            for k, (r, s) in enumerate(solution_path):
                bit_r = (current_v >> r) & 1
                if bit_r:
                    current_v ^= (1 << s)
                if current_v in unit_vectors:
                    for j in range(m):
                        if current_v == (1 << j):
                            col_injection_times[val] = (k + 1, j)
                            found = True
                            break
                if found:
                    break
        assert found, f"Vector {val} never hit a unit vector!"
    circuit = stim.Circuit()
    
    # Initialize all ancilla qubits
    if basis == 'Z':
        circuit.append("R", [n + i for i in range(m)])
    else:
        circuit.append("RX", [n + i for i in range(m)])

    # We group injections by forward time t_inj = K - k
    injections_by_time = {t: [] for t in range(K + 1)}
    for i, val in enumerate(cols):
        if val != 0:
            k, j = col_injection_times[val]
            t_inj = K - k
            injections_by_time[t_inj].append((i, j))

    # Forward time goes from t=0 to t=K
    for t in range(K + 1):
        # Injections
        for data_idx, ancilla_idx in injections_by_time[t]:
            if basis == 'Z':
                circuit.append("CX", [data_idx, n + ancilla_idx])
            else:
                circuit.append("CX", [n + ancilla_idx, data_idx])

        if t < K:
            r, s = forward_ops[t]
            if basis == 'Z':
                circuit.append("CX", [n + r, n + s])
            else:
                circuit.append("CX", [n + s, n + r])

    # Measure ancilla qubits and add detectors
    for i in range(m):
        if basis == 'Z':
            circuit.append("M", [n + i])
        else:
            circuit.append("MX", [n + i])
        circuit.append("DETECTOR", [stim.target_rec(-1)])

    return circuit


if __name__ == '__main__':
    stabs1 = ['1111000', '0110110']
    print("Test 1 (Z basis):")
    H_T = strings_to_H_T(stabs1)
    print(find_circuit(H_T, 'Z'))

    stabs2 = [
        '0000000000101000011',
        '0011000001001000110',
        '0111000110000010000'
    ]
    print("\nTest 2 (X basis):")
    H_T = strings_to_H_T(stabs2)
    print(find_circuit(H_T, 'X'))

    stabs3 = ['0000000000011100010', '0001010010010100100', '0010010011000000000', '1011000110000001000']
    print("\nTest 3 (Z basis):")
    H_T = strings_to_H_T(stabs3)
    print(find_circuit(H_T, 'Z'))
