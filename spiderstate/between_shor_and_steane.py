from pprint import pprint

import numpy as np
import stim

from spiderstate.utils import strings_to_H_T


def decompose_H_transversal_data(H_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a syndrome parity-check matrix H^T into a gate matrix \\Gamma^T
    and an ancilla check matrix \\tilde{H}^T, such that H^T = \\tilde{H}^T \\Gamma^T,
    and \\Gamma^T has at most 1 element per column (i.e. each data qubit interacts 
    with at most one ancilla qubit).
    
    Args:
        H_T: Parity check matrix of size (num_stabs, num_data_qubits).
             Element (i, j) is 1 if stabilizer i acts on data qubit j.
             
    Returns:
        Gamma_T: Gate matrix of size (num_ancilla_qubits, num_data_qubits).
                 Element (i, j) is 1 if a CNOT is applied from data qubit j 
                 to ancilla qubit i.
        H_tilde_T: Ancilla check matrix of size (num_stabs, num_ancilla_qubits).
                   Element (i, j) is 1 if the ancilla state has a Z-stabilizer
                   that includes ancilla qubit j for syndrome bit i.
    """
    num_stabs, num_data_qubits = H_T.shape
    
    # We want to find unique columns of H_T (ignoring the all-zero column).
    # Since numpy unique works over rows, we transpose H_T.
    cols = H_T.T
    
    unique_cols = []
    col_to_unique_idx = {}
    
    for i, col in enumerate(cols):
        if not np.any(col):
            # All-zero column, no CNOTs
            col_to_unique_idx[i] = -1
            continue
            
        col_tuple = tuple(col)
        if col_tuple not in unique_cols:
            unique_cols.append(col_tuple)
            
        col_to_unique_idx[i] = unique_cols.index(col_tuple)
        
    num_ancilla_qubits = len(unique_cols)
    
    Gamma_T = np.zeros((num_ancilla_qubits, num_data_qubits), dtype=np.int8)
    H_tilde_T = np.zeros((num_stabs, num_ancilla_qubits), dtype=np.int8)
    
    for j in range(num_ancilla_qubits):
        H_tilde_T[:, j] = unique_cols[j]
        
    for i in range(num_data_qubits):
        idx = col_to_unique_idx[i]
        if idx != -1:
            Gamma_T[idx, i] = 1
            
    return Gamma_T, H_tilde_T

def decompose_H_scheme_A(H_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a syndrome parity-check matrix H^T using Scheme A.
    Here, \\Gamma^T is a subset of the identity matrix, and \\tilde{H}^T is
    H^T with all-zero columns removed. This corresponds to each data qubit
    in the support of the stabilizers having exactly one dedicated ancilla qubit.
    """
    num_stabs, num_data_qubits = H_T.shape
    cols = H_T.T
    
    active_indices = []
    for i, col in enumerate(cols):
        if np.any(col):
            active_indices.append(i)
            
    num_ancilla_qubits = len(active_indices)
    
    Gamma_T = np.zeros((num_ancilla_qubits, num_data_qubits), dtype=np.int8)
    H_tilde_T = np.zeros((num_stabs, num_ancilla_qubits), dtype=np.int8)
    
    for j, data_idx in enumerate(active_indices):
        Gamma_T[j, data_idx] = 1
        H_tilde_T[:, j] = cols[data_idx]
        
    return Gamma_T, H_tilde_T


def offset_circuit_by(circ: stim.Circuit, offset: int) -> stim.Circuit:
    new_circ = stim.Circuit()
    for op in circ:
        new_targets = []
        for t in op.targets_copy():
            if t.is_qubit_target:
                new_targets.append(stim.GateTarget(t.value + offset))
            elif t.is_x_target:
                new_targets.append(stim.target_x(t.value + offset))
            elif t.is_y_target:
                new_targets.append(stim.target_y(t.value + offset))
            elif t.is_z_target:
                new_targets.append(stim.target_z(t.value + offset))
            else:
                new_targets.append(t)
        new_circ.append(stim.CircuitInstruction(op.name, new_targets, op.gate_args_copy()))
    return new_circ


def measure_stabilizers_scheme_B(H_T, d, basis):
    return measure_stabilizers(H_T, d, basis, decompose_H_transversal_data)


def measure_stabilizers_scheme_A(H_T, d, basis):
    return measure_stabilizers(H_T, d, basis, decompose_H_scheme_A)


def measure_stabilizers(H_T, d, basis, decomposition_method) -> stim.Circuit:
    from spiderstate.cat_at_origin import row_optimized_cat_at_origin
    Gamma_T, H_tilde_T = decomposition_method(H_T)

    state = row_optimized_cat_at_origin(H_tilde_T, d)

    num_data_qubits = H_T.shape[1]
    num_ancilla_qubits = H_tilde_T.shape[1]

    circ = offset_circuit_by(state, num_data_qubits)

    ancilla_qubits = list(range(num_data_qubits, num_data_qubits + num_ancilla_qubits))

    if basis == "Z":
        circ.append("H", ancilla_qubits)

    cnots = zip(*np.where(Gamma_T))
    for ancilla_idx, data_idx in cnots:
        ancilla_q = ancilla_idx + num_data_qubits
        data_q = data_idx
        if basis == "Z":
            circ.append("CNOT", [data_q, ancilla_q])
        else:
            circ.append("CNOT", [ancilla_q, data_q])

    if basis == "X":
        circ.append("MX", ancilla_qubits)
    else:
        circ.append("M", ancilla_qubits)

    for stab in H_tilde_T:
        ixs = np.where(stab)[0]
        targets = [stim.target_rec(int(-num_ancilla_qubits + ix)) for ix in ixs]
        circ.append("DETECTOR", targets)

    return circ


def unifying_syndrome_measurement_test():
    stabs = [
        '0000000000101000011',
        '0011000001001000110',
        '0111000110000010000'
    ]
    stabs = ['0000000000011100010', '0001010010010100100', '0010010011000000000', '1011000110000001000']
    # stabs = ['1111000', '0110110']
    H_T = strings_to_H_T(stabs)
    circ = measure_stabilizers_scheme_B(H_T, 5, basis="Z")
    print("Measurement circuit generated with width:", circ.num_qubits)
    print(circ)

if __name__ == "__main__":
    unifying_syndrome_measurement_test()