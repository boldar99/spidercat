import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import stim


TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}
Z_MEASUREMENTS = {"MR", "M", "MZ"}
X_MEASUREMENTS = {"MX"}
Z_INITIALIZATIONS = {"MR", "R"}
X_INITIALIZATIONS = {"RX"}
SPECIAL_GATES = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


def layered_ops_to_noisy_stim_circuit(layered_ops: list[list[tuple]], num_qubits: int, p_1: float, p_2: float, p_init: float, p_meas: float, p_mem: float, mem_error_after_every_cnot=False) -> stim.Circuit:
    circuit = stim.Circuit()
    for i, ops in enumerate(layered_ops):
        unused_qubits = set(range(num_qubits))
        has_physical_gates = any(op_tuple[0] not in SPECIAL_GATES for op_tuple in ops)
        
        for op_name, targets, params in ops:
            qubit_targets = [t for t in targets if isinstance(t, int)]
            unused_qubits -= set(qubit_targets)
            
            reconstructed_targets = []
            for t in targets:
                if isinstance(t, tuple) and t[0] == 'rec':
                    reconstructed_targets.append(stim.target_rec(t[1]))
                else:
                    reconstructed_targets.append(t)
            targets = reconstructed_targets

            if op_name in Z_MEASUREMENTS:
                circuit.append("X_ERROR", targets, p_meas)
            elif op_name in X_MEASUREMENTS:
                circuit.append("Z_ERROR", targets, p_meas)

            if params:
                circuit.append(op_name, targets, params)
            else:
                circuit.append(op_name, targets)

            if op_name in X_INITIALIZATIONS:
                circuit.append("Z_ERROR", targets, p_init)
            elif op_name in Z_INITIALIZATIONS:
                circuit.append("X_ERROR", targets, p_init)
            elif op_name in TWO_QUBIT_GATES:
                circuit.append("DEPOLARIZE1", targets, p_2)
                if mem_error_after_every_cnot and p_mem != 0:
                    circuit.append("DEPOLARIZE1", set(range(num_qubits)) - set(qubit_targets), p_mem)

            elif op_name in SPECIAL_GATES:
                pass
            else:
                circuit.append("DEPOLARIZE1", targets, p_1)

        if not mem_error_after_every_cnot and i != len(layered_ops) - 1 and has_physical_gates:
            circuit.append("DEPOLARIZE1", unused_qubits, p_mem)
    return circuit


def _expand_stim_operation_list(operations: list[tuple]):
    stim_operations = []
    for op_name, targets, params in operations:
        if op_name in TWO_QUBIT_GATES:
            for i in range(0, len(targets), 2):
                stim_operations.append((op_name, [targets[i], targets[i + 1]], params))
        elif op_name in SPECIAL_GATES:
            stim_operations.append((op_name, targets, params))
        else:
            for t in targets:
                stim_operations.append((op_name, [t], params))
    return stim_operations


from collections import defaultdict


def _layer_circuit_ops(operations: list[tuple], num_qubits: int):
    # Minor correction: range(num_qubits) avoids creating a ghost qubit tracker
    all_qubits = range(num_qubits)

    # --- PASS 1: ASAP Forward Layering ---
    next_free_layer = {q: 0 for q in all_qubits}
    asap_layers = defaultdict(list)

    for op_name, targets, params in operations:
        qubit_targets = [t for t in targets if isinstance(t, int)]

        if op_name in SPECIAL_GATES:
            last_layer = max(next_free_layer.values(), default=0)
            asap_layers[last_layer].append((op_name, targets, params))
            for i in all_qubits:
                next_free_layer[i] = last_layer + 1
        else:
            last_layer = max((next_free_layer[i] for i in qubit_targets), default=0)
            asap_layers[last_layer].append((op_name, targets, params))
            for i in qubit_targets:
                next_free_layer[i] = last_layer + 1

    # Convert dict to a dense list of lists
    max_layer = max(asap_layers.keys(), default=-1)
    layers = [asap_layers[i] for i in range(max_layer + 1)]

    # --- PASS 2: ALAP Backward Reset Shifting ---
    # Track the exact layer index where a qubit is NEXT used.
    # Initialize to the length of layers (representing the end of the circuit)
    next_required = {q: len(layers) for q in all_qubits}

    # Iterate backwards through the ASAP layers
    for i in range(len(layers) - 1, -1, -1):
        current_layer_ops = layers[i]
        kept_ops = []

        for op_name, targets, params in current_layer_ops:
            if op_name in {"R", "RX"}:
                # Splinter the reset: Handle each qubit independently
                for t in targets:
                    if not isinstance(t, int): continue
                    target_layer = next_required[t] - 1

                    if target_layer > i:
                        # Push this specific qubit's reset forward in time
                        layers[target_layer].append((op_name, [t], params))
                    else:
                        # It's already as late as it can be, keep it here
                        kept_ops.append((op_name, [t], params))
            else:
                # Keep normal gates where they are
                kept_ops.append((op_name, targets, params))
                # Mark these qubits as required at the current layer i
                for t in targets:
                    if isinstance(t, int):
                        next_required[t] = i

        # Update the current layer with only the operations that didn't get pushed
        layers[i] = kept_ops

    # --- PASS 3: Cleanup ---
    # Shifting resets out of early layers might leave some layers completely empty.
    # We strip them out to prevent unnecessary DEPOLARIZE1 idle cycles in your noise model.
    return [layer for layer in layers if layer]


def make_stim_circ_noisy(circ: stim.Circuit, p: float) -> stim.Circuit:
    operations = list(circ.flattened_operations())
    operations = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(operations, circ.num_qubits)
    # final_ops, num_sim_qubits = apply_qubit_reuse(layered_ops)
    noisy_circ = layered_ops_to_noisy_stim_circuit(layered_ops, circ.num_qubits, 0, p, 2 / 3 * p,
                                                   2 / 3 * p, 0, mem_error_after_every_cnot=True)
    return noisy_circ


def apply_qubit_reuse(layers: list[list[tuple]]) -> tuple[list[list[tuple]], int]:
    """
    Takes a temporally optimized list of layers and maps logical qubits
    to a minimal set of physical qubits.
    """
    # 1. Calculate the lifespan (birth layer, death layer) of every logical qubit
    births = {}
    deaths = {}

    for layer_idx, layer in enumerate(layers):
        for op_name, targets, params in layer:
            for t in targets:
                if isinstance(t, int):
                    if t not in births:
                        births[t] = layer_idx
                    deaths[t] = layer_idx

    # 2. Map logical qubits to physical qubits
    logical_to_physical = {}
    physical_freelist = []
    next_new_physical_qubit = 0

    # We track which physical qubits become free at the end of which layer
    # Format: free_at_layer[layer_index] = [physical_q1, physical_q2, ...]
    free_at_layer = {i: [] for i in range(len(layers))}

    for layer_idx in range(len(layers)):
        # Free up physical qubits whose logical occupants died in the PREVIOUS layer
        if layer_idx > 0:
            for p_q in free_at_layer[layer_idx - 1]:
                physical_freelist.append(p_q)

        # Find all logical qubits born in this layer and allocate them
        for logical_q, birth_layer in births.items():
            if birth_layer == layer_idx:
                if physical_freelist:
                    # Reuse an available physical qubit
                    assigned_physical = physical_freelist.pop()
                else:
                    # Allocate a brand new physical qubit
                    assigned_physical = next_new_physical_qubit
                    next_new_physical_qubit += 1

                logical_to_physical[logical_q] = assigned_physical

                # Schedule this physical qubit to be freed after the logical qubit dies
                death_layer = deaths[logical_q]
                free_at_layer[death_layer].append(assigned_physical)

    # 3. Rewrite the layers using the new physical mapping
    optimized_layers = []
    for layer in layers:
        new_layer = []
        for op_name, targets, params in layer:
            mapped_targets = [logical_to_physical[t] if isinstance(t, int) else t for t in targets]
            new_layer.append((op_name, mapped_targets, params))
        optimized_layers.append(new_layer)

    total_physical_qubits_used = next_new_physical_qubit
    return optimized_layers, total_physical_qubits_used

def flatten(ls: list) -> list:
    return list(itertools.chain(*ls))


def find_pivots_in_matrix(parity_matrix):
    r, c = parity_matrix.shape

    # Dictionary to store {row_index: pivot_column_index}
    pivots = {}
    # List to track any rows that do not have a valid pivot
    rows_without_pivots = []

    for i in range(r):
        # 1. Find all columns where the current row has a '1'
        candidate_cols = np.where(parity_matrix[i] == 1)[0]

        found_pivot = False
        for j in candidate_cols:
            # 2. Check if this column is a valid pivot (the sum of the column must be exactly 1)
            if np.sum(parity_matrix[:, j]) == 1:
                pivots[i] = int(j)
                found_pivot = True
                break  # We only need one pivot per row

        if not found_pivot:
            rows_without_pivots.append(i)

    return pivots, rows_without_pivots


def ed(v1: int, v2: int) -> tuple[int, int]:
    return (v1, v2) if v1 < v2 else (v2, v1)


def get_project_root() -> Path:
    return Path(__file__).parent


def load_qecc(code: str):
    root = get_project_root()
    code_file = f"{code}.json"
    for lib in os.listdir(root.joinpath("qeccs")):
        if code_file in os.listdir(root.joinpath("qeccs", lib)):
            method = lib
            break
    else:
        raise FileNotFoundError(code)

    file = root.joinpath("qeccs", method, f"{code}.json")

    with open(file, "r") as f:
        data = json.load(f)

    is_self_dual = data["is_self_dual"]
    H_x, H_z = data.get("H_x"), data.get("H_z")
    L_x, L_z = data.get("L_x"), data.get("L_z")
    if is_self_dual:
        return (
            True,
            np.array(data.get("H_x", H_z), dtype=np.int8), np.array(data.get("H_z", H_x), dtype=np.int8),
            np.array(data.get("L_x", L_z), dtype=np.int8), np.array(data.get("L_z", L_x), dtype=np.int8),
            data["d"]
        )

    assert H_x is not None and H_z is not None
    return False, np.array(H_x, dtype=np.int8), np.array(H_z, dtype=np.int8), np.array(L_x, dtype=np.int8), np.array(L_z, dtype=np.int8), data["d"]


def code_sort_key(code: str):
    n, k, dplus = code.split("_")
    return int(dplus[:-5]), int(n)


def FAO_QECCS():
    root = get_project_root()
    fao = root.joinpath("qeccs", "FAO")
    for file_name in sorted(os.listdir(fao), key=code_sort_key):
        yield file_name[:-5]


def MQT_QECCS():
    root = get_project_root()
    fao = root.joinpath("qeccs", "MQT")
    for file_name in sorted(os.listdir(fao), key=code_sort_key):
        yield file_name[:-5]


def MQT_simp_QECCS():
    yield from [
        "7_1_3",
        # "9_1_3_surface",
        "17_1_5",
        "19_1_5",
        "25_1_5",
        "20_2_6",
        "31_1_7",
        "39_1_7"
    ]


def count_operations(circ: stim.Circuit) -> tuple[int, int]:
    """
    Given a stim circuit, counts the number of operations.
    Returns:
        (num_two_qubit_ops, num_measurements)
    """
    operations = list(circ.flattened_operations())
    
    num_two_qubit_ops = 0
    num_measurements = circ.num_measurements
    
    for op_name, targets, params in operations:
        if op_name in TWO_QUBIT_GATES:
            num_two_qubit_ops += len(targets) // 2
            
    return num_two_qubit_ops, num_measurements


if __name__ == "__main__":
    from spiderstate.cat_at_origin import row_optimized_cat_at_origin

    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc("20_2_6", "FAO")
    circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=5_000)
    p = 0.001
    noisy_circ = make_stim_circ_noisy(circ, p)
    print(noisy_circ)
