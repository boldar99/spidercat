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


def layered_ops_to_noisy_stim_circuit(layered_ops: list[list[tuple[str, list[int]]]], num_qubits: int, p_1: float, p_2: float, p_init: float, p_meas: float, p_mem: float, mem_error_after_every_cnot=False) -> stim.Circuit:
    circuit = stim.Circuit()
    for i, ops in enumerate(layered_ops):
        unused_qubits = set(range(num_qubits))
        for op_name, targets in ops:
            unused_qubits -= set(targets)

            if op_name in Z_MEASUREMENTS:
                circuit.append("X_ERROR", targets, p_meas)
            elif op_name in X_MEASUREMENTS:
                circuit.append("Z_ERROR", targets, p_meas)

            circuit.append(op_name, targets)

            if op_name in X_INITIALIZATIONS:
                circuit.append("Z_ERROR", targets, p_init)
            elif op_name in Z_INITIALIZATIONS:
                circuit.append("X_ERROR", targets, p_init)
            elif op_name in TWO_QUBIT_GATES:
                circuit.append("DEPOLARIZE2", targets, p_2)
                if mem_error_after_every_cnot:
                    circuit.append("DEPOLARIZE1", set(range(num_qubits)) - set(targets), p_mem)

            elif op_name in SPECIAL_GATES:
                pass
            else:
                circuit.append("DEPOLARIZE1", targets, p_1)

        if not mem_error_after_every_cnot and i != len(layered_ops) - 1:
            circuit.append("DEPOLARIZE1", unused_qubits, p_mem)
    return circuit


def _expand_stim_operation_list(operations: list[tuple[str, list[int]]]):
    stim_operations = []
    for op_name, targets in operations:
        if op_name in TWO_QUBIT_GATES:
            for i in range(0, len(targets), 2):
                stim_operations.append((op_name, [targets[i], targets[i + 1]]))
        elif op_name in SPECIAL_GATES:
            stim_operations.append((op_name, targets))
        else:
            for t in targets:
                stim_operations.append((op_name, [t]))
    return stim_operations


from collections import defaultdict


def _layer_circuit_ops(operations: list[tuple[str, list[int]]], num_qubits: int):
    # Minor correction: range(num_qubits) avoids creating a ghost qubit tracker
    all_qubits = range(num_qubits)

    # --- PASS 1: ASAP Forward Layering ---
    next_free_layer = {q: 0 for q in all_qubits}
    asap_layers = defaultdict(list)

    for op_name, targets in operations:
        # Note: If you pass "MR" in here, it will be kept as one block.
        # For optimal noise, you should preprocess your operations list
        # to convert ("MR", targets) into ("M", targets) and ("R", targets)
        # before calling this function!

        last_layer = max((next_free_layer[i] for i in targets), default=0)
        asap_layers[last_layer].append((op_name, targets))

        for i in targets:
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

        for op_name, targets in current_layer_ops:
            if op_name in {"R", "RX"}:
                # Splinter the reset: Handle each qubit independently
                for t in targets:
                    target_layer = next_required[t] - 1

                    if target_layer > i:
                        # Push this specific qubit's reset forward in time
                        layers[target_layer].append((op_name, [t]))
                    else:
                        # It's already as late as it can be, keep it here
                        kept_ops.append((op_name, [t]))
            else:
                # Keep normal gates where they are
                kept_ops.append((op_name, targets))
                # Mark these qubits as required at the current layer i
                for t in targets:
                    next_required[t] = i

        # Update the current layer with only the operations that didn't get pushed
        layers[i] = kept_ops

    # --- PASS 3: Cleanup ---
    # Shifting resets out of early layers might leave some layers completely empty.
    # We strip them out to prevent unnecessary DEPOLARIZE1 idle cycles in your noise model.
    return [layer for layer in layers if layer]


def make_stim_circ_noisy(circ: stim.Circuit, p: float) -> stim.Circuit:
    operations = [(op, targets) for (op, targets, params) in circ.flattened_operations() if op != "DETECTOR"]
    detectors = [(op, [stim.target_rec(targets[0][1])]) for (op, targets, params) in circ.flattened_operations() if
                 op == "DETECTOR"]
    operations = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(operations, circ.num_qubits)
    # final_ops, num_sim_qubits = apply_qubit_reuse(layered_ops)
    noisy_circ = layered_ops_to_noisy_stim_circuit(layered_ops + [detectors], circ.num_qubits, 0, p, 2 / 3 * p,
                                                   2 / 3 * p, p / 100, mem_error_after_every_cnot=True)
    return noisy_circ


def apply_qubit_reuse(layers: list[list[tuple[str, list[int]]]]) -> tuple[list[list[tuple[str, list[int]]]], int]:
    """
    Takes a temporally optimized list of layers and maps logical qubits
    to a minimal set of physical qubits.
    """
    # 1. Calculate the lifespan (birth layer, death layer) of every logical qubit
    births = {}
    deaths = {}

    for layer_idx, layer in enumerate(layers):
        for op_name, targets in layer:
            for t in targets:
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
        for op_name, targets in layer:
            mapped_targets = [logical_to_physical[t] for t in targets]
            new_layer.append((op_name, mapped_targets))
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


def load_qecc(code: str, method="FAO"):
    root = get_project_root()
    method = {
        "fao": "FAO",
        "mqt": "MQT",
    }.get(method, method)

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
        "17_1_5",
        "19_1_5",
        "25_1_5",
        "20_2_6",
        "31_1_7",
        "39_1_7"
    ]


if __name__ == "__main__":
    from spiderstate.cat_at_origin import row_optimized_cat_at_origin

    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc("20_2_6", "FAO")
    circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=5_000)
    operations = [(op, targets) for (op, targets, params) in circ.flattened_operations() if op != "DETECTOR"]
    detectors = [(op, [stim.target_rec(targets[0][1])]) for (op, targets, params) in circ.flattened_operations() if op == "DETECTOR"]
    print(detectors)
    operations = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(operations, circ.num_qubits)
    p = 0.001
    noisy_circ = layered_ops_to_noisy_stim_circuit(layered_ops + [detectors], circ.num_qubits, p, p, p, p, p/100)
    print(noisy_circ)
