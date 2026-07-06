from __future__ import annotations

from collections import defaultdict
from typing import Literal, TYPE_CHECKING

import pyzx as zx
import stim
import stimcirq
from cirq.contrib.qasm_import import circuit_from_qasm

from spiderstate.utils import flatten


TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}
Z_MEASUREMENTS = {"MR", "M", "MZ"}
X_MEASUREMENTS = {"MX"}
Z_INITIALIZATIONS = {"MR", "R"}
X_INITIALIZATIONS = {"RX"}
SPECIAL_GATES = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


def qasm_str_to_stim_circuit(qasm_str: str) -> stim.Circuit:
    cirq_circuit = circuit_from_qasm(qasm_str)
    return stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit)


def stim_to_pyzx(stim_circuit: stim.Circuit, n_data: int) -> zx.Graph:
    circ = zx.Circuit(n_data)

    for op, targets, _ in stim_circuit.flattened_operations():
        if op in ("R", "RX"):
            for t in targets:
                if t < n_data:
                    continue
                if op == "R":
                    circ.add_gate("InitAncilla", label=t, state="0")
                elif op == "RX":
                    circ.add_gate("InitAncilla", label=t, state="+")

        elif op == "CX":
            for i in range(0, len(targets), 2):
                c, n = targets[i], targets[i + 1]
                circ.add_gate("CNOT", c, n)

        elif op == "H":
            for t in targets:
                circ.add_gate("H", t)

        elif op in ("M", "MX", "MR"):
            for t in targets:
                if op in ("M", "MR"):
                    circ.add_gate("PostSelect", label=t, state="0")
                elif op == "MX":
                    circ.add_gate("PostSelect", label=t, state="+")

    return circ.to_graph()


def explode_circuit(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """
    Decomposes a circuit into a list of atomic instructions.
    E.g., 'CX 0 1 2 3' becomes ['CX 0 1', 'CX 2 3'].
    This allows injecting faults *between* gates that were originally grouped.
    """
    atomized_ops = []

    for op in circuit.flattened():
        # Handle 2-Qubit Gates (Target pairs)
        if op.name in TWO_QUBIT_GATES:
            targets = op.targets_copy()
            # Iterate in steps of 2
            for k in range(0, len(targets), 2):
                atomized_ops.append(
                    stim.CircuitInstruction(op.name, targets[k:k + 2], op.gate_args_copy())
                )

        # Handle Annotations (Don't split, just keep)
        elif op.name in SPECIAL_GATES:
            atomized_ops.append(op)

        # Handle 1-Qubit Gates & Measurements (Single targets)
        else:
            # e.g. H, X, Z, M, R, MR
            targets = op.targets_copy()
            for t in targets:
                atomized_ops.append(
                    stim.CircuitInstruction(op.name, [t], op.gate_args_copy())
                )

    return atomized_ops


def steane_se_from_stim_state_prep(circ: stim.Circuit, se_basis: Literal["X"] | Literal["Z"], n: int, offset = 0) -> stim.Circuit:
    ret = stim.Circuit()
    for op in circ:
        targets = [stim.GateTarget(t.value + n + offset) for t in op.targets_copy()]
        new_op = stim.CircuitInstruction(op.name, targets, op.gate_args_copy())
        ret.append(new_op)
    if se_basis == "Z":
        ret.append("CX", flatten(zip(range(n + offset, 2 * n + offset), range(n))))
        ret.append("MX", range(n + offset, 2 * n + offset))
    elif se_basis == "X":
        ret.append("CX", flatten(zip(range(n), range(n + offset, 2 * n + offset))))
        ret.append("M", range(n + offset, 2 * n + offset))
    else:
        raise Exception("Unknown se_basis: {}".format(se_basis))

    return ret


def _get_target_values(targets: list[stim.GateTarget]) -> list[int]:
    """Safely extracts integer indices from Stim GateTargets, ignoring records."""
    return [t.value for t in targets if t.is_qubit_target]


def _expand_stim_operation_list(operations: list[tuple[str, list[stim.GateTarget | int]]]):
    stim_operations = []
    for op_name, targets in operations:
        if isinstance(targets, stim.GateTarget):
            t_vals = _get_target_values(targets)
        else:
            t_vals = targets
        if not targets:
            continue

        if op_name in TWO_QUBIT_GATES:
            for i in range(0, len(t_vals), 2):
                stim_operations.append((op_name, [t_vals[i], t_vals[i + 1]]))
        elif op_name in SPECIAL_GATES:
            stim_operations.append((op_name, t_vals))
        else:
            for t in t_vals:
                stim_operations.append((op_name, [t]))
    return stim_operations


def _layer_circuit_ops(operations: list[tuple[str, list[int]]], num_qubits: int):
    all_qubits = range(num_qubits)
    next_free_layer = {q: 0 for q in all_qubits}
    asap_layers = defaultdict(list)
    meas_id = 0

    # --- PASS 1: ASAP Forward Layering ---
    for op_name, targets in operations:
        last_layer = max((next_free_layer[i] for i in targets), default=0)

        if op_name in ("M", "MX"):
            asap_layers[last_layer].append(((op_name, meas_id), targets))
            meas_id += 1
        else:
            asap_layers[last_layer].append((op_name, targets))

        for i in targets:
            next_free_layer[i] = last_layer + 1

    max_layer = max(asap_layers.keys(), default=-1)
    if max_layer == -1:
        return []

    layers = [asap_layers[i] for i in range(max_layer + 1)]

    # --- PASS 2: ALAP Backward Reset Shifting ---
    next_required = {q: len(layers) for q in all_qubits}

    for i in range(len(layers) - 1, -1, -1):
        current_layer_ops = layers[i]
        kept_ops = []

        for item in current_layer_ops:
            # Handle tuple unpacking for tagged measurements
            is_tagged_meas = isinstance(item[0], tuple)
            op_name = item[0][0] if is_tagged_meas else item[0]
            targets = item[1]

            if op_name in {"R", "RX"}:
                for t in targets:
                    target_layer = next_required[t] - 1
                    if target_layer > i:
                        layers[target_layer].append((op_name, [t]))
                    else:
                        kept_ops.append((op_name, [t]))
            else:
                kept_ops.append(item)
                for t in targets:
                    next_required[t] = i

        layers[i] = kept_ops

    return [layer for layer in layers if layer]


def layered_ops_to_noisy_stim_circuit(
    layered_ops: list[list[tuple]],
    num_qubits: int,
    p_1: float,
    p_2: float,
    p_init: float,
    p_meas: float,
    p_mem: float
) -> tuple[stim.Circuit, dict[int, int]]:
    circuit = stim.Circuit()
    measurement_mapping = {}
    meas_id = 0

    for i, ops in enumerate(layered_ops):
        # We can now safely subtract ints from ints
        unused_qubits = set(range(num_qubits))

        for item in ops:
            is_tagged_meas = isinstance(item[0], tuple)
            op_name = item[0][0] if is_tagged_meas else item[0]
            targets = item[1]

            unused_qubits -= set(targets)

            if is_tagged_meas:
                _, og_meas_id = item[0]
                measurement_mapping[meas_id] = og_meas_id
                meas_id += 1

            if op_name in Z_MEASUREMENTS and p_meas > 0:
                circuit.append("X_ERROR", targets, p_meas)
            elif op_name in X_MEASUREMENTS and p_meas > 0:
                circuit.append("Z_ERROR", targets, p_meas)

            circuit.append(op_name, targets)

            if op_name in X_INITIALIZATIONS and p_init > 0:
                circuit.append("Z_ERROR", targets, p_init)
            elif op_name in Z_INITIALIZATIONS and p_init > 0:
                circuit.append("X_ERROR", targets, p_init)
            elif op_name in TWO_QUBIT_GATES and p_2 > 0:
                circuit.append("DEPOLARIZE2", targets, p_2)
            elif op_name not in SPECIAL_GATES and p_1 > 0:
                circuit.append("DEPOLARIZE1", targets, p_1)

        if i != len(layered_ops) - 1 and p_mem > 0 and unused_qubits:
            circuit.append("DEPOLARIZE1", sorted(list(unused_qubits)), p_mem)

        circuit.append("TICK", [])

    return circuit, measurement_mapping


def make_stim_circ_noisy(circ: stim.Circuit, p: float, one_cnot_per_layer: bool=False) -> tuple[stim.Circuit, dict[int, int]]:
    """Properly utilizes the layer structure to construct the noisy circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op != "DETECTOR"]

    expanded_ops = _expand_stim_operation_list(operations)
    if one_cnot_per_layer:
        layered_ops = [[op] for op in expanded_ops]
    else:
        layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)

    noisy_circ, mm = layered_ops_to_noisy_stim_circuit(
        layered_ops=layered_ops,
        num_qubits=circ.num_qubits,
        p_1=0,
        p_2=p,
        p_init=(2 / 3) * p,
        p_meas=(2 / 3) * p,
        p_mem=p / 100
    )
    return noisy_circ, mm



def get_circuit_depth(circ: stim.Circuit) -> int:
    """Returns the strict ASAP depth of the circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op not in SPECIAL_GATES]
    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)
    return len(layered_ops)

def get_circuit_width_per_timestep(circ: stim.Circuit) -> list[int]:
    """Returns the strict ASAP depth of the circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op not in SPECIAL_GATES]
    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits)
    return [len(ops) for ops in layered_ops]


def get_num_cnots(circ: stim.Circuit) -> int:
    raw_cnots = [l for (name, l, _) in circ.flattened_operations() if name in ("CX", "CNOT")]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    return len(cnots)


def get_num_measurements(circ: stim.Circuit) -> int:
    return sum(
        len(l) for (name, l, _) in circ.flattened_operations() if name in Z_MEASUREMENTS or name in X_MEASUREMENTS
    )


def get_spacetime_volume(circ: stim.Circuit) -> int:
    """Calculates the sum of active ticks for all qubits across the circuit."""
    return get_circuit_depth(circ) * circ.num_qubits
