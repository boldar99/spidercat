import itertools
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import stim
from qiskit import QuantumCircuit


TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}
Z_MEASUREMENTS = {"MR", "M", "MZ"}
X_MEASUREMENTS = {"MX"}
Z_INITIALIZATIONS = {"MR", "R"}
X_INITIALIZATIONS = {"RX"}
SPECIAL_GATES = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}


def graph_exists_with_girth(N, girth):
    if N % 2 != 0: raise ValueError("N must be even.")
    if N <= 2: return False
    if N % 2 != 0: return False
    if girth >= 6 and N < 14: return False
    if girth >= 7 and N < 24: return False
    if girth >= 8 and N < 30: return False
    if girth >= 9 and N < 58: return False
    return True


def qasm_to_stim(qasm_str: str) -> stim.Circuit:
    """
    Parses QASM 2.0 directly to a Stim circuit, bypassing Cirq to avoid
    deprecated import issues.
    """
    try:
        import qiskit.qasm2
        qc = qiskit.qasm2.loads(qasm_str)
    except (ImportError, AttributeError):
        qc = QuantumCircuit.from_qasm_str(qasm_str)

    qubit_map = {q: i for i, q in enumerate(qc.qubits)}
    stim_circuit = stim.Circuit()

    gate_translation = {
        'id': 'I', 'x': 'X', 'y': 'Y', 'z': 'Z',
        'h': 'H', 's': 'S', 'sdg': 'S_DAG',
        'sx': 'SQRT_X', 'sxdg': 'SQRT_X_DAG',  # Square-root X
        'cx': 'CNOT', 'cy': 'CY', 'cz': 'CZ', 'swap': 'SWAP',
        'reset': 'R', 'measure': 'M', 'barrier': 'TICK'
    }

    for instruction in qc.data:
        op = instruction.operation
        name = op.name
        indices = [qubit_map[q] for q in instruction.qubits]
        if name in gate_translation:
            stim_circuit.append(gate_translation[name], indices)
        else:
            raise ValueError(f"Gate '{name}' is not supported in Stim (Non-Clifford or Unknown).")

    return stim_circuit


def ed(v1: int, v2: int) -> tuple[int, int]:
    return (v1, v2) if v1 < v2 else (v2, v1)

def load_solution_triplet(n, t, p):
    root = get_project_root()
    file = root.joinpath( "circuits_data", f"cat_state_t{t}_n{n}_p{p}.json")
    if not file.exists():
        return None
    json_object = json.loads(file.read_text())

    G = nx.from_edgelist(json_object["G.edges"])
    M_inv = json_object["M_inv"]
    M = dict()
    for k, v in M_inv.items():
        for pair in v:
            M[tuple(pair)] = int(k)
    forest_edgelist = json_object.get("forest")
    forest = forest_edgelist and nx.from_edgelist(forest_edgelist)
    matching = {int(k): [tuple(l) for l in v] for k, v in json_object["matching"].items()}

    return G, forest, dict(M), matching


def get_project_root() -> Path:
    return Path(__file__).parent


def flatten(ls: list) -> list:
    return list(itertools.chain(*ls))


def implement_CNOT_circuit(cnots, num_qubits, p_2, p_mem):
    circ = stim.Circuit()
    all_qubits = set(range(num_qubits + 1))
    free_qubits = all_qubits.copy()
    for c, n in cnots:
        if c in free_qubits and n in free_qubits:
            free_qubits -= {c, n}
        else:
            if p_mem > 0:
                circ.append("DEPOLARIZE1", free_qubits, p_mem)
                circ.append("TICK")
                free_qubits = all_qubits.copy() - {c, n}
        circ.append("CNOT", [c, n])

        if p_2 > 0 and not c.is_measurement_record_target:
            circ.append("DEPOLARIZE2", [c, n], p_2)
    if p_mem > 0:
        circ.append("Z_ERROR", free_qubits, p_mem)
    return circ


def layered_ops_to_noisy_stim_circuit(
        layered_ops: list[list[tuple]],
        num_qubits: int,
        p_1: float,
        p_2: float,
        p_init: float,
        p_meas: float,
        p_mem: float
) -> tuple[stim.Circuit, dict[int, int]]:
    measurement_mapping = {}
    meas_id = 0
    circuit_str_lines = []

    def append_gate(name, t_list, p=0.0):
        if not t_list: return
        t_str = " ".join(map(str, t_list))
        if p > 0:
            circuit_str_lines.append(f"{name}({p}) {t_str}")
        else:
            circuit_str_lines.append(f"{name} {t_str}")

    for i, ops in enumerate(layered_ops):
        used_qubits = set()

        for item in ops:
            is_tagged_meas = isinstance(item[0], tuple)
            op_name = item[0][0] if is_tagged_meas else item[0]
            targets = item[1]

            used_qubits.update(targets)

            if is_tagged_meas:
                _, og_meas_id = item[0]
                measurement_mapping[meas_id] = og_meas_id
                meas_id += 1

            if op_name in Z_MEASUREMENTS and p_meas > 0:
                append_gate("X_ERROR", targets, p_meas)
            elif op_name in X_MEASUREMENTS and p_meas > 0:
                append_gate("Z_ERROR", targets, p_meas)

            append_gate(op_name, targets)

            if (op_name in X_INITIALIZATIONS or op_name in Z_INITIALIZATIONS) and p_init > 0:
                append_gate("DEPOLARIZE1", targets, p_init)
            elif op_name in TWO_QUBIT_GATES and p_2 > 0:
                append_gate("DEPOLARIZE2", targets, p_2)
            elif op_name not in SPECIAL_GATES and p_1 > 0:
                append_gate("DEPOLARIZE1", targets, p_1)

        if i != len(layered_ops) - 1 and p_mem > 0:
            unused_qubits = [q for q in range(num_qubits) if q not in used_qubits]
            if unused_qubits:
                append_gate("DEPOLARIZE1", unused_qubits, p_mem)

        circuit_str_lines.append("TICK")

    circuit_str = "\n".join(circuit_str_lines)
    return stim.Circuit(circuit_str), measurement_mapping


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


def _layer_circuit_ops(operations: list[tuple[str, list[int]]], num_qubits: int, one_cnot_per_layer: bool = False):
    all_qubits = range(num_qubits)
    next_free_layer = {q: 0 for q in all_qubits}
    asap_layers = defaultdict(list)
    meas_id = 0
    cx_layers = set()

    # --- PASS 1: ASAP Forward Layering ---
    for op_name, targets in operations:
        last_layer = max((next_free_layer[i] for i in targets), default=0)

        if one_cnot_per_layer and op_name in TWO_QUBIT_GATES:
            while last_layer in cx_layers:
                last_layer += 1
            cx_layers.add(last_layer)

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


def make_stim_circ_noisy(circ: stim.Circuit, p_1, p_2, p_init, p_meas, p_mem, one_cnot_per_layer: bool = False) -> tuple[
    stim.Circuit, dict[int, int]]:
    """Properly utilizes the layer structure to construct the noisy circuit."""
    operations = [(op, targets) for (op, targets, _) in circ.flattened_operations() if op != "DETECTOR"]

    expanded_ops = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(expanded_ops, circ.num_qubits, one_cnot_per_layer=one_cnot_per_layer)

    noisy_circ, mm = layered_ops_to_noisy_stim_circuit(
        layered_ops=layered_ops,
        num_qubits=circ.num_qubits,
        p_1=p_1,
        p_2=p_2,
        p_init=p_init,
        p_meas=p_meas,
        p_mem=p_mem
    )
    return noisy_circ, mm


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


def load_stim_circuit(n: int, t: int):
    my_file = get_project_root().joinpath("circuits", f"cat_state_t{t}_n{n}_p1.stim")
    if not my_file.is_file():
        return None

    return stim.Circuit(my_file.read_text())


if __name__ == "__main__":
    print(load_solution_triplet(33, 3, 1))
