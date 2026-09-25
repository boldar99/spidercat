import itertools
import json
from pathlib import Path

import networkx as nx
import stim
from qiskit import QuantumCircuit


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


def make_stim_circ_noisy(circ: stim.Circuit, p_1=0., p_2=0., p_mem=0., p_meas=0., p_init=0.) -> stim.Circuit:
    noisy_circ = stim.Circuit()
    num_qubits = circ.num_qubits

    if p_init > 0:
        noisy_circ.append("DEPOLARIZE1", range(num_qubits), p_init)

    for instruction in circ:
        gate_name = instruction.name
        targets = instruction.targets_copy()

        if gate_name in ("CNOT", "CX", "CZ", "SWAP"):
            split_targets = [
                (targets[i], targets[i+1])
                for i in range(0, len(targets), 2)
            ]
            noisy_circ += implement_CNOT_circuit(split_targets, num_qubits, p_2, p_mem)

        elif gate_name in ("H", "X", "Y", "Z", "I"):
            noisy_circ.append(gate_name, targets)
            if p_1 > 0:
                noisy_circ.append("DEPOLARIZE1", targets, p_1)

        elif gate_name in ("M", "MZ", "MR", "R", "RX", "RY"):
            if gate_name in ("M", "MZ", "MR") and p_meas > 0:
                noisy_circ.append("DEPOLARIZE1", targets, p_meas)

            noisy_circ.append(gate_name, targets)

            if gate_name in ("R", "RX", "RY", "MR") and p_init > 0:
                noisy_circ.append("DEPOLARIZE1", targets, p_init)

        else:
            noisy_circ.append(gate_name, targets, instruction.gate_args_copy())

    return noisy_circ


if __name__ == "__main__":
    print(load_solution_triplet(33, 3, 1))
