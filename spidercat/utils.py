import itertools
import json
from pathlib import Path

import networkx as nx
import stim


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
        from qiskit import QuantumCircuit

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


if __name__ == "__main__":
    print(load_solution_triplet(33, 3, 1))
