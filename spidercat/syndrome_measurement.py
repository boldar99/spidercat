from typing import Literal, Sequence

import networkx as nx
import stim

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder, expand_graph_and_forest
from spidercat.draw import draw_forest_on_graph
from spidercat.path_cover import find_all_path_covers
from spidercat.spanning_tree import match_forest_leaves_to_marked_edges
from spidercat.utils import load_solution_triplet, get_project_root, qasm_to_stim


def G_F_alt_for_t_0(N) -> tuple[nx.Graph, nx.Graph, int]:
    G = nx.Graph()
    G.add_nodes_from([0])
    G.add_nodes_from(range(1, N + 1), is_mark=True)
    for i in range(N):
        G.add_edge(i, i + 1)
    F = G.copy()
    return G, F, 0


def G_F_alt_for_t_1(N) -> tuple[nx.Graph, nx.Graph, int]:
    G = nx.Graph()
    G.add_nodes_from([0])
    G.add_nodes_from(range(2, 2 + N), is_mark=True)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    for i in range(N - 2):
        G.add_edge(2 + i, 4 + i)
    G.add_edge(N, N + 1)
    F = G.copy()
    F.remove_edge(N + 1, N)
    F.remove_edge(0, 3)
    for i in range(3, N, 2):
        F.remove_edge(i, i + 2)
    return G, F, 0


def G_F_n_6() -> tuple[nx.Graph, nx.Graph, int]:
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_nodes_from(range(2, 8), is_mark=True)
    for i in range(3):
        G.add_edge(0, i + 2)
        G.add_edge(1, i + 5)
        G.add_edge(i + 2, i + 5)

    F = G.copy()
    F.remove_edge(0, 4)
    F.remove_edge(1, 5)
    F.remove_edge(2, 5)
    F.remove_edge(2, 0)
    return G, F, 0


def syndrome_measurement_circuit(qubits: Sequence[int], ancilla_start: int, t: int, basis: Literal["X"] | Literal["Z"] = "Z") -> stim.Circuit:
    n = len(qubits) + 1
    other_basis = "X" if basis == "Z" else "Z"

    if n <= 4:
        G_exp_vis, F_exp_vis, root = G_F_alt_for_t_0(n)
        roots = {0: root}
        e = n
    elif t == 1 or n <= 6:
        G_exp_vis, F_exp_vis, root = G_F_alt_for_t_1(n)
        roots = {0: root}
    elif n == 7:
        G_exp_vis, F_exp_vis, root = G_F_n_6()
        roots = {0: root}
    else:
        G, _, M, _ = load_solution_triplet(n, t, 1)
        paths = next(find_all_path_covers(G, 1))
        H = G.copy()
        H.remove_edges_from(G.edges())
        path = paths[0]
        for v, w in zip(path, path[1:]):
            H.add_edge(v, w)

        matchings = match_forest_leaves_to_marked_edges(G, H, M)
        G_exp_vis, F_exp_vis = expand_graph_and_forest(G, H, M, {path[-1]: matchings[path[-1]]}, expand_flags=False)
        roots = {0: path[0]}

    for n in G_exp_vis.nodes:
        G_exp_vis.nodes[n]["spider_type"] = other_basis

    # draw_forest_on_graph(G_exp_vis, F_exp_vis)
    # Extract the circuit using the unmatched graph
    extractor = CatStateExtractor(StimBuilder(), verbose=False)
    circ = extractor.extract(G_exp_vis, F_exp_vis, roots)

    q_to_mapped = {}
    q_to_mapped[0] = ancilla_start
    for v in range(1, len(qubits) + 1):
        q_to_mapped[v] = qubits[v - 1]

    ancilla_idx = ancilla_start + 1
    for q in range(circ.num_qubits):
        if q not in q_to_mapped:
            q_to_mapped[q] = ancilla_idx
            ancilla_idx += 1

    permuted_circuit = permute_circ(circ, q_to_mapped, qubits)

    permuted_circuit.append("M" if basis == "Z" else "MX", ancilla_start)

    return permuted_circuit


def permute_circ(circ: stim.Circuit, q_to_mapped: dict[int, int], qubits: Sequence[int]) -> stim.Circuit:
    permuted_circuit = stim.Circuit()
    data_qubits_set = set(qubits)

    for op in circ:
        new_targets = []
        for t in op.targets_copy():
            if t.is_qubit_target:
                mapped_q = q_to_mapped[t.value]
                if mapped_q in data_qubits_set and op.name in ["R", "RX", "H"]:
                    continue
                new_targets.append(mapped_q)
            else:
                new_targets.append(t)

        if new_targets:
            permuted_circuit.append(op.name, new_targets, op.gate_args_copy())
    return permuted_circuit


def fao_se_circuit(qubits: Sequence[int], ancilla_start: int, t: int, basis: Literal["X"] | Literal["Z"] = "Z") -> stim.Circuit:
    n = len(qubits)

    file = get_project_root().joinpath("flag_at_origin_circuits", f"d{2 * t + 1}-q{n + 1}-GHZ.qasm")
    qasm_str = file.read_text()

    fao_circ = qasm_to_stim(qasm_str)

    q_to_mapped = {}
    num_ancilla = fao_circ.num_qubits - n
    for q in range(num_ancilla - 1):
        q_to_mapped[q] = ancilla_start + q + 1
    q_to_mapped[num_ancilla - 1] = ancilla_start
    for q in range(n):
        q_to_mapped[num_ancilla + q] = qubits[q]

    permuted_circuit = permute_circ(fao_circ, q_to_mapped, qubits)
    permuted_circuit.insert(0, stim.CircuitInstruction("R", range(ancilla_start, permuted_circuit.num_qubits)))
    permuted_circuit.append("M", range(ancilla_start + 1, permuted_circuit.num_qubits))
    for q in range(permuted_circuit.num_qubits - ancilla_start - 1):
        permuted_circuit.append("DETECTOR", stim.target_rec(-q - 1))
    permuted_circuit.append("MX", [ancilla_start])
    if basis == "Z":
        permuted_circuit.insert(0, stim.CircuitInstruction("H", qubits))
        permuted_circuit.append(stim.CircuitInstruction("H", qubits))

    return permuted_circuit

if __name__ == '__main__':
    N, t = 6, 5
    circ = fao_se_circuit(qubits=[3, 6, 7, 10, 11, 12], ancilla_start=21, t=t, basis="X")
    # circ.append("M", range(N))
    print(circ.compile_sampler().sample(10))
    print(circ.diagram())
