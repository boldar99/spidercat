from typing import Literal, Sequence

import stim

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder, expand_graph_and_forest
from spidercat.path_cover import find_all_path_covers
from spidercat.spanning_tree import match_forest_leaves_to_marked_edges
from spidercat.utils import load_solution_triplet


def syndrome_measurement_circuit(qubits: Sequence[int], ancilla_start: int, t: int, basis: Literal["X"] | Literal["Z"] = "Z") -> stim.Circuit:
    n = len(qubits) + 1
    G, _, M, _ = load_solution_triplet(n, t, 1)
    paths = next(find_all_path_covers(G, 1))
    H = G.copy()
    H.remove_edges_from(G.edges())
    path = paths[0]
    for v, w in zip(path, path[1:]):
        H.add_edge(v, w)

    matchings = match_forest_leaves_to_marked_edges(G, H, M)
    extractor = CatStateExtractor(StimBuilder(), verbose=False)
    # Visualize without matchings so that unmatched marks are not covered
    G_exp_vis, F_exp_vis = expand_graph_and_forest(G, H, M, {path[-1]: matchings[path[-1]]}, expand_flags=False)
    for n in G_exp_vis.nodes:
        G_exp_vis.nodes[n]["spider_type"] = basis

    # Extract the circuit using the unmatched graph
    circ = extractor.extract(G_exp_vis, F_exp_vis, {0: path[0]})
    
    q_to_mapped = {}
    q_to_mapped[0] = ancilla_start
    for v in range(1, len(qubits) + 1):
        q_to_mapped[v] = qubits[v - 1]

    ancilla_idx = ancilla_start + 1
    for q in range(circ.num_qubits):
        if q not in q_to_mapped:
            q_to_mapped[q] = ancilla_idx
            ancilla_idx += 1

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

    permuted_circuit.append("M" if basis == "X" else "MX", ancilla_start)

    return permuted_circuit


if __name__ == '__main__':
    N, t = 10, 4
    circ = syndrome_measurement_circuit(qubits=range(N), ancilla_start=N, t=t, basis="X")
    circ.append("M", range(N))
    print(circ.compile_sampler().sample(10))
    print(circ.diagram())
