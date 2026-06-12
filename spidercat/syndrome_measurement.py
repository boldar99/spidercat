from typing import Iterable

import networkx as nx
import stim

from spidercat.circuit_extraction import extract_circuit_rooted, CatStateExtractor, StimBuilder, expand_graph_and_forest
from spidercat.draw import draw_qubit_lines_state, draw_spanning_forest_solution, draw_forest_on_graph
from spidercat.path_cover import find_all_path_covers
from spidercat.spanning_tree import match_forest_leaves_to_marked_edges
from spidercat.utils import load_solution_triplet


def syndrome_measurement_circuit(qubits: Iterable[int], t: int, num_measurement_ancillae = 1) -> stim.Circuit:
    n = len(qubits)
    G, _, M, _ = load_solution_triplet(n, t, 1)
    paths = next(find_all_path_covers(G, 1))
    H = G.copy()
    H.remove_edges_from(G.edges())
    path = paths[0]
    for v, w in zip(path, path[1:]):
        H.add_edge(v, w)

    matchings = match_forest_leaves_to_marked_edges(G, H, M)
    extractor = CatStateExtractor(StimBuilder(), verbose=True)
    G_exp, F_exp = expand_graph_and_forest(G, H, M, matchings)
    draw_forest_on_graph(G_exp, F_exp)
    D = nx.DiGraph()
    return extractor.extract(G_exp, F_exp, {0: path[0]})


if __name__ == '__main__':
    circ = syndrome_measurement_circuit(qubits=range(7), t=3, num_measurement_ancillae=1)
    circ.append("M", range(7))
    print(circ.compile_sampler().sample(10))
