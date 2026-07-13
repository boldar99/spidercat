import random

import networkx as nx
from matplotlib import pyplot as plt

from spidercat.circuit_extraction import expand_graph_and_forest, build_traversal_digraph, \
    resolve_dag_by_removing_missing_link
from spidercat.draw import draw_forest_on_graph, display_digraph
from spidercat.mdsf import constrained_mdsf_generation
from spidercat.spanning_tree import find_min_height_degree_3_roots
from spidercat.utils import load_solution_triplet


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
    return G, F, 0


def load_state_data(n, t):
    import json
    from pathlib import Path
    
    root = Path(__file__).parent
    file = root.joinpath("cat_states_data", f"well_ordered_state_t{t}_n{n}.json")
    if not file.exists():
        return None
        
    try:
        with open(file, "r") as f:
            data = json.load(f)
        G = nx.node_link_graph(data["G"], edges="links")
        F = nx.node_link_graph(data["F"], edges="links")
        D = nx.node_link_graph(data["D"], edges="links")
        roots = {int(k): v for k, v in data["roots"].items()}
        edge = data["edge"]
        return G, F, roots, D, edge
    except Exception as e:
        return None


def well_ordered_ft_cat_state_data(n, t, force_generate=False, regenerate_graph=False) -> tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, int]:
    if not force_generate:
        cached_data = load_state_data(n, t)
        if cached_data is not None:
            return cached_data

    max_retries = 10 if force_generate else 0
    for attempt in range(max_retries + 1):
        try:
            if n <= 3 or t == 0:
                G_alt, F_alt, root = G_F_alt_for_t_0(n)
                roots = {0: root}
                e = n
            elif t == 1 or n <= 5:
                G_alt, F_alt, root = G_F_alt_for_t_1(n)
                roots = {0: root}
            elif n == 6:
                G_alt, F_alt, root = G_F_n_6()
                roots = {0: root}
            else:
                if regenerate_graph:
                    from spidercat.generate import minimum_E_and_V, cat_state_FT_random
                    import numpy as np
                    
                    T_effective = min(t, int(np.floor(n / 2) - 1))
                    _, N_nodes = minimum_E_and_V(n, T_effective)
                    solution_triplet = cat_state_FT_random(n, N_nodes, T_effective, [1], max_new_graphs=100)
                    if solution_triplet is None:
                        raise ValueError(f"cat_state_FT_random failed to find a graph for n={n}, t={t}")
                    grf, spacing_trees, M = solution_triplet
                    tree = spacing_trees[1]
                    from spidercat.spanning_tree import match_forest_leaves_to_marked_edges
                    matchings = match_forest_leaves_to_marked_edges(grf, tree, M)
                else:
                    try:
                        grf, tree, M, matchings = load_solution_triplet(n, t, 1)
                    except TypeError:
                        if n == 26 and t == 7:
                            grf, tree, M, matchings = load_solution_triplet(27, 7, 1)
                        else:
                            grf, tree, M, matchings = load_solution_triplet({6: 21, 7: 24}.get(t), t, 1)
                        marks = [e for e, v in M.items() if v == 1]
                        for i in range(len(marks) - n):
                            M[marks[i]] = 0

                    if attempt > 0:
                        import random
                        from spidercat.markings import GraphMarker
                        from spidercat.spanning_tree import build_trivial_spanning_forest, build_min_diameter_spanning_tree, match_forest_leaves_to_marked_edges
                        from spidercat.utils import ed
                        import numpy as np
                        
                        nodes = list(grf.nodes())
                        random.shuffle(nodes)
                        mapping = {u: v for u, v in zip(grf.nodes(), nodes)}
                        inv_mapping = {v: k for k, v in mapping.items()}
                        shuffled_grf = nx.relabel_nodes(grf, mapping)
                        
                        marker = GraphMarker(shuffled_grf, max_marks=n)
                        T_effective = min(t, int(np.floor(n / 2) - 1))
                        shuffled_marks = marker.find_solution(T_effective)
                        if sum(shuffled_marks.values()) != n:
                            raise ValueError("Failed to find marking")
                            
                        M = {}
                        for (u, v), val in shuffled_marks.items():
                            M[ed(inv_mapping[u], inv_mapping[v])] = val
                            
                        forest = build_trivial_spanning_forest(grf, M)
                        tree = build_min_diameter_spanning_tree(grf, forest, M, 1)
                        matchings = match_forest_leaves_to_marked_edges(grf, tree, M)

                G_alt, _ = expand_graph_and_forest(grf, tree, M, matchings, expand_flags=False)
                F_alt = constrained_mdsf_generation(G_alt, 1, seed=9001 + attempt, cooling_rate=0.995)
                F_alt = F_alt.copy()
                roots = find_min_height_degree_3_roots(F_alt)
            D = build_traversal_digraph(G_alt, F_alt, roots[0])

            pos = nx.spring_layout(G_alt)
            _, edge, dependency_graph = resolve_dag_by_removing_missing_link(D)
            assert nx.is_directed_acyclic_graph(dependency_graph)

            return G_alt, F_alt, roots, dependency_graph, edge[0][0] if len(edge) else e
            
        except Exception as ex:
            if attempt == max_retries:
                raise ex

if __name__ == "__main__":
    random.seed(1)
    from spidercat.circuit_extraction import CatStateExtractor, StimBuilder

    G_alt, F_alt, roots, dependency_graph, edge = well_ordered_ft_cat_state_data(22, 7)
    extractor = CatStateExtractor(StimBuilder(), verbose=False)
    circ = extractor.extract(G_alt, F_alt, roots, dependency_graph)
    print(circ)
