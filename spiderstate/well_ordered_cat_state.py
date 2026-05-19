import networkx as nx

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


def well_ordered_ft_cat_state_data(n, t) -> tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, int]:
    if n <= 3:
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
        grf, tree, M, matchings = load_solution_triplet(n, t, 1)
        G_alt, _ = expand_graph_and_forest(grf, tree, M, matchings, expand_flags=False)
        F_alt = constrained_mdsf_generation(G_alt, 1, seed=9001, cooling_rate=0.995)
        F_alt = F_alt.copy()
        roots = find_min_height_degree_3_roots(F_alt)
    D = build_traversal_digraph(G_alt, F_alt, roots[0])

    # display_digraph(D, figsize=(6,6))
    _, edge, dependency_graph = resolve_dag_by_removing_missing_link(D)
    assert nx.is_directed_acyclic_graph(dependency_graph)

    return G_alt, F_alt, roots, dependency_graph, edge[0][0] if len(edge) else e

if __name__ == "__main__":
    G_alt, F_alt, roots, dependency_graph, edge = well_ordered_ft_cat_state_data(10, 4)
    draw_forest_on_graph(G_alt, F_alt, figsize=(7,7))
    display_digraph(dependency_graph, figsize=(9,5))
