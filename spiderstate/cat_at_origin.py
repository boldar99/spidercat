import networkx as nx
import numpy as np
import stim

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder
from spidercat.draw import draw_forest_on_graph, display_digraph
from spiderstate.spider_leg_matcher import match_edges
from spiderstate.utils import find_pivots_in_matrix
from spiderstate.well_ordered_cat_state import well_ordered_ft_cat_state_data
from spiderstate.optimize_parity_matrix import has_unique_ones_property, optimize_fault_tolerant_matrix, row_optimize_matrix


def col_reduced_cat_at_origin(H: np.ndarray, d: int, max_col_ops: int = 0, max_basis_tries: int = 5000):
    t = (d - 1) // 2
    _, final_matrix_after_col_ops, col_ops_performed = optimize_fault_tolerant_matrix(H, t, max_col_ops, max_basis_tries)
    circ = cat_at_origin(final_matrix_after_col_ops, d)
    for (c, n) in col_ops_performed:
        circ.append("CX", [c, n])

    return circ


def row_optimized_cat_at_origin(H: np.ndarray, d: int, max_basis_tries: int = 10_000):
    t = (d - 1) // 2
    best_row_op_cost, matrix_after_row_ops = row_optimize_matrix(H, t, max_basis_tries)
    return cat_at_origin(matrix_after_row_ops, d)


def cat_at_origin(H: np.ndarray, d: int, draw_solutions=False) -> stim.Circuit:
    if not has_unique_ones_property(H):
        raise ValueError(f"H is not representing a bipartite graph state.")

    N = H.shape[1]
    t = (d - 1) // 2

    pivots, rows_without_pivots = find_pivots_in_matrix(H)
    pivots_perm = [row for row, col in sorted(pivots.items(), key=lambda item: item[1])]
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(H, axis=1)
    x_spiders = np.sum(H[:, non_pivots], axis=0) + 1

    z_data = [well_ordered_ft_cat_state_data(zs, t) for zs in z_spiders]
    x_data = [well_ordered_ft_cat_state_data(xs, t) for xs in x_spiders]
    z_graphs, x_graphs, z_trees, x_trees, z_mains, x_mains = [], [], [], [], [], []
    z_digraphs, x_digraphs = [], []
    z_candidates, x_candidates = [], []
    z_roots, x_roots = [], []
    for (G, F, roots, D, e) in z_data:
        nx.set_node_attributes(G, "Z", 'spider_type')
        z_graphs.append(G);
        z_trees.append(F);
        z_roots.append(roots)
        z_digraphs.append(D);
        z_mains.append(e)

        # Flatten topological generations into prioritized 1D candidate pools
        cands = []
        for layer in nx.topological_generations(D):
            cands.extend([l for l in layer if l != e and G.nodes[l].get("is_mark", False)])
        z_candidates.append(cands)

    for (G, F, roots, D, e) in x_data:
        nx.set_node_attributes(G, "X", 'spider_type')
        x_graphs.append(G);
        x_trees.append(F);
        x_roots.append(roots)
        x_digraphs.append(D);
        x_mains.append(e)

        cands = []
        for layer in nx.topological_generations(D):
            cands.extend([l for l in layer if l != e and G.nodes[l].get("is_mark", False)])
        x_candidates.append(cands)

    matched_edges = match_edges(H, non_pivots, z_digraphs, x_digraphs, z_candidates, x_candidates)

    # Build global graphs
    z_node_mapping: dict[tuple[int, int], int] = {}
    x_node_mapping: dict[tuple[int, int], int] = {}
    global_G = nx.Graph()
    global_F = nx.Graph()
    global_roots = {}
    global_D = nx.DiGraph()
    global_primary_paths = {}
    i, j = 0, 0
    k = 0
    non_pivots_set = set(non_pivots)

    # Phase 1: Adding each individual cat state to the global graphs
    while i + j < N:
        curr_col = i + j
        is_non_pivot = curr_col in non_pivots_set

        if is_non_pivot:
            graph, trees, digraph = x_graphs[i], x_trees[i], x_digraphs[i]
            node_mapping, root, index = x_node_mapping, x_roots[i], i
        else:
            row = pivots_perm[j]
            graph, trees, digraph = z_graphs[row], z_trees[row], z_digraphs[row]
            node_mapping, root, index = z_node_mapping, z_roots[row], row
        for node, data in graph.nodes(data=True):
            node_mapping[(index, node)] = k
            global_G.add_node(k, **data)
            global_F.add_node(k)
            global_D.add_node(k)
            k += 1
        for u, v, data in graph.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_G.add_edge(u_prime, v_prime, **data)
        for u, v, data in trees.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_F.add_edge(u_prime, v_prime, **data)
        for u, v, data in digraph.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_D.add_edge(u_prime, v_prime, **data)

        global_roots[i + j] = node_mapping[(index, root[0])]
        global_primary_paths[i + j] = nx.shortest_path(
            global_F,
            source=global_roots[i + j],
            target=node_mapping[(index, x_mains[i] if i + j in non_pivots else z_mains[pivots_perm[j]])]
        )
        if i + j in non_pivots:
            i += 1
        else:
            j += 1

    # Phase 1: Connecting the cat states in the global graphs
    while matched_edges:
        (z_graph, x_graph), (z_val, x_val) = matched_edges.pop(0)
        global_G.add_edge(z_node_mapping[(z_graph, z_val)], x_node_mapping[(x_graph, x_val)], edge_type="cnot")
        global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_mark"] = False
        global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_mark"] = False

        if global_F.degree(z_node_mapping[(z_graph, z_val)]) == 1:
            global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_flag"] = True
        if global_F.degree(x_node_mapping[(x_graph, x_val)]) == 1:
            global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_flag"] = True

        for u, _ in z_digraphs[z_graph].in_edges(z_val):
            global_D.add_edge(z_node_mapping[(z_graph, u)], x_node_mapping[(x_graph, x_val)], edge_type="cnot")
        for u, _ in x_digraphs[x_graph].in_edges(x_val):
            global_D.add_edge(x_node_mapping[(x_graph, u)], z_node_mapping[(z_graph, z_val)], edge_type="cnot")

    # Extract circuit using the global graphs
    extractor = CatStateExtractor(StimBuilder(), verbose=False)
    if draw_solutions:
        draw_forest_on_graph(global_G, global_F, figsize=(8, 8))
        display_digraph(global_D, figsize=(8, 8))
    circ = extractor.extract(global_G, global_F, global_roots, global_D, global_primary_paths)
    return circ


if __name__ == "__main__":
    H_x, d = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1]
    ]), 3
    circ = cat_at_origin(H_x, d)
