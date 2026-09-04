"""
Well-ordered fault-tolerant cat state synthesis and data management.

This module generates and caches graph data structures representing fault-tolerant
Greenberger-Horne-Zeilinger (GHZ) / cat states with a traversal directed acyclic
graph (DAG). The DAG guarantees a well-ordered dependency schedule for fault-tolerant
syndrome extraction and circuit synthesis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
from typing import Sequence

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from spidercat.circuit_extraction import (
    build_traversal_digraph,
    expand_graph_and_forest,
    resolve_dag_by_removing_missing_link,
)
from spidercat.draw import draw_forest_on_graph
from spidercat.generate import cat_state_FT_random, minimum_E_and_V
from spidercat.markings import GraphMarker
from spidercat.mdsf import constrained_mdsf_generation
from spidercat.spanning_tree import (
    build_min_diameter_spanning_tree,
    build_trivial_spanning_forest,
    find_min_height_degree_k_roots,
    match_forest_leaves_to_marked_edges,
)
from spidercat.utils import ed, load_solution_triplet

logger = logging.getLogger(__name__)

# Constants
CAT_STATES_DATA_DIR: Path = Path(__file__).parent / "cat_states_data"
DEFAULT_MAX_RETRIES: int = 10
DEFAULT_COOLING_RATE: float = 0.995
MDSF_SEED_BASE: int = 9001
FALLBACK_TRIPLET_SIZES: dict[int, int] = {6: 21, 7: 24}


def build_base_chain_graph(n: int) -> tuple[nx.Graph, nx.Graph, int]:
    """
    Constructs a linear chain graph and spanning tree for t=0 or n<=3.

    Args:
        n: Number of marked qubits.

    Returns:
        tuple of (interaction_graph, spanning_forest, root_node_id)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n), is_mark=True)
    for i in range(n - 1):
        graph.add_edge(i, i + 1)

    forest = graph.copy()
    return graph, forest, 0


def build_base_t1_graph(n: int) -> tuple[nx.Graph, nx.Graph, int]:
    """
    Constructs base graph and spanning tree for t=1 or n<=5.

    Args:
        n: Number of marked qubits.

    Returns:
        tuple of (interaction_graph, spanning_forest, root_node_id)
    """
    graph = nx.Graph()
    graph.add_nodes_from([0])
    graph.add_nodes_from(range(2, 2 + n), is_mark=True)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    for i in range(n - 2):
        graph.add_edge(2 + i, 4 + i)
    graph.add_edge(n, n + 1)

    forest = graph.copy()
    forest.remove_edge(n + 1, n)
    return graph, forest, 0


def build_base_n6_graph() -> tuple[nx.Graph, nx.Graph, int]:
    """
    Constructs base bipartite graph and spanning tree for n=6.

    Returns:
        tuple of (interaction_graph, spanning_forest, root_node_id)
    """
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    graph.add_nodes_from(range(2, 8), is_mark=True)
    for i in range(3):
        graph.add_edge(0, i + 2)
        graph.add_edge(1, i + 5)
        graph.add_edge(i + 2, i + 5)

    forest = graph.copy()
    forest.remove_edge(0, 4)
    forest.remove_edge(1, 5)
    return graph, forest, 0


# Backward compatibility aliases
G_F_alt_for_t_0 = build_base_chain_graph
G_F_alt_for_t_1 = build_base_t1_graph
G_F_n_6 = build_base_n6_graph


def load_state_data(
    n: int, t: int, data_dir: Path = CAT_STATES_DATA_DIR
) -> tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, int] | None:
    """
    Loads precomputed well-ordered cat state data structures from disk.

    Args:
        n: Number of marked qubits.
        t: Target fault-tolerance distance / parameter.
        data_dir: Directory containing cached JSON state files.

    Returns:
        A tuple of (G, F, roots, D, edge) if the cached file exists and is valid,
        or None otherwise.
    """
    file_path = data_dir / f"well_ordered_state_t{t}_n{n}.json"
    if not file_path.is_file():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data["G"], edges="links")
        forest = nx.node_link_graph(data["F"], edges="links")
        traversal_dag = nx.node_link_graph(data["D"], edges="links")
        roots = {int(k): v for k, v in data["roots"].items()}
        edge = data["edge"]
        return graph, forest, roots, traversal_dag, edge
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Failed to load cached cat state data for n=%d, t=%d from %s: %s",
            n,
            t,
            file_path,
            e,
        )
        return None


def _build_base_case(
    n: int, t: int
) -> tuple[nx.Graph, nx.Graph, dict[int, int], int] | None:
    """
    Constructs known analytical base cases for small n or t.

    Returns:
        tuple of (graph, forest, roots, fallback_node) if a base case applies,
        or None if general synthesis is required.
    """
    if n <= 3 or t == 0:
        graph, forest, root = build_base_chain_graph(n)
        return graph, forest, {0: root}, n - 1
    if t == 1 or n <= 5:
        graph, forest, root = build_base_t1_graph(n)
        return graph, forest, {0: root}, n
    if n == 6:
        graph, forest, root = build_base_n6_graph()
        return graph, forest, {0: root}, 0
    return None


def _generate_random_solution(
    n: int, t: int
) -> tuple[nx.Graph, nx.Graph, dict[tuple[int, int], int], dict[int, list[tuple[int, int]]]]:
    """
    Generates a new random fault-tolerant cat state graph triplet.
    """
    effective_t = min(t, int(np.floor(n / 2) - 1))
    _, num_nodes = minimum_E_and_V(n, effective_t)
    solution_triplet = cat_state_FT_random(n, num_nodes, effective_t, [1], max_new_graphs=100)
    if solution_triplet is None:
        raise ValueError(f"cat_state_FT_random failed to find a graph for n={n}, t={t}")

    base_graph, spanning_trees, marked_edges = solution_triplet
    spanning_tree = spanning_trees[1]
    matchings = match_forest_leaves_to_marked_edges(base_graph, spanning_tree, marked_edges)
    return base_graph, spanning_tree, marked_edges, matchings


def _load_or_adapt_solution_triplet(
    n: int, t: int
) -> tuple[nx.Graph, nx.Graph, dict[tuple[int, int], int], dict[int, list[tuple[int, int]]]]:
    """
    Loads a precomputed solution triplet, or adapts one with excess marks if the exact
    size is not directly available.
    """
    result = load_solution_triplet(n, t, 1)
    if result is not None:
        return result

    # Fallback to predefined nearby configurations
    if n == 26 and t == 7:
        fallback_result = load_solution_triplet(27, 7, 1)
    else:
        fallback_n = FALLBACK_TRIPLET_SIZES.get(t)
        fallback_result = (
            load_solution_triplet(fallback_n, t, 1) if fallback_n is not None else None
        )

    if fallback_result is None:
        raise ValueError(f"No precomputed solution triplet found for n={n}, t={t}")

    base_graph, spanning_tree, marked_edges, matchings = fallback_result
    # Reduce the number of marked edges down to n
    marks = [edge for edge, val in marked_edges.items() if val == 1]
    excess_marks = len(marks) - n
    for i in range(excess_marks):
        marked_edges[marks[i]] = 0

    return base_graph, spanning_tree, marked_edges, matchings


def _permute_and_remark_graph(
    base_graph: nx.Graph, n: int, t: int
) -> tuple[nx.Graph, dict[tuple[int, int], int], dict[int, list[tuple[int, int]]]]:
    """
    Randomly permutes graph nodes and searches for a fresh valid marking and spanning tree.
    Used during retries to explore alternative DAG traversals.
    """
    nodes = list(base_graph.nodes())
    random.shuffle(nodes)
    mapping = {u: v for u, v in zip(base_graph.nodes(), nodes)}
    inv_mapping = {v: k for k, v in mapping.items()}
    shuffled_graph = nx.relabel_nodes(base_graph, mapping)

    marker = GraphMarker(shuffled_graph, max_marks=n)
    effective_t = min(t, int(np.floor(n / 2) - 1))
    shuffled_marks = marker.find_solution(effective_t)
    if sum(shuffled_marks.values()) != n:
        raise ValueError(f"GraphMarker failed to find valid marking for n={n}, t={t}")

    marked_edges = {
        ed(inv_mapping[u], inv_mapping[v]): val
        for (u, v), val in shuffled_marks.items()
    }

    forest = build_trivial_spanning_forest(base_graph, marked_edges)
    spanning_tree = build_min_diameter_spanning_tree(base_graph, forest, marked_edges, 1)
    matchings = match_forest_leaves_to_marked_edges(base_graph, spanning_tree, marked_edges)
    return spanning_tree, marked_edges, matchings


def _synthesize_expanded_graphs(
    base_graph: nx.Graph,
    spanning_tree: nx.Graph,
    marked_edges: dict[tuple[int, int], int],
    matchings: dict[int, list[tuple[int, int]]],
    attempt: int,
    cooling_rate: float = DEFAULT_COOLING_RATE,
) -> tuple[nx.Graph, nx.Graph, dict[int, int]]:
    """
    Expands base graph and tree into interaction graph and spanning forest,
    optimizing via constrained MDSF.
    """
    expanded_graph, _ = expand_graph_and_forest(
        base_graph, spanning_tree, marked_edges, matchings, expand_flags=False
    )
    spanning_forest = constrained_mdsf_generation(
        expanded_graph, 1, seed=MDSF_SEED_BASE + attempt, cooling_rate=cooling_rate
    )
    spanning_forest = spanning_forest.copy()
    roots = find_min_height_degree_k_roots(spanning_forest)
    return expanded_graph, spanning_forest, roots


def _build_and_resolve_dependency_dag(
    graph: nx.Graph,
    forest: nx.Graph,
    root: int,
    fallback_node: int | None = None,
) -> tuple[nx.DiGraph, int]:
    """
    Constructs the traversal digraph and resolves cycle closures by removing
    the cycle-forming missing link to produce a valid DAG.

    Args:
        graph: Full interaction graph.
        forest: Spanning forest/tree.
        root: Root node ID for traversal.
        fallback_node: Default node to return if no missing links were present.

    Returns:
        tuple of (dependency_dag, cycle_closure_node_or_fallback)
    """
    traversal_digraph = build_traversal_digraph(graph, forest, root)
    _, valid_edges, dependency_dag = resolve_dag_by_removing_missing_link(traversal_digraph)

    if dependency_dag is None or not nx.is_directed_acyclic_graph(dependency_dag):
        raise RuntimeError("Failed to resolve dependency graph into a directed acyclic graph (DAG)")

    if len(valid_edges) > 0:
        main_node = valid_edges[0][0]
    elif fallback_node is not None:
        main_node = fallback_node
    else:
        sink_nodes = [x for x in dependency_dag.nodes() if dependency_dag.out_degree(x) == 0]
        if not sink_nodes:
            raise ValueError("No cycle closure edge found and no sink node available")
        main_node = sink_nodes[0]

    return dependency_dag, main_node


def well_ordered_composite_cat_state_data(
    ns: Sequence[int],
    t: int,
    force_generate: bool = False,
    regenerate_graph: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, int]:
    """
    Builds a well-ordered composite cat state for an ordered list of component sizes.

    When an allowed hook error is safe within small components but would propagate
    too widely on a single large cat state, the cat state can be partitioned into
    a chain of smaller cat states connected via 1-1 outputs.

    Args:
        ns: Sequence of partition sizes, e.g. [5, 5] or [3, 2, 3].
            The total logical cat state output size is sum(ns).
        t: Fault-tolerance distance parameter.
        force_generate: If True, bypasses cache and forces fresh generation.
        regenerate_graph: If True, uses random graph synthesis instead of triplets.
        max_retries: Maximum retry attempts.

    Returns:
        tuple of (composite_graph, composite_forest, roots_dict, composite_dag, last_exit_node)
    """
    if not ns:
        raise ValueError("ns cannot be empty")

    num_chunks = len(ns)
    if num_chunks == 1:
        return well_ordered_ft_cat_state_data(
            ns[0],
            t,
            force_generate=force_generate,
            regenerate_graph=regenerate_graph,
            max_retries=max_retries,
        )

    composite_graph = nx.Graph()
    composite_forest = nx.Graph()
    composite_dag = nx.DiGraph()

    node_offset = 0
    prev_exit: int | None = None
    global_root: int | None = None

    for k, n_logical in enumerate(ns):
        # Calculate chunk size with connecting leg overhead:
        # First chunk: n_0 + 1 (1 forward connection)
        # Middle chunks: n_k + 2 (1 backward + 1 forward connection)
        # Last chunk: n_{m-1} + 1 (1 backward connection)
        if k == 0:
            chunk_size = n_logical + 1
        elif k == num_chunks - 1:
            chunk_size = n_logical + 1
        else:
            chunk_size = n_logical + 2

        if not(chunk_size <= 3 or t == 0) and (chunk_size <= 5 or t == 1):
            graph_k, forest_k, root_k = build_base_t1_graph(n_logical)
            dependency_dag_k, edge_k = _build_and_resolve_dependency_dag(
                graph_k, forest_k, root_k
            )
            roots_k = [0]
        else:
            if chunk_size <= 3 or t == 0:
                chunk_size -= chunk_size - n_logical

            graph_k, forest_k, roots_k, dependency_dag_k, edge_k = well_ordered_ft_cat_state_data(
                chunk_size,
                t,
                force_generate=force_generate,
                regenerate_graph=regenerate_graph,
                max_retries=max_retries,
            )

        if k == 0:
            root_k = roots_k[0]
            exit_k = edge_k
        else:
            # Find degree-2 root so adding inter-chunk edge maintains degree <= 3 in forest
            deg2_roots = find_min_height_degree_k_roots(forest_k, degree=2)
            root_k = next(iter(deg2_roots.values()))
            dependency_dag_k, exit_k = _build_and_resolve_dependency_dag(
                graph_k, forest_k, root_k
            )



        # Relabel nodes to guarantee disjoint continuous ranges
        node_map = {old_node: node_offset + i for i, old_node in enumerate(graph_k.nodes())}
        graph_k_rel = nx.relabel_nodes(graph_k, node_map, copy=True)
        forest_k_rel = nx.relabel_nodes(forest_k, node_map, copy=True)
        dag_k_rel = nx.relabel_nodes(dependency_dag_k, node_map, copy=True)

        composite_graph = nx.compose(composite_graph, graph_k_rel)
        composite_forest = nx.compose(composite_forest, forest_k_rel)
        composite_dag = nx.compose(composite_dag, dag_k_rel)

        draw_forest_on_graph(composite_graph, composite_forest)
        plt.show()

        mapped_root = node_map[root_k]
        mapped_exit = node_map[exit_k]

        if k == 0:
            global_root = mapped_root
        else:
            assert prev_exit is not None
            composite_graph.add_edge(prev_exit, mapped_root)
            composite_forest.add_edge(prev_exit, mapped_root)
            composite_dag.add_edge(prev_exit, mapped_root, edge_type="tree")

        prev_exit = mapped_exit
        node_offset += len(graph_k)

    if not nx.is_directed_acyclic_graph(composite_dag):
        raise RuntimeError("Composite dependency graph is not a directed acyclic graph (DAG)")
    if not nx.is_tree(composite_forest):
        raise RuntimeError("Composite spanning forest is not a valid single tree")

    assert global_root is not None
    assert prev_exit is not None
    return composite_graph, composite_forest, {0: global_root}, composite_dag, prev_exit


# Alias for composite cat state data
well_ordered_split_cat_state_data = well_ordered_composite_cat_state_data


def well_ordered_ft_cat_state_data(
    n: int | Sequence[int],
    t: int,
    force_generate: bool = False,
    regenerate_graph: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, int]:
    """
    Loads or generates well-ordered fault-tolerant cat state graph data structures.

    A well-ordered cat state consists of:
        - G: The interaction graph of the cat state with marked nodes.
        - F: A spanning forest / tree for circuit extraction.
        - roots: Mapping from component index to root node ID.
        - D: An acyclic directed dependency graph (DAG) governing CNOT scheduling.
        - edge: The source node of the resolved missing link or terminal fallback node.

    Args:
        n: Number of marked qubits / logical spider legs, or an ordered sequence
           of partition sizes (e.g. [5, 5] or [3, 2, 3]).
        t: Fault-tolerance distance parameter.
        force_generate: If True, ignores cached state files and generates freshly.
        regenerate_graph: If True, uses random graph synthesis instead of precomputed triplets.
        max_retries: Maximum retry attempts for simulated annealing and graph perturbation.

    Returns:
        tuple of (G, F, roots, D, edge)
    """
    if isinstance(n, (list, tuple, Sequence)) and not isinstance(n, (str, bytes)):
        return well_ordered_composite_cat_state_data(
            ns=n,
            t=t,
            force_generate=force_generate,
            regenerate_graph=regenerate_graph,
            max_retries=max_retries,
        )

    if not force_generate:
        cached_data = load_state_data(n, t)
        if cached_data is not None:
            return cached_data

    retries = max_retries if force_generate else 0
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            base_case = _build_base_case(n, t)
            if base_case is not None:
                graph, forest, roots, fallback_node = base_case
            else:
                fallback_node = None
                if regenerate_graph:
                    base_graph, spanning_tree, marked_edges, matchings = _generate_random_solution(n, t)
                else:
                    base_graph, spanning_tree, marked_edges, matchings = _load_or_adapt_solution_triplet(n, t)
                    if attempt > 0:
                        spanning_tree, marked_edges, matchings = _permute_and_remark_graph(base_graph, n, t)

                graph, forest, roots = _synthesize_expanded_graphs(
                    base_graph, spanning_tree, marked_edges, matchings, attempt
                )

            dependency_dag, main_node = _build_and_resolve_dependency_dag(
                graph, forest, roots[0], fallback_node=fallback_node
            )
            return graph, forest, roots, dependency_dag, main_node

        except Exception as ex:
            last_error = ex
            logger.debug(
                "Cat state generation attempt %d/%d failed for n=%d, t=%d: %s",
                attempt,
                retries,
                n,
                t,
                ex,
            )

    raise RuntimeError(
        f"Failed to generate well-ordered cat state for n={n}, t={t} after {retries + 1} attempts"
    ) from last_error


def main(draw: bool = False) -> None:
    """Demo extraction of a well-ordered composite cat state circuit."""
    random.seed(1)
    from spidercat.circuit_extraction import CatStateExtractor, StimBuilder

    ns, t = [4, 2], 3
    print(f"Generating well-ordered composite cat state for ns={ns}, t={t}...")
    graph, forest, roots, dependency_dag, edge = well_ordered_composite_cat_state_data(ns, t, regenerate_graph=True, force_generate=True)

    if draw:
        from matplotlib import pyplot as plt
        from spidercat.draw import display_digraph, draw_forest_on_graph

        draw_forest_on_graph(graph, forest)
        plt.show()
        # display_digraph(dependency_dag)
        # plt.show()

    extractor = CatStateExtractor(StimBuilder(), verbose=True)
    circuit = extractor.extract(graph, forest, roots, dependency_dag)
    print(
        f"Successfully extracted circuit with {circuit.num_qubits} qubits and {len(circuit)} instructions."
    )
    print(circuit)


if __name__ == "__main__":
    main(True)
