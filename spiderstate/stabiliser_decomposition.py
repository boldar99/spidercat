"""Fault-tolerant stabiliser-state decomposition.

This module stops at a validated, circuit-independent intermediate
representation.  In particular, its public contract contains no Stim objects
and it never instantiates a circuit builder.  Circuit extraction lives in
``spiderstate.circuit_extraction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import networkx as nx
import numpy as np

from spiderstate.optimize_parity_matrix import has_unique_ones_property
from spiderstate.spider_leg_matcher import match_edges
from spiderstate.utils import find_pivots_in_matrix
from spiderstate.well_ordered_cat_state import well_ordered_ft_cat_state_data


SpiderType = Literal["X", "Z"]
CatStateData = tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, int]


@dataclass(frozen=True)
class ComponentMetadata:
    """Traceability information for one CAT-state component."""

    tree_id: int
    spider_type: SpiderType
    matrix_row: int | None
    matrix_column: int
    nodes: tuple[int, ...]
    root: int
    main_node: int
    primary_path: tuple[int, ...]


@dataclass(frozen=True)
class InterComponentCoupling:
    """One matrix edge realized as a Z-to-X component CNOT."""

    matrix_row: int
    matrix_column: int
    control_node: int
    target_node: int


@dataclass(frozen=True)
class StabiliserStateDecomposition:
    """The stable contract between decomposition and circuit extraction.

    The NetworkX objects remain mutable Python objects, so callers should use
    :meth:`extraction_inputs` when handing them to a backend.  That method
    returns defensive copies.
    """

    parity_matrix: np.ndarray
    distance: int
    fault_budget: int
    graph: nx.Graph
    forest: nx.Graph
    roots: dict[int, int]
    dependency_graph: nx.DiGraph
    primary_paths: dict[int, tuple[int, ...]]
    components: tuple[ComponentMetadata, ...]
    couplings: tuple[InterComponentCoupling, ...]
    candidate_id: str = "default"
    schema_version: int = 1

    @property
    def num_data_qubits(self) -> int:
        return int(self.parity_matrix.shape[1])

    @property
    def dependency_depth(self) -> int:
        if not self.dependency_graph:
            return 0
        return int(nx.dag_longest_path_length(self.dependency_graph))

    def extraction_inputs(
        self,
    ) -> tuple[nx.Graph, nx.Graph, dict[int, int], nx.DiGraph, dict[int, list[int]]]:
        """Return isolated backend inputs in the legacy extractor's format."""

        return (
            self.graph.copy(),
            self.forest.copy(),
            dict(self.roots),
            self.dependency_graph.copy(),
            {tree_id: list(path) for tree_id, path in self.primary_paths.items()},
        )

    def validate(self, *, strict_target_alignment: bool = False) -> None:
        """Check the cross-module contract before circuit generation.

        ``strict_target_alignment`` additionally enforces the mark-count
        convention used by ``CatStateExtractor``.  It is optional so that the
        IR can also be consumed by extraction backends with a different data
        encoding.
        """

        matrix = np.asarray(self.parity_matrix)
        if matrix.ndim != 2:
            raise ValueError("parity_matrix must be two-dimensional")
        if not np.all((matrix == 0) | (matrix == 1)):
            raise ValueError("parity_matrix must be binary")
        if self.distance < 1:
            raise ValueError("distance must be positive")
        if self.fault_budget != self.distance // 2:
            raise ValueError("fault_budget is inconsistent with distance")

        graph_nodes = set(self.graph.nodes)
        if set(self.forest.nodes) != graph_nodes:
            raise ValueError("graph and forest must contain the same nodes")
        if set(self.dependency_graph.nodes) != graph_nodes:
            raise ValueError("graph and dependency_graph must contain the same nodes")
        if not nx.is_forest(self.forest):
            raise ValueError("forest must be acyclic")
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            raise ValueError("dependency_graph must be acyclic")
        if any(not self.graph.has_edge(u, v) for u, v in self.forest.edges):
            raise ValueError("every forest edge must also be a graph edge")

        forest_components = list(nx.connected_components(self.forest))
        if len(forest_components) != len(self.roots):
            raise ValueError("roots must contain exactly one root per forest component")
        root_nodes = set(self.roots.values())
        if not root_nodes <= graph_nodes:
            raise ValueError("every root must be a graph node")
        for component in forest_components:
            if len(component & root_nodes) != 1:
                raise ValueError("each forest component must contain exactly one root")

        if set(self.primary_paths) != set(self.roots):
            raise ValueError("primary_paths and roots must use the same tree identifiers")
        for tree_id, path in self.primary_paths.items():
            if not path or path[0] != self.roots[tree_id]:
                raise ValueError(f"primary path {tree_id} must start at its root")
            if any(not self.forest.has_edge(u, v) for u, v in zip(path, path[1:])):
                raise ValueError(f"primary path {tree_id} must follow forest edges")

        component_ids = {component.tree_id for component in self.components}
        if component_ids != set(self.roots):
            raise ValueError("component metadata and roots must use the same identifiers")
        component_columns = {
            component.matrix_column for component in self.components
        }
        if component_columns != set(range(self.num_data_qubits)):
            raise ValueError("components must cover every target matrix column exactly once")
        covered_nodes: set[int] = set()
        for component in self.components:
            component_nodes = set(component.nodes)
            if covered_nodes & component_nodes:
                raise ValueError("component node sets must be disjoint")
            covered_nodes.update(component_nodes)
            if component.root != self.roots[component.tree_id]:
                raise ValueError(f"component {component.tree_id} has an inconsistent root")
            if component.primary_path != self.primary_paths[component.tree_id]:
                raise ValueError(
                    f"component {component.tree_id} has an inconsistent primary path"
                )
            if not {component.root, component.main_node} <= component_nodes:
                raise ValueError(
                    f"component {component.tree_id} does not contain its root and main node"
                )
        if covered_nodes != graph_nodes:
            raise ValueError("component metadata must partition all graph nodes")

        coupling_edges = {
            (coupling.control_node, coupling.target_node)
            for coupling in self.couplings
        }
        graph_cnot_edges = {
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if data.get("edge_type") == "cnot"
        }
        if {frozenset(edge) for edge in coupling_edges} != {
            frozenset(edge) for edge in graph_cnot_edges
        }:
            raise ValueError("coupling metadata must cover every graph CNOT edge")
        for coupling in self.couplings:
            edge = (coupling.control_node, coupling.target_node)
            if not self.graph.has_edge(*edge):
                raise ValueError(f"coupling edge {edge} is absent from graph")
            if self.graph.edges[edge].get("edge_type") != "cnot":
                raise ValueError(f"coupling edge {edge} is not tagged as a CNOT")
            if self.forest.has_edge(*edge):
                raise ValueError(f"coupling edge {edge} must not be a forest edge")
            if self.graph.nodes[coupling.control_node].get("spider_type") != "Z":
                raise ValueError(f"coupling control {coupling.control_node} must be a Z spider")
            if self.graph.nodes[coupling.target_node].get("spider_type") != "X":
                raise ValueError(f"coupling target {coupling.target_node} must be an X spider")
            if self.parity_matrix[
                coupling.matrix_row,
                coupling.matrix_column,
            ] != 1:
                raise ValueError(
                    "coupling metadata must refer to a non-zero parity-matrix entry"
                )

        if strict_target_alignment:
            marked_nodes = sum(
                bool(data.get("is_mark", False))
                for _, data in self.graph.nodes(data=True)
            )
            if marked_nodes != self.num_data_qubits:
                raise ValueError(
                    "decomposition mark count does not match the target data-qubit count: "
                    f"{marked_nodes} != {self.num_data_qubits}"
                )


def _candidate_nodes(
    graph: nx.Graph,
    dependency_graph: nx.DiGraph,
    main_node: int,
) -> list[int]:
    candidates: list[int] = []
    for generation in nx.topological_generations(dependency_graph):
        candidates.extend(
            node
            for node in generation
            if node != main_node and graph.nodes[node].get("is_mark", False)
        )
    return candidates


def _copy_cat_state_data(data: CatStateData) -> CatStateData:
    graph, forest, roots, dependency_graph, main_node = data
    return (
        graph.copy(),
        forest.copy(),
        dict(roots),
        dependency_graph.copy(),
        int(main_node),
    )


def decompose_stabiliser_state(
    H: np.ndarray,
    d: int,
    *,
    candidate_id: str = "default",
    unique_ones_validator: Callable[[np.ndarray], bool] = has_unique_ones_property,
    pivot_finder: Callable[[np.ndarray], tuple[dict[int, int], list[int]]] = find_pivots_in_matrix,
    cat_state_factory: Callable[[int, int], CatStateData] = well_ordered_ft_cat_state_data,
    edge_matcher: Callable[
        [
            np.ndarray,
            list[int],
            list[nx.DiGraph],
            list[nx.DiGraph],
            list[list[int]],
            list[list[int]],
        ],
        list[tuple[tuple[int, int], tuple[int, int]]],
    ] = match_edges,
) -> StabiliserStateDecomposition:
    """Decompose a bipartite CSS-state matrix into an extraction-ready IR."""

    raw_matrix = np.asarray(H)
    if raw_matrix.ndim != 2:
        raise ValueError("H must be a two-dimensional parity matrix")
    if not np.all((raw_matrix == 0) | (raw_matrix == 1)):
        raise ValueError("H must be binary")
    if d < 1:
        raise ValueError("d must be positive")

    matrix = np.array(raw_matrix, dtype=np.int8, copy=True)
    if not unique_ones_validator(matrix):
        raise ValueError("H is not representing a bipartite graph state.")

    num_data_qubits = matrix.shape[1]
    fault_budget = d // 2
    pivots, rows_without_pivots = pivot_finder(matrix)
    if rows_without_pivots:
        raise ValueError(
            "H is not representing a bipartite graph state: "
            f"rows without pivots are {rows_without_pivots}"
        )

    pivot_rows_by_column = {
        int(column): int(row) for row, column in pivots.items()
    }
    non_pivots = [
        column for column in range(num_data_qubits)
        if column not in pivot_rows_by_column
    ]

    z_sizes = np.sum(matrix, axis=1)
    x_sizes = np.sum(matrix[:, non_pivots], axis=0) + 1
    z_data = [
        _copy_cat_state_data(cat_state_factory(int(size), fault_budget))
        for size in z_sizes
    ]
    x_data = [
        _copy_cat_state_data(cat_state_factory(int(size), fault_budget))
        for size in x_sizes
    ]

    z_graphs = [data[0] for data in z_data]
    z_forests = [data[1] for data in z_data]
    z_roots = [data[2] for data in z_data]
    z_digraphs = [data[3] for data in z_data]
    z_mains = [data[4] for data in z_data]
    x_graphs = [data[0] for data in x_data]
    x_forests = [data[1] for data in x_data]
    x_roots = [data[2] for data in x_data]
    x_digraphs = [data[3] for data in x_data]
    x_mains = [data[4] for data in x_data]

    for graph in z_graphs:
        nx.set_node_attributes(graph, "Z", "spider_type")
    for graph in x_graphs:
        nx.set_node_attributes(graph, "X", "spider_type")

    z_candidates = [
        _candidate_nodes(graph, digraph, main)
        for graph, digraph, main in zip(z_graphs, z_digraphs, z_mains)
    ]
    x_candidates = [
        _candidate_nodes(graph, digraph, main)
        for graph, digraph, main in zip(x_graphs, x_digraphs, x_mains)
    ]
    matched_edges = edge_matcher(
        matrix,
        non_pivots,
        z_digraphs,
        x_digraphs,
        z_candidates,
        x_candidates,
    )

    z_node_mapping: dict[tuple[int, int], int] = {}
    x_node_mapping: dict[tuple[int, int], int] = {}
    global_graph = nx.Graph()
    global_forest = nx.Graph()
    global_roots: dict[int, int] = {}
    global_dependency_graph = nx.DiGraph()
    global_primary_paths: dict[int, tuple[int, ...]] = {}
    components: list[ComponentMetadata] = []
    next_global_node = 0
    x_position_by_column = {
        column: position for position, column in enumerate(non_pivots)
    }

    for tree_id, matrix_column in enumerate(range(num_data_qubits)):
        if matrix_column in x_position_by_column:
            source_index = x_position_by_column[matrix_column]
            graph = x_graphs[source_index]
            forest = x_forests[source_index]
            roots = x_roots[source_index]
            dependency_graph = x_digraphs[source_index]
            main_node = x_mains[source_index]
            node_mapping = x_node_mapping
            spider_type: SpiderType = "X"
            matrix_row = None
        else:
            source_index = pivot_rows_by_column[matrix_column]
            graph = z_graphs[source_index]
            forest = z_forests[source_index]
            roots = z_roots[source_index]
            dependency_graph = z_digraphs[source_index]
            main_node = z_mains[source_index]
            node_mapping = z_node_mapping
            spider_type = "Z"
            matrix_row = source_index

        if len(roots) != 1:
            raise ValueError(
                "each CAT-state component must have exactly one extraction root"
            )
        local_root = next(iter(roots.values()))

        component_nodes: list[int] = []
        for node, data in graph.nodes(data=True):
            node_mapping[(source_index, node)] = next_global_node
            component_nodes.append(next_global_node)
            global_graph.add_node(next_global_node, **data)
            global_forest.add_node(next_global_node, **data)
            global_dependency_graph.add_node(next_global_node, **data)
            next_global_node += 1

        for u, v, data in graph.edges(data=True):
            global_graph.add_edge(
                node_mapping[(source_index, u)],
                node_mapping[(source_index, v)],
                **data,
            )
        for u, v, data in forest.edges(data=True):
            global_forest.add_edge(
                node_mapping[(source_index, u)],
                node_mapping[(source_index, v)],
                **data,
            )
        for u, v, data in dependency_graph.edges(data=True):
            global_dependency_graph.add_edge(
                node_mapping[(source_index, u)],
                node_mapping[(source_index, v)],
                **data,
            )

        global_root = node_mapping[(source_index, local_root)]
        global_main = node_mapping[(source_index, main_node)]
        primary_path = tuple(
            nx.shortest_path(global_forest, source=global_root, target=global_main)
        )
        global_roots[tree_id] = global_root
        global_primary_paths[tree_id] = primary_path
        components.append(
            ComponentMetadata(
                tree_id=tree_id,
                spider_type=spider_type,
                matrix_row=matrix_row,
                matrix_column=matrix_column,
                nodes=tuple(component_nodes),
                root=global_root,
                main_node=global_main,
                primary_path=primary_path,
            )
        )

    couplings: list[InterComponentCoupling] = []
    for (z_index, x_index), (z_node, x_node) in matched_edges:
        control_node = z_node_mapping[(z_index, z_node)]
        target_node = x_node_mapping[(x_index, x_node)]
        global_graph.add_edge(
            control_node,
            target_node,
            edge_type="cnot",
        )
        global_graph.nodes[control_node]["is_mark"] = False
        global_graph.nodes[target_node]["is_mark"] = False

        if global_forest.degree(control_node) == 1:
            global_graph.nodes[control_node]["is_flag"] = True
        if global_forest.degree(target_node) == 1:
            global_graph.nodes[target_node]["is_flag"] = True

        for predecessor, _ in z_digraphs[z_index].in_edges(z_node):
            global_dependency_graph.add_edge(
                z_node_mapping[(z_index, predecessor)],
                target_node,
                edge_type="cnot",
            )
        for predecessor, _ in x_digraphs[x_index].in_edges(x_node):
            global_dependency_graph.add_edge(
                x_node_mapping[(x_index, predecessor)],
                control_node,
                edge_type="cnot",
            )

        couplings.append(
            InterComponentCoupling(
                matrix_row=int(z_index),
                matrix_column=int(non_pivots[x_index]),
                control_node=control_node,
                target_node=target_node,
            )
        )

    decomposition = StabiliserStateDecomposition(
        parity_matrix=matrix,
        distance=int(d),
        fault_budget=int(fault_budget),
        graph=global_graph,
        forest=global_forest,
        roots=global_roots,
        dependency_graph=global_dependency_graph,
        primary_paths=global_primary_paths,
        components=tuple(components),
        couplings=tuple(couplings),
        candidate_id=candidate_id,
    )
    decomposition.validate()
    return decomposition
