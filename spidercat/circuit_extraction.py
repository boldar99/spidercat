from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spidercat.draw import draw_spanning_forest_solution
from spidercat.fault_analysis import GraphEdge, make_boundary_edge
from spidercat.spanning_tree import inject_root_nodes
from spidercat.utils import ed
from spidercat.weight_optimization import EdgeWeightValue, WeightCandidate


class CircuitBuilder(ABC):
    @abstractmethod
    def add_h(self, qubit): pass

    @abstractmethod
    def add_cnot(self, control, target): pass

    @abstractmethod
    def init_ancilla(self, qubit):
        """Inits ancilla and applies H for your specific extraction logic."""
        pass

    @abstractmethod
    def post_select(self, qubit):
        """Applies H and post-selects (or measures) for your logic."""
        pass

    @abstractmethod
    def add_feedback_x(self, meas_idx, target_qubit):
        """
        Adds an X gate on target_qubit controlled by the measurement at absolute index meas_idx.
        Stim uses relative indexing (rec[-k]), so we calculate offset.
        """
        pass

    @abstractmethod
    def add_detector(self, m_idx1, m_idx2):
        """
        Adds a detector that fires if measurement[m_idx1] != measurement[m_idx2].
        (i.e., parity is 1).
        """
        pass

    @abstractmethod
    def get_circuit(self): pass


class PyZXBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = zx.Circuit(0)

    def add_h(self, q): self.circ.add_gate("H", q)

    def add_cnot(self, c, t): self.circ.add_gate("CNOT", c, t)

    def init_ancilla(self, q):
        self.circ.add_gate("InitAncilla", q)
        self.add_h(q)

    def post_select(self, q):
        self.add_h(q)
        self.circ.add_gate("PostSelect", q)

    def get_circuit(self): return self.circ


class StimBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = stim.Circuit()
        self.meas_count = 0

    def add_h(self, q):
        self.circ.append("H", [q])

    def add_cnot(self, c, t):
        self.circ.append("CNOT", [c, t])

    def init_ancilla(self, q):
        # self.circ.append("R", [q])
        pass

    def post_select(self, q):
        """Performs MR and returns the absolute index of this measurement."""
        self.circ.append("M", [q])
        idx = self.meas_count
        self.meas_count += 1
        return idx

    def add_feedback_x(self, meas_idx, target_qubit):
        offset = meas_idx - self.meas_count
        self.circ.append("CX", [stim.target_rec(offset), target_qubit])

    def add_detector(self, *meas_indices):
        """
        Adds a detector on the parity of the provided measurement indices.
        - If 1 index is provided: Checks that measurement == 0.
        - If 2 indices are provided: Checks that m1 == m2.
        """
        targets = []
        for m_idx in meas_indices:
            offset = m_idx - self.meas_count
            targets.append(stim.target_rec(offset))
        self.circ.append("DETECTOR", targets)

    def get_circuit(self):
        return self.circ


def expand_graph_and_forest(
        graph: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        matchings: dict[int, list[tuple[int, int]]]
) -> tuple[nx.Graph, nx.Graph]:

    G_new = graph.copy()
    F_new = forest.copy()

    edge_to_matches = defaultdict(list)
    for matched_node, edges in matchings.items():
        for edge in edges:
            edge_to_matches[tuple(sorted(edge))].append(matched_node)

    graph_edges = {tuple(sorted(e)) for e in graph.edges()}
    forest_edges = {tuple(sorted(e)) for e in forest.edges()}
    edge_diff = graph_edges - forest_edges

    marked_edges = {tuple(sorted(e)): c for e, c in markings.items() if c > 0}
    flagged_edges = {edge: 1 for edge in edge_diff if edge not in marked_edges}

    def expand_edge(edge, count, is_mark):
        u, v = edge
        G_new.remove_edge(u, v)
        is_forest_edge = F_new.has_edge(u, v)
        if is_forest_edge:
            F_new.remove_edge(u, v)

        next_id = max(G_new.nodes()) + 1

        new_nodes = []
        for i in range(count):
            node_id = next_id
            next_id += 1
            new_nodes.append(node_id)
            # Tag the nodes so the extractor knows what pool to pull qubits from
            G_new.add_node(node_id, is_mark=is_mark, is_flag=(not is_mark), original_edge=edge)
            F_new.add_node(node_id, is_mark=is_mark, is_flag=(not is_mark), original_edge=edge)

        # Full chain: u -> n0 -> n1 -> ... -> v
        path = [u] + new_nodes + [v]
        edges_to_add = [(path[i], path[i+1]) for i in range(len(path)-1)]
        G_new.add_edges_from(edges_to_add)

        if is_forest_edge:
            F_new.add_edges_from(edges_to_add)
        else:
            # CROSS-LINK LOGIC: Drop exactly one edge to form the gap
            u_count = edge_to_matches.get(tuple(sorted(edge)), []).count(u)

            # The gap is placed immediately after u's claimed domain.
            # (If u_count > count, cap it to prevent out-of-bounds)
            gap_idx = min(u_count, count)

            for i, step_edge in enumerate(edges_to_add):
                if i != gap_idx:
                    F_new.add_edge(*step_edge)

    for edge, count in marked_edges.items(): expand_edge(edge, count, is_mark=True)
    for edge, count in flagged_edges.items(): expand_edge(edge, count, is_mark=False)

    return G_new, F_new


WeightsInput = (
    WeightCandidate
    | Mapping[GraphEdge, EdgeWeightValue]
    | Mapping[int, Mapping[GraphEdge, int]]
    | None
)


def _normalize_roots_dict(
    roots: int | np.integer | Mapping[int, int] | Iterable[int],
) -> dict[int, int]:
    """Normalizes `roots` into a `{tree_id: root_node}` dictionary."""
    if isinstance(roots, (int, np.integer)):
        return {0: int(roots)}
    if isinstance(roots, Mapping):
        return {int(k): int(v) for k, v in roots.items()}
    return {idx: int(r) for idx, r in enumerate(roots)}


def normalize_node_weights(
    G_new: nx.Graph,
    F_new: nx.Graph,
    roots: int | np.integer | Mapping[int, int] | Iterable[int],
    weights: WeightsInput,
) -> dict[int, dict[GraphEdge, int]] | None:
    """
    Normalizes a `WeightCandidate`, an `edge_weights` dict, or a `node_choices` dict
    into a per-node outgoing-leg weight mapping `node_choices[node][out_edge] -> int`.

    For each node `v`:
    - If `v` is a root node: maps its two outgoing tree edges `e0, e1` to their integer
      weights `(w0, w1)` (e.g., `(2, 1)`, `(1, 2)`, `(3, 0)`, or `(0, 3)`).
    - If `v` is a non-root node: maps its outgoing legs in the open graph
      (outgoing tree edges, incident non-tree flag edges, and `(v, "boundary")` if
      `is_mark=True`) to their local weights `1` or `2` at `v`.
    """
    if weights is None:
        return None

    roots_dict = _normalize_roots_dict(roots)
    root_set = set(roots_dict.values())

    parent_of: dict[int, int | None] = {}
    for r in roots_dict.values():
        parent_of[r] = None
        for u, v in nx.bfs_edges(F_new, r):
            parent_of[v] = u

    raw_node_choices: Mapping[int, Mapping[GraphEdge, int]] | None = None
    raw_edge_weights: Mapping[GraphEdge, EdgeWeightValue] | None = None

    if isinstance(weights, WeightCandidate):
        raw_node_choices = weights.node_choices
        raw_edge_weights = weights.edge_weights
    elif isinstance(weights, Mapping):
        first_key = next(iter(weights.keys()), None)
        if isinstance(first_key, (int, np.integer)):
            raw_node_choices = weights  # type: ignore[assignment]
        else:
            raw_edge_weights = weights  # type: ignore[assignment]
    else:
        raise TypeError(f"Unsupported weights type: {type(weights)!r}")

    def _lookup_edge_local_weight(node: int, edge: GraphEdge) -> int | None:
        if raw_node_choices is not None and node in raw_node_choices:
            nc = raw_node_choices[node]
            if edge in nc:
                return int(nc[edge])
            if edge[1] != "boundary":
                rev = (edge[1], edge[0])
                if rev in nc:
                    return int(nc[rev])

        if raw_edge_weights is not None:
            val = raw_edge_weights.get(edge)
            if val is None and edge[1] != "boundary":
                val = raw_edge_weights.get((edge[1], edge[0]))
            if val is not None:
                if isinstance(val, (tuple, list, np.ndarray)):
                    if len(val) != 2:
                        raise ValueError(
                            f"Expected 2-tuple weight for flag edge {edge}, got {val!r}"
                        )
                    u, v = ed(int(edge[0]), int(edge[1]))
                    return int(val[0]) if node == u else int(val[1])
                return int(val)
        return None

    normalized: dict[int, dict[GraphEdge, int]] = {}
    for node in G_new.nodes():
        node_int = int(node)
        is_mark = bool(G_new.nodes[node].get("is_mark", False))
        parent = parent_of.get(node_int)

        if node_int in root_set:
            out_edges: list[GraphEdge] = [
                ed(node_int, int(nbr)) for nbr in sorted(F_new.neighbors(node_int))
            ]
        else:
            out_edges = [
                ed(node_int, int(nbr))
                for nbr in sorted(G_new.neighbors(node_int))
                if int(nbr) != parent
            ]
            if is_mark:
                out_edges.append(make_boundary_edge(node_int))

        leg_weights: dict[GraphEdge, int] = {}
        for e in out_edges:
            w = _lookup_edge_local_weight(node_int, e)
            if w is not None:
                leg_weights[e] = w

        # If one of two legs on a non-root node (such as an implicit boundary leg) was omitted,
        # infer it from the other leg since non-root spider legs are {1, 2}.
        if node_int not in root_set and len(out_edges) == 2 and len(leg_weights) == 1:
            e0, e1 = out_edges
            if e0 in leg_weights:
                leg_weights[e1] = 1 if leg_weights[e0] == 2 else 2
            else:
                leg_weights[e0] = 1 if leg_weights[e1] == 2 else 2

        if len(leg_weights) != len(out_edges):
            missing = [e for e in out_edges if e not in leg_weights]
            raise KeyError(
                f"Missing outgoing weight(s) {missing} at node {node_int}."
            )

        normalized[node_int] = leg_weights

    return normalized


# --- 2. The Main Extractor Class ---
class CatStateExtractor:
    def __init__(self, builder: CircuitBuilder, verbose: bool = False):
        self.builder = builder
        self.verbose = verbose

        self.node_to_qubit: dict[int, int] = {}
        self.edge_to_flag: dict[tuple[int, int], int] = {}
        self.tree_to_qubits: dict[int, set[int]] = defaultdict(set)
        self.tree_of_node: dict[int, int] = {}
        self.link_measurements: dict[tuple[int, int], list[int]] = {}
        self.depths: dict[int, float] = {}
        self.boundary_node_to_qubit: dict[int, int] = {}
        self.node_order: list[set[int]] | None = None
        self.node_weights: dict[int, dict[GraphEdge, int]] | None = None
        self.queue: deque[int] = deque()

    def _get_new_data_qubit(self) -> int:
        q = self.next_data_idx
        self.next_data_idx += 1
        return q

    def _get_new_flag_qubit(self) -> int:
        q = self.next_flag_idx
        self.next_flag_idx += 1
        return q

    def _compute_depth(self, node: int, parent: int | None, F_new: nx.Graph) -> float:
        children = [neighbor for neighbor in F_new.neighbors(node) if neighbor != parent]

        # Base case: If the node has no children, it is a leaf. Distance is 0.
        if not children:
            self.depths[node] = 0
            return 0

        # Recursive step: 1 + the minimum leaf-distance among all children
        min_d = float('inf')
        for child in children:
            child_depth = self._compute_depth(child, node, F_new)
            min_d = min(min_d, 1 + child_depth)

        self.depths[node] = min_d
        return min_d

    def _precompute_wire_fates(
        self,
        G_new: nx.Graph,
        F_new: nx.Graph,
        roots_dict: dict[int, int],
    ) -> dict[int, bool]:
        """
        Determines for each node `v` whether the qubit wire arriving at (or initialized at) `v`
        and following the chain of weight-1 (continuing) legs terminates at a non-tree flag edge
        (`True` -> allocate from `_get_new_flag_qubit`) or at a boundary edge `(m, "boundary")`
        (`False` -> allocate from `_get_new_data_qubit`).
        """
        assert self.node_weights is not None
        root_set = set(roots_dict.values())
        memo: dict[int, bool] = {}

        def _ends_at_flag(node: int) -> bool:
            if node in memo:
                return memo[node]

            choices = self.node_weights[node]
            if not choices:
                memo[node] = False
                return False

            # The continuing wire follows the outgoing leg with the minimum weight
            # (weight 1 on green nodes, or weight 1 / 0 at the root).
            w1_edge = min(choices.keys(), key=lambda e: (choices[e], str(e)))
            if w1_edge[1] == "boundary":
                memo[node] = False
                return False

            u, v = int(w1_edge[0]), int(w1_edge[1])
            nbr = v if u == node else u
            if F_new.has_edge(node, nbr) and node in root_set or (
                F_new.has_edge(node, nbr)
            ):
                memo[node] = _ends_at_flag(nbr)
            else:
                # Non-tree edge in G_new \ F_new: wire continues onto the flag edge and is measured
                memo[node] = True
            return memo[node]

        for n in G_new.nodes():
            _ends_at_flag(int(n))
        return memo

    def _allocate_wire_for_node(self, node: int, G_new: nx.Graph) -> int:
        """Allocates a data or flag qubit index depending on where `node`'s weight-1 path ends."""
        if self.node_weights is not None and hasattr(self, "_wire_becomes_flag"):
            if self._wire_becomes_flag.get(int(node), False):
                return self._get_new_flag_qubit()
            return self._get_new_data_qubit()
        is_flag = G_new.nodes[node].get("is_flag", False)
        return self._get_new_flag_qubit() if is_flag else self._get_new_data_qubit()

    def pop_next_from_queue(self) -> int:
        """
        Pops the next ready node from `self.queue` respecting the topological generations
        of `dependency_graph` when provided.
        """
        if not self.node_order:
            return self.queue.popleft()

        while self.node_order and not self.node_order[0]:
            self.node_order.pop(0)

        for q_node in list(self.queue):
            if q_node in self.node_order[0]:
                self.queue.remove(q_node)
                self.node_order[0].remove(q_node)
                if not self.node_order[0]:
                    self.node_order.pop(0)
                return q_node

        # If no node in the earliest non-empty generation is currently in `self.queue`
        # (e.g., if `dependency_graph` has multiple roots or layers), find the earliest
        # generation that intersects `self.queue`.
        for gen in self.node_order:
            for q_node in list(self.queue):
                if q_node in gen:
                    self.queue.remove(q_node)
                    gen.remove(q_node)
                    return q_node

        return self.queue.popleft()

    def extract(
        self,
        G_new: nx.Graph,
        F_new: nx.Graph,
        roots: int | np.integer | Mapping[int, int] | Iterable[int],
        dependency_graph: nx.DiGraph | None = None,
        weights: WeightsInput = None,
    ):
        if self.verbose:
            print("=== Starting Elegant Extraction (BFS) ===")

        self.node_to_qubit.clear()
        self.edge_to_flag.clear()
        self.tree_to_qubits.clear()
        self.tree_of_node.clear()
        self.link_measurements.clear()
        self.depths.clear()
        self.boundary_node_to_qubit.clear()
        self.queue.clear()

        roots_dict = _normalize_roots_dict(roots)

        self.next_data_idx = 0
        self.next_flag_idx = len(
            [v for v in G_new.nodes if G_new.nodes[v].get("is_mark", False)]
        )

        self.node_weights = normalize_node_weights(G_new, F_new, roots_dict, weights)

        if self.node_weights is not None:
            if dependency_graph is None:
                dependency_graph = build_traversal_digraph(
                    G_new, F_new, roots_dict, weights=self.node_weights
                )
            else:
                dependency_graph = _inject_weight_dependencies(
                    dependency_graph, G_new, F_new, self.node_weights
                )

        if dependency_graph is not None:
            if not nx.is_directed_acyclic_graph(dependency_graph):
                raise ValueError("dependency_graph must be a directed acyclic graph (DAG).")
            self.node_order = [
                set(gen) for gen in nx.topological_generations(dependency_graph)
            ]
        else:
            self.node_order = None

        for tree_id, root in roots_dict.items():
            self._compute_depth(root, None, F_new)
            for u in nx.dfs_preorder_nodes(F_new, root):
                self.tree_of_node[int(u)] = tree_id

        if self.node_weights is not None:
            self._wire_becomes_flag = self._precompute_wire_fates(
                G_new, F_new, roots_dict
            )
            self._grow_trees_weighted(roots_dict, G_new, F_new)
        elif dependency_graph is not None:
            self._grow_trees_unweighted_dag(roots_dict, G_new, F_new)
        else:
            # PASS 1: Grow Trees Level-by-Level (legacy per-tree BFS)
            for tree_id, root in roots_dict.items():
                self._grow_tree_bfs(root, tree_id, G_new, F_new)

        self._generate_detectors()
        self._generate_feedback()

        return self.builder.get_circuit()

    def _init_root_node(self, root_node: int, tree_id: int, G_new: nx.Graph) -> int:
        root_qubit = self._allocate_wire_for_node(root_node, G_new)
        self.builder.init_ancilla(root_qubit)
        self.builder.add_h(root_qubit)

        self.node_to_qubit[root_node] = root_qubit
        self.tree_to_qubits[tree_id].add(root_qubit)
        self.tree_of_node[root_node] = tree_id

        if self.verbose:
            print(f"Init Root {root_node} (Tree {tree_id}) -> Q{root_qubit}")
        return root_qubit

    def _handle_flag_edge_weight_2(
        self,
        node: int,
        flag_neighbor: int,
        current_qubit: int,
    ) -> None:
        """
        Executes a weight-2 operation on a non-tree flag edge `(node, flag_neighbor)`:
        - If the flag edge has not been initialized yet (first visited endpoint of a `(2, 2)`
          flag edge), initializes a new flag ancilla and applies `CNOT(current_qubit, flag_qubit)`.
        - If the flag edge is already present in `self.edge_to_flag` (either initialized as weight 2
          by the other endpoint or continued as weight 1 by the other endpoint), applies
          `CNOT(current_qubit, flag_qubit)` and measures (`post_select`) `flag_qubit`.
        """
        edge = ed(node, flag_neighbor)
        if edge in self.edge_to_flag:
            flag_qubit = self.edge_to_flag[edge]
            self.builder.add_cnot(current_qubit, flag_qubit)
            m_idx = self.builder.post_select(flag_qubit)
            t_u = self.tree_of_node[node]
            t_v = self.tree_of_node[flag_neighbor]
            self._record_meas(t_u, t_v, m_idx)
            if self.verbose:
                print(
                    f"  Flag ({node}, {flag_neighbor}) finalised (w=2): "
                    f"CNOT Q{current_qubit} -> Q{flag_qubit} + M"
                )
        else:
            flag_qubit = self._get_new_flag_qubit()
            self.builder.init_ancilla(flag_qubit)
            self.builder.add_cnot(current_qubit, flag_qubit)
            self.edge_to_flag[edge] = flag_qubit
            if self.verbose:
                print(
                    f"  New flag initialised ({node}, {flag_neighbor}) (w=2): "
                    f"CNOT Q{current_qubit} -> Q{flag_qubit}"
                )

    def _handle_flag_edge_weight_1(
        self,
        node: int,
        flag_neighbor: int,
        current_qubit: int,
    ) -> None:
        """
        Executes a weight-1 operation on a non-tree flag edge `(node, flag_neighbor)`:
        The incoming wire `current_qubit` continues directly onto the flag edge without
        initializing a new qubit or applying a CNOT at `node`. The other endpoint
        `flag_neighbor` (which has weight 2 and is ordered after `node` in `dependency_graph`)
        will apply its CNOT and measure `current_qubit`.
        """
        edge = ed(node, flag_neighbor)
        if edge in self.edge_to_flag:
            raise RuntimeError(
                f"Flag edge {edge} already in edge_to_flag when processing weight-1 endpoint "
                f"{node}. Ensure dependency_graph orders weight-1 endpoint before weight-2 endpoint."
            )
        self.edge_to_flag[edge] = current_qubit
        if self.verbose:
            print(
                f"  Flag ({node}, {flag_neighbor}) takes role of flag (w=1): "
                f"Q{current_qubit} continues onto flag edge"
            )

    def _grow_trees_weighted(
        self,
        roots_dict: dict[int, int],
        G_new: nx.Graph,
        F_new: nx.Graph,
    ) -> None:
        """
        Extracts the circuit by realizing the explicit outgoing leg weights in `self.node_weights`
        and respecting the topological node ordering in `self.node_order`.
        """
        assert self.node_weights is not None
        root_set = set(roots_dict.values())

        for tree_id, root_node in roots_dict.items():
            self._init_root_node(root_node, tree_id, G_new)
            self.queue.append(root_node)

        while self.queue:
            node = self.pop_next_from_queue()
            current_qubit = self.node_to_qubit[node]
            tree_id = self.tree_of_node[node]
            choices = self.node_weights[node]

            if node in root_set:
                # Root node has 2 outgoing tree edges with weights in {(2, 1), (1, 2), (3, 0), (0, 3)}.
                # Sort outgoing edges in descending order of weight so the larger-weight leg
                # (weight 2 or 3) spawns a CNOT first and the smaller-weight leg (weight 1 or 0)
                # inherits `current_qubit`.
                sorted_legs = sorted(
                    choices.items(), key=lambda kv: (-kv[1], str(kv[0]))
                )
                w2_edge, _ = sorted_legs[0]
                w1_edge, _ = sorted_legs[1]

                sec_child = int(w2_edge[1] if int(w2_edge[0]) == node else w2_edge[0])
                pri_child = int(w1_edge[1] if int(w1_edge[0]) == node else w1_edge[0])

                new_q = self._allocate_wire_for_node(sec_child, G_new)
                self.builder.init_ancilla(new_q)
                self.builder.add_cnot(current_qubit, new_q)
                self.node_to_qubit[sec_child] = new_q
                self.tree_to_qubits[tree_id].add(new_q)
                self.tree_of_node[sec_child] = tree_id
                if self.verbose:
                    print(
                        f"  Root {node} -> Secondary {sec_child} (w=2): "
                        f"Spawned CNOT Q{current_qubit} -> Q{new_q}"
                    )
                self.queue.append(sec_child)

                self.node_to_qubit[pri_child] = current_qubit
                self.tree_to_qubits[tree_id].add(current_qubit)
                self.tree_of_node[pri_child] = tree_id
                if self.verbose:
                    print(
                        f"  Root {node} -> Primary {pri_child} (w=1): "
                        f"Inherits Q{current_qubit}"
                    )
                self.queue.append(pri_child)
                continue

            # Non-root green node: process weight-2 leg(s) BEFORE weight-1 leg
            # so `current_qubit` acts as control before continuing onto the weight-1 leg.
            sorted_legs = sorted(
                choices.items(), key=lambda kv: (-kv[1], str(kv[0]))
            )
            for leg_edge, weight in sorted_legs:
                if weight >= 2:
                    if leg_edge[1] == "boundary":
                        # Boundary leg with weight 2 (when fix_boundary_weight_one=False):
                        # spawn a new data qubit that stays at the boundary for `node`.
                        b_q = self._get_new_data_qubit()
                        self.builder.init_ancilla(b_q)
                        self.builder.add_cnot(current_qubit, b_q)
                        self.tree_to_qubits[tree_id].add(b_q)
                        self.boundary_node_to_qubit[node] = b_q
                        if self.verbose:
                            print(
                                f"  Mark on {node} boundary (w=2): "
                                f"Spawned CNOT Q{current_qubit} -> Q{b_q}"
                            )
                    else:
                        u, v = int(leg_edge[0]), int(leg_edge[1])
                        nbr = v if u == node else u
                        if F_new.has_edge(node, nbr):
                            # Tree child with weight 2: spawn a new qubit and CNOT
                            new_q = self._allocate_wire_for_node(nbr, G_new)
                            self.builder.init_ancilla(new_q)
                            self.builder.add_cnot(current_qubit, new_q)
                            self.node_to_qubit[nbr] = new_q
                            self.tree_to_qubits[tree_id].add(new_q)
                            self.tree_of_node[nbr] = tree_id
                            if self.verbose:
                                print(
                                    f"  Node {node} -> Tree child {nbr} (w=2): "
                                    f"Spawned CNOT Q{current_qubit} -> Q{new_q}"
                                )
                            self.queue.append(nbr)
                        else:
                            # Non-tree flag edge with weight 2
                            self._handle_flag_edge_weight_2(node, nbr, current_qubit)
                else:
                    # Weight-1 leg: `current_qubit` continues along this leg
                    if leg_edge[1] == "boundary":
                        self.boundary_node_to_qubit[node] = current_qubit
                        if self.verbose:
                            print(
                                f"  Mark on {node} boundary (w=1): "
                                f"Q{current_qubit} continues to boundary"
                            )
                    else:
                        u, v = int(leg_edge[0]), int(leg_edge[1])
                        nbr = v if u == node else u
                        if F_new.has_edge(node, nbr):
                            self.node_to_qubit[nbr] = current_qubit
                            self.tree_to_qubits[tree_id].add(current_qubit)
                            self.tree_of_node[nbr] = tree_id
                            if self.verbose:
                                print(
                                    f"  Node {node} -> Primary {nbr} (w=1): "
                                    f"Inherits Q{current_qubit}"
                                )
                            self.queue.append(nbr)
                        else:
                            self._handle_flag_edge_weight_1(node, nbr, current_qubit)

    def _grow_trees_unweighted_dag(
        self,
        roots_dict: dict[int, int],
        G_new: nx.Graph,
        F_new: nx.Graph,
    ) -> None:
        for tree_id, root_node in roots_dict.items():
            self._init_root_node(root_node, tree_id, G_new)
            self.queue.append(root_node)

        while self.queue:
            node = self.pop_next_from_queue()
            self._process_unweighted_node(node, self.tree_of_node[node], G_new, F_new, self.queue)

    def _process_unweighted_node(self, node, tree_id, G_new, F_new, queue):
        current_qubit = self.node_to_qubit[node]
        self.tree_of_node[node] = tree_id

        children = [n for n in F_new.neighbors(node) if n not in self.node_to_qubit]
        is_mark = G_new.nodes[node].get("is_mark", False)
        is_flag = G_new.nodes[node].get("is_flag", False)

        flag_children = [n for n in G_new.neighbors(node) if n not in F_new.neighbors(node)]
        for child in flag_children:
            edge = tuple(sorted((node, child)))
            if edge in self.edge_to_flag:
                flag_qubit = self.edge_to_flag[edge]
                self.builder.add_cnot(current_qubit, flag_qubit)
                m_idx = self.builder.post_select(flag_qubit)
                t_u, t_v = self.tree_of_node[node], self.tree_of_node[child]
                self._record_meas(t_u, t_v, m_idx)
                if self.verbose:
                    print(f"  Flag ({node}, {child}) finalised: CNOT Q{current_qubit} -> Q{flag_qubit}")
            else:
                flag_qubit = self._get_new_flag_qubit()
                self.builder.add_cnot(current_qubit, flag_qubit)
                self.edge_to_flag[edge] = flag_qubit
                if self.verbose:
                    print(f"  New flag initialised ({node}, {child}): CNOT Q{current_qubit} -> Q{flag_qubit}")

        if not children:
            if is_mark:
                self.boundary_node_to_qubit[node] = current_qubit
            if self.verbose and is_mark:
                print(f"  Node {node} serves as a sink point for Q{current_qubit}")
            return

        if is_mark:
            self.boundary_node_to_qubit[node] = current_qubit
            new_q = self._get_new_flag_qubit() if is_flag else self._get_new_data_qubit()
            self.builder.init_ancilla(new_q)
            self.builder.add_cnot(current_qubit, new_q)
            self.tree_to_qubits[tree_id].add(new_q)
            if self.verbose:
                print(f"  Mark on {node}: Spawned CNOT Q{current_qubit} -> Q{new_q}")
            self.node_to_qubit[node] = new_q
            current_qubit = new_q

        # Sort children by depth to identify the primary branch
        children.sort(key=lambda c: self.depths.get(c, 0), reverse=True)
        primary = children[-1]
        secondaries = children[:-1]

        # 3. SECONDARY CHILDREN (Spawn new qubits)
        for child in secondaries:
            new_q = self._get_new_data_qubit()
            self.builder.init_ancilla(new_q)
            self.builder.add_cnot(current_qubit, new_q)

            self.node_to_qubit[child] = new_q
            self.tree_to_qubits[tree_id].add(new_q)
            self.tree_of_node[child] = tree_id

            if self.verbose:
                print(f"  Node {node} -> Branch {child}: Spawned CNOT Q{current_qubit} -> Q{new_q}")

            queue.append(child)

        # 4. PRIMARY CHILD (Inherit current qubit)
        self.node_to_qubit[primary] = current_qubit
        self.tree_to_qubits[tree_id].add(current_qubit)
        self.tree_of_node[primary] = tree_id

        if self.verbose:
            print(f"  Node {node} -> Primary {primary} (Inherits Q{current_qubit})")

        queue.append(primary)

    def _grow_tree_bfs(self, root_node, tree_id, G_new, F_new):
        # 1. INITIALIZE ROOT
        self._init_root_node(root_node, tree_id, G_new)
        queue = deque([root_node])
        while queue:
            node = queue.popleft()
            self._process_unweighted_node(node, tree_id, G_new, F_new, queue)

    def _record_meas(self, t1, t2, m_idx):
        if t1 == t2: self.builder.add_detector(m_idx)
        else:
            k = tuple(sorted((t1, t2)))
            self.link_measurements.setdefault(k, []).append(m_idx)

    def _generate_detectors(self):
        if self.verbose: print("Generating Detectors...")
        for indices in self.link_measurements.values():
            for i in range(len(indices)-1): self.builder.add_detector(indices[i], indices[i+1])
        meta = nx.Graph()
        for (t1,t2), idxs in self.link_measurements.items():
            meta.add_edge(t1, t2, m=idxs[0])
        for cyc in nx.cycle_basis(meta):
            det = [meta[u][v]['m'] for u,v in zip(cyc, cyc[1:]+cyc[:1])]
            self.builder.add_detector(*det)
        self.meta_graph = meta

    def _generate_feedback(self):
        if not self.link_measurements: return
        root = min(list(self.meta_graph.nodes()))
        preds = dict(nx.bfs_predecessors(self.meta_graph, root))
        for t in self.meta_graph.nodes():
            if t == root or t not in preds: continue
            path_ms = []
            cur = t
            while cur != root:
                path_ms.append(self.meta_graph[preds[cur]][cur]['m'])
                cur = preds[cur]
            for m in path_ms:
                for q in self.tree_to_qubits[t]: self.builder.add_feedback_x(m, q)


def extract_from_expanded_graph(
    G_exp: nx.Graph,
    F_exp: nx.Graph,
    roots: int | np.integer | Mapping[int, int] | Iterable[int] | None = None,
    weights: WeightsInput = None,
    dependency_graph: nx.DiGraph | None = None,
    builder: CircuitBuilder | None = None,
    verbose: bool = False,
):
    """
    Extracts a quantum circuit directly from an expanded graph `G_exp` and spanning
    forest/tree `F_exp`, optionally realizing `weights` and `dependency_graph`.
    """
    if roots is None:
        roots = inject_root_nodes(G_exp, F_exp)
    extractor = CatStateExtractor(builder if builder is not None else StimBuilder(), verbose)
    return extractor.extract(
        G_exp, F_exp, roots, dependency_graph=dependency_graph, weights=weights
    )


def extract_circuit_rooted(
    G,
    forest,
    roots,
    markings,
    matches,
    weights: WeightsInput = None,
    dependency_graph: nx.DiGraph | None = None,
    verbose: bool = False,
) -> stim.Circuit:
    extractor = CatStateExtractor(StimBuilder(), verbose)
    G_exp, F_exp = expand_graph_and_forest(G, forest, markings, matches)
    if roots is None:
        roots = inject_root_nodes(G_exp, F_exp)
    return extractor.extract(
        G_exp, F_exp, roots, dependency_graph=dependency_graph, weights=weights
    )


def implement_CNOT_circuit(cnots, num_qubits, p_2, p_mem):
    circ = stim.Circuit()
    all_qubits = range(num_qubits + 1)
    for c, n in cnots:
        circ.append("CNOT", [c, n])
        if p_2 > 0 and not c.is_measurement_record_target:
            circ.append("DEPOLARIZE2", [c, n], p_2)
        if p_mem > 0:
            circ.append("DEPOLARIZE1", all_qubits, p_mem)
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
            noisy_circ.append(gate_name, targets)

    return noisy_circ


def unflagged_cat(n):
    circ = stim.Circuit()
    circ.append("H", 0)
    for i in range(1, n):
        circ.append("CNOT", [0, i])
    return circ


def one_flagged_cat(n):
    circ = stim.Circuit()
    circ.append("H", 0)
    circ.append("CNOT", [0, n])
    for i in range(1, n - 1, 2):
        circ.append("CNOT", [0, i])
        circ.append("CNOT", [n, i + 1])
    if n % 2 == 0:
        circ.append("CNOT", [0, n - 1])
    circ.append("CNOT", [0, n])
    circ.append("M", n)
    circ.append("DETECTOR", stim.target_rec(-1))
    return circ


def cat_state_6():
    return stim.Circuit("""
        H 2
        CNOT 2 3 2 1 2 4 2 0 2 5 2 6 2 1 2 7 2 0
        M 0 1 
    """)


def _inject_weight_dependencies(
    di_graph: nx.DiGraph,
    G: nx.Graph,
    F: nx.Graph,
    node_choices: Mapping[int, Mapping[GraphEdge, int]],
) -> nx.DiGraph:
    """
    Returns a copy of `di_graph` with directed edges `u_w1 -> v_w2`
    (`edge_type='missing_link'`) added for every non-tree flag edge in `G \\ F`
    whose local weights are `(1, 2)` or `(2, 1)`.
    """
    augmented = di_graph.copy()
    for u, v in G.edges():
        if F.has_edge(u, v):
            continue
        a, b = ed(int(u), int(v))
        wa = node_choices.get(a, {}).get((a, b))
        wb = node_choices.get(b, {}).get((a, b))
        if wa == 1 and wb == 2 and not augmented.has_edge(a, b):
            augmented.add_edge(a, b, edge_type="missing_link")
        elif wb == 1 and wa == 2 and not augmented.has_edge(b, a):
            augmented.add_edge(b, a, edge_type="missing_link")
    return augmented


def build_traversal_digraph(
    G: nx.Graph,
    F: nx.Graph,
    root: int | np.integer | Mapping[int, int] | Iterable[int],
    weights: WeightsInput = None,
) -> nx.DiGraph:
    """
    Builds a directed graph representing the tree hierarchy from `root`,
    and when `weights` is provided, injects directed dependency edges
    `u_w1 -> v_w2` (with `edge_type='missing_link'`) for every non-tree flag edge
    with weights `(1, 2)` or `(2, 1)` so the weight-1 endpoint is processed
    before the weight-2 endpoint.
    """
    di_graph = nx.DiGraph()
    for node, data in G.nodes(data=True):
        di_graph.add_node(node, **data)

    roots_dict = _normalize_roots_dict(root)
    root_list = list(roots_dict.values())
    queue = deque(root_list)
    visited = set(root_list)

    while queue:
        current = queue.popleft()

        # 1. Standard Tree Traversal (Parent -> Child)
        for neighbor in F.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # Label the edge type for visualization
                di_graph.add_edge(current, neighbor, edge_type='tree')

    # 2. Inject artificial dependencies (w=1 -> w=2) for flag edges in G \ F
    if weights is not None:
        node_choices = normalize_node_weights(G, F, roots_dict, weights)
        assert node_choices is not None
        di_graph = _inject_weight_dependencies(di_graph, G, F, node_choices)

    return di_graph


build_weighted_dependency_digraph = build_traversal_digraph


if __name__ == "__main__":
    from spidercat.utils import load_solution_triplet
    from spidercat.spanning_tree import find_min_height_roots, match_forest_leaves_to_marked_edges

    grf, forest, M, matchings = load_solution_triplet(12, 2, 1)
    roots = find_min_height_roots(forest)
    draw_spanning_forest_solution(grf, forest, M, matchings, roots)
    extract_circuit_rooted(grf, forest, roots, M, matchings, verbose=False)
