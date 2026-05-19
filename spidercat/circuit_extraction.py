from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spidercat.draw import draw_spanning_forest_solution, draw_forest_on_graph
from spidercat.utils import ed, flatten


class CircuitBuilder(ABC):
    @abstractmethod
    def add_h(self, qubit): pass

    @abstractmethod
    def add_cnot(self, control, target): pass

    @abstractmethod
    def init_ancilla(self, qubit, basis):
        """Inits ancilla and applies H for your specific extraction logic."""
        pass

    @abstractmethod
    def post_select(self, qubit, basis):
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

    @abstractmethod
    def permute_qubits(self, qubit_to_qubit_mapping: dict[int, int]): pass


class PyZXBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = zx.Circuit(0)

    def add_h(self, q): self.circ.add_gate("H", q)

    def add_cnot(self, c, t): self.circ.add_gate("CNOT", c, t)

    def init_ancilla(self, q, basis):
        if basis == "X":
            self.circ.add_gate("H", q)
        self.circ.add_gate("InitAncilla", q)
        self.add_h(q)

    def post_select(self, q, basis):
        if basis == "Z":
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

    def init_ancilla(self, q, basis):
        if basis == "X":
            self.circ.append("RX", [q])
        else:
            self.circ.append("R", [q])

    def post_select(self, q, basis):
        """Performs MR and returns the absolute index of this measurement."""
        if basis == "X":
            self.circ.append("MX", [q])
        else:
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

    def permute_qubits(self, qubit_to_qubit_mapping: dict[int, int]) -> stim.Circuit:
        new_circ = stim.Circuit()
        for op in self.circ:
            new_targets = []
            for t in op.targets_copy():
                if t.is_qubit_target:
                    new_targets.append(qubit_to_qubit_mapping.get(t.value, t.value))
                else:
                    new_targets.append(t)
            new_circ.append(op.name, new_targets, op.gate_args_copy())
        self.circ = new_circ


def expand_graph_and_forest(
        graph: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        matchings: dict[int, list[tuple[int, int]]],
        expand_flags: bool = True
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
    if expand_flags:
        flagged_edges = {edge: 1 for edge in edge_diff if edge not in marked_edges}
    else:
        flagged_edges = {}


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


def resolve_dag_by_removing_missing_link(di_graph: nx.DiGraph) -> tuple[bool, list[tuple], nx.DiGraph | None]:
    """
    Tests if a directed graph can be made acyclic by removing exactly one 'missing_link'.

    Returns:
        is_possible (bool): True if at least one solution exists.
        valid_edges (list): A list of all specific (u, v) missing_link edges that work.
        resolved_dag (nx.DiGraph): A copy of the graph with the FIRST valid edge removed.
    """
    # Base Case: Is it already a DAG?
    if nx.is_directed_acyclic_graph(di_graph):
        return True, [], di_graph.copy()

    valid_edges_to_remove = []
    resolved_dag = None

    # Filter for ONLY the cycle-closure edges
    missing_links = [(u, v, data) for u, v, data in di_graph.edges(data=True)
                     if data.get('edge_type') == 'missing_link']

    for u, v, data in missing_links:
        # Temporarily drop the syndrome extraction link
        di_graph.remove_edge(u, v)

        if nx.is_directed_acyclic_graph(di_graph):
            valid_edges_to_remove.append((u, v))

            # Capture the state of the graph on the very first success
            if resolved_dag is None:
                resolved_dag = di_graph.copy()

        # Put the edge back to test the next candidate cleanly
        di_graph.add_edge(u, v, **data)

    return len(valid_edges_to_remove) > 0, valid_edges_to_remove, resolved_dag


def build_traversal_digraph(G: nx.Graph, F: nx.Graph, root) -> nx.DiGraph:
    """
    Builds a directed graph representing the tree hierarchy,
    adding directed edges (l -> t) for missing cycle-closure edges.
    """
    di_graph = nx.DiGraph()
    for node, data in G.nodes(data=True):
        di_graph.add_node(node, **data)

    queue = [root]
    visited = {root}

    while queue:
        current = queue.pop(0)

        # 1. Standard Tree Traversal (Parent -> Child)
        for neighbor in F.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # Label the edge type for visualization
                di_graph.add_edge(current, neighbor, edge_type='tree')

        # 2. Cycle Closure Jumps (Leaf -> Target)
        # A leaf in the forest has degree 1. We ignore the root if it happens to be degree 1.
        if F.degree(current) == 1 and current != root:
            # Find all neighbors in the base graph G that are NOT connected in F
            missing_neighbors = [n for n in G.neighbors(current) if not F.has_edge(current, n)]

            # This safely handles cases where missing_neighbors is 0 (marked edges) or >1
            for t in missing_neighbors:
                di_graph.add_edge(current, t, edge_type='missing_link')

    return di_graph


# --- 2. The Main Extractor Class ---
class CatStateExtractor:
    def __init__(self, builder: CircuitBuilder, verbose=False):
        self.builder = builder
        self.verbose = verbose

        self.node_to_qubit = {}
        # Maps an edge to the flag qubit that it corresponds to
        self.edge_to_flag_qubit: dict[tuple[int, int], int] = {}
        self.tree_to_qubits = defaultdict(set)
        self.node_to_tree = {}
        self.link_measurements = {}
        self.flag_measurements = []
        self.depths = {}
        self.branch_mark_values = {}
        self.flag_distances = {}
        self.data_flags = []

        self.queue = []
        self.processed = set()
        self.processed_cnots = set()
        self.primary_paths = {}

    def _get_new_data_qubit(self):
        q = self.next_data_idx; self.next_data_idx += 1; return q

    def _get_new_flag_qubit(self):
        q = self.next_flag_idx; self.next_flag_idx += 1; return q

    def _compute_depth(self, node, parent, F_new):
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

    def _compute_branch_marking_values(self, node, parent, G_new, F_new):
        children = [neighbor for neighbor in F_new.neighbors(node) if neighbor != parent]

        # Base case: If the node has no children, it is a leaf. Distance is 0.
        children_mark_value = sum(self._compute_branch_marking_values(child, node, G_new, F_new) for child in children)
        mark_value = int(G_new.nodes[node].get("is_mark", False))
        self.branch_mark_values[node] = children_mark_value + mark_value
        return children_mark_value + mark_value

    def extract(self, G, F, roots, dependency_graph: nx.DiGraph | None = None, primary_paths: dict[int, list[int]] | None = None):
        if self.verbose: print("=== Starting Elegant Extraction (BFS) ===")
        roots = roots if isinstance(roots, dict) else {i: r for i, r in enumerate(roots)}

        N = len([v for v in G.nodes if G.nodes[v].get("is_mark", False)])
        # num_data_flags = len([v for v in G.nodes if G.nodes[v].get("is_flag", False)])
        num_data_flags = 0

        self.next_data_idx = 0
        self.next_flag_idx = N + num_data_flags

        self.node_order = dependency_graph and flatten(list(nx.topological_generations(dependency_graph))) or []

        for root in roots.values():
            self._compute_depth(root, None, F)
            self._compute_branch_marking_values(root, None, G, F)
        self.primary_paths = {tree_id: ls[1:] for tree_id, ls in primary_paths.items()} if primary_paths is not None else {}

        # PASS 1: Grow Trees Level-by-Level
        self._grow_tree_bfs(roots, G, F)

        self._generate_detectors()
        self._generate_feedback()
        # PASS 2: Close Gaps
        return self.builder.get_circuit()


    def _grow_tree_bfs(self, roots, G_new, F_new):
        # 1. INITIALIZE ROOT
        self.init_roots(G_new, roots)

        while self.queue:
            current_qubit, node, tree_id = self.pop_next_from_queue()
            self.node_to_tree[node] = tree_id

            tree_children = [n for n in F_new.neighbors(node) if n not in self.node_to_qubit]
            non_tree_children = [n for n in G_new.neighbors(node) if n not in F_new.neighbors(node)]
            is_mark = G_new.nodes[node].get("is_mark", False)
            is_flag_node = G_new.nodes[node].get("is_flag", False)

            for child in non_tree_children:
                edge = ed(node, child)
                is_cnot_edge = G_new.edges[edge].get("edge_type", "") == "cnot"

                if is_cnot_edge:
                    self.process_cnot_edge(G_new, node, child, edge)
                elif edge in self.edge_to_flag_qubit:
                    self.close_flag(G_new, node, child, edge)
                elif is_flag_node:
                    self.take_role_of_flag(G_new, node, edge)
                else:
                    self.initialize_flag(G_new, node, child, edge)

            if is_flag_node:
                assert not tree_children
                continue

            if not tree_children:
                assert is_mark
                if self.verbose:
                    print(f"  Node {node} serves as a sink point for Q{current_qubit}")
                continue

            if is_mark:
                self.spawn_mark_cnot(G_new, node)

            primary, secondaries = self.split_primary_secondaries(tree_children)

            # 3. SECONDARY CHILDREN (Spawn new qubits)
            for child in secondaries:
                self.process_branching(G_new, node, child)

            # Gather all newly discovered nodes and their assigned qubits
            new_nodes = [(tree_id, child, new_q) for child, new_q in
                         zip(secondaries, [self.node_to_qubit[c] for c in secondaries])]
            new_nodes.append((tree_id, primary, current_qubit))
            self.queue += new_nodes

            # 4. PRIMARY CHILD (Inherit current qubit)
            self.node_to_tree[primary] = tree_id
            self.node_to_qubit[primary] = current_qubit
            self.tree_to_qubits[tree_id].add(current_qubit)

            if self.verbose:
                print(f"  Node {node} -> Primary {primary} (Inherits Q{current_qubit})")

            if (tree_id, primary, current_qubit) not in self.queue:
                self.queue.append((tree_id, primary, current_qubit))

    def initialize_flag(self, G, node: int, child, edge: tuple[int, int]):
        current_qubit = self.node_to_qubit[node]
        spider_type = G.nodes[node].get("spider_type", "Z")
        tree_id = self.node_to_tree[node]

        flag_qubit = self._get_new_flag_qubit()
        c, n = (current_qubit, flag_qubit) if spider_type == "Z" else (flag_qubit, current_qubit)
        self.builder.init_ancilla(flag_qubit, spider_type)
        self.builder.add_cnot(c, n)
        self.edge_to_flag_qubit[edge] = flag_qubit
        self.tree_to_qubits[tree_id].add(child)
        self.node_to_tree[child] = tree_id
        if self.verbose:
            print(f"  New flag initialised ({node}, {child}): CNOT Q{c} -> Q{n}")

    def process_branching(self, G, node: int, child: int):
        current_qubit = self.node_to_qubit[node]
        spider_type = G.nodes[node].get("spider_type", "Z")
        tree_id = self.node_to_tree[node]

        if self.branch_mark_values[child] > 0:
            new_q = self._get_new_data_qubit()
        else:
            new_q = self._get_new_flag_qubit()

        self.builder.init_ancilla(new_q, spider_type)
        c, n = (current_qubit, new_q) if spider_type == "Z" else (new_q, current_qubit)
        self.builder.add_cnot(c, n)

        self.node_to_qubit[child] = new_q
        self.tree_to_qubits[tree_id].add(new_q)
        self.node_to_tree[child] = tree_id

        if self.verbose:
            print(f"  Node {node} -> Branch {child}: Spawned CNOT Q{current_qubit} -> Q{new_q}")

    def spawn_mark_cnot(self, G, node: int):
        # new_q = self._get_new_flag_qubit() if is_flag_node else self._get_new_data_qubit()
        current_qubit = self.node_to_qubit[node]
        spider_type = G.nodes[node].get("spider_type", "Z")
        tree_id = self.node_to_tree[node]

        new_q = self._get_new_data_qubit()
        self.builder.init_ancilla(new_q, spider_type)
        c, n = (current_qubit, new_q) if spider_type == "Z" else (new_q, current_qubit)
        self.builder.add_cnot(c, n)
        self.tree_to_qubits[tree_id].add(new_q)
        if self.verbose:
            print(f"  Mark on {node}: Spawned CNOT Q{current_qubit} -> Q{new_q}")

    def process_cnot_edge(self, G, current_node: int, other_node: int, edge: tuple[int, int]):
        current_qubit = self.node_to_qubit[current_node]
        other_qubit = self.node_to_qubit[other_node]
        current_spider_type = G.nodes[current_node].get("spider_type", "Z")
        other_spider_type = G.nodes[other_node].get("spider_type", "Z")
        assert current_spider_type != other_spider_type

        if edge in self.processed_cnots:
            if self.verbose: print(f"  Skipping CNOT edge at {edge}")
            return

        c, n = (current_qubit, other_qubit) if current_spider_type == "Z" else (other_qubit, current_qubit)
        self.builder.add_cnot(c, n)
        self.processed_cnots.add(edge)
        if self.verbose:
            print(f"  Adding CNOT at {edge}: CNOT Q{c} -> Q{n}")

    def take_role_of_flag(self, G, node: int, edge: tuple[int, int]):
        current_qubit = self.node_to_qubit[node]
        self.edge_to_flag_qubit[edge] = current_qubit
        self.data_flags.append(current_qubit)
        if self.verbose:
            print(f"  Node {node} assumes the role of a flag qubit for edge {edge} on Q{current_qubit}.")

    def close_flag(self, G, current_node: int, other_node: int, edge: tuple[int, int]) -> None:
        current_qubit = self.node_to_qubit[current_node]
        flag_qubit = self.edge_to_flag_qubit[edge]
        current_spider_type = G.nodes[current_node].get("spider_type", "Z")
        other_spider_type = G.nodes[other_node].get("spider_type", "Z")
        assert current_spider_type == other_spider_type

        c, n = (current_qubit, flag_qubit) if current_spider_type == "Z" else (flag_qubit, current_qubit)
        self.builder.add_cnot(c, n)
        m_idx = self.builder.post_select(flag_qubit, current_spider_type)
        self.flag_measurements.append((current_node, other_node, m_idx))
        if self.verbose:
            print(f"  Flag ({current_node}, {other_node}) finalised: CNOT Q{c} -> Q{n}; PostSelect_{current_spider_type} {flag_qubit}")

    def pop_next_from_queue(self) -> tuple[int, int, int]:
        if len(self.node_order) > 0:
            node_to_process = self.node_order.pop(0)
            pop_index = [i for i, (parent, n, _) in enumerate(self.queue) if n == node_to_process][0]
        else:
            pop_index = 0
        tree_id, node, current_qubit = self.queue.pop(pop_index)
        return current_qubit, node, tree_id

    def init_roots(self, G_new, roots):
        for tree_id, root_node in roots.items():
            root_qubit = self._get_new_data_qubit()
            spider_type = G_new.nodes[root_node].get("spider_type", "Z")
            self.builder.init_ancilla(root_qubit, "X" if spider_type == "Z" else "Z")

            self.node_to_qubit[root_node] = root_qubit
            self.tree_to_qubits[tree_id].add(root_qubit)
            self.node_to_tree[root_node] = tree_id

            if self.verbose:
                print(f"Init Root {root_node} (Tree {tree_id}) -> Q{root_qubit}")

            self.queue.append((tree_id, root_node, root_qubit))
            self.processed.add((tree_id, root_node, root_qubit))

    def split_primary_secondaries(self, children: list[int]) -> tuple[int, list[int]]:
        # Sort children by depth to identify the primary branch
        if self.primary_paths:
            for path in self.primary_paths.values():
                if path and path[0] in children:
                    primary = path.pop(0)
                    secondaries = [c for c in children if c != primary]
                    return primary, secondaries

        children.sort(key=lambda c: self.depths.get(c, 0), reverse=True)
        primary = children[-1]
        secondaries = children[:-1]
        return primary, secondaries

    # def _record_meas(self, t1, t2, m_idx):
    #     if t1 == t2: self.builder.add_detector(m_idx)
    #     else:
    #         k = tuple(sorted((t1, t2)))
    #         self.link_measurements.setdefault(k, []).append(m_idx)

    def _generate_detectors(self):
        if self.verbose: print("Generating Detectors...")

        # 1. Process all stored flag measurements first
        for current_node, other_node, m_idx in self.flag_measurements:
            t1 = self.node_to_tree[current_node]
            t2 = self.node_to_tree[other_node]
            if t1 == t2:
                self.builder.add_detector(m_idx)
            else:
                k = tuple(sorted((t1, t2)))
                self.link_measurements.setdefault(k, []).append(m_idx)

        # 2. Process cross-tree link measurements
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


def extract_circuit_rooted(G, forest, roots, markings, matches, verbose=False) -> stim.Circuit:
    extractor = CatStateExtractor(StimBuilder(), verbose)
    G_exp, F_exp = expand_graph_and_forest(G, forest, markings, matches)
    return extractor.extract(G_exp, F_exp, roots)


def extract_from_expanded_graph(G_exp, F_exp, roots, dependency_graph=None, verbose=False) -> stim.Circuit:
    extractor = CatStateExtractor(StimBuilder(), verbose)
    return extractor.extract(G_exp, F_exp, roots, dependency_graph)


def implement_CNOT_circuit(cnots, num_qubits, p_2, p_mem):
    circ = stim.Circuit()
    all_qubits = set(range(num_qubits + 1))
    free_qubits = all_qubits.copy()
    for c, n in cnots:
        if c in free_qubits and n in free_qubits:
            free_qubits -= {c, n}
        else:
            if p_mem > 0:
                circ.append("DEPOLARIZE1", free_qubits, p_mem)
                circ.append("TICK")
                free_qubits = all_qubits.copy() - {c, n}
        circ.append("CNOT", [c, n])

        if p_2 > 0 and not c.is_measurement_record_target:
            circ.append("DEPOLARIZE2", [c, n], p_2)
    if p_mem > 0:
        circ.append("Z_ERROR", free_qubits, p_mem)
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
            noisy_circ.append(gate_name, targets, instruction.gate_args_copy())

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
        H 0
        CX 0 7 7 1 7 2 0 6 0 3 0 4 0 7 0 5 0 6
        M 6
        DETECTOR rec[-1]
        M 7
        DETECTOR rec[-1]
    """)


def find_mdst(G):
    """
    Finds the Absolute Center and Minimum Diameter Spanning Tree of an unweighted graph.
    """
    # 1. Compute All-Pairs Shortest Paths (APSP)/
    # Using dictionary comprehension for $O(n^2)$ lookup efficiency
    apsp = dict(nx.all_pairs_shortest_path_length(G))

    min_radius = float('inf')
    absolute_center = None
    is_edge_center = False

    # 2. Check all vertices for their eccentricity
    for v in G.nodes():
        eccentricity = max(apsp[v].values())
        if eccentricity < min_radius:
            min_radius = eccentricity
            absolute_center = v
            is_edge_center = False

    # 3. Check all edge midpoints for their eccentricity
    for u, v in G.edges():
        # Distance from an edge midpoint to any node w is min(d(u,w), d(v,w)) + 0.5
        edge_eccentricity = max(min(apsp[u][w], apsp[v][w]) + 0.5 for w in G.nodes())
        if edge_eccentricity < min_radius:
            min_radius = edge_eccentricity
            absolute_center = (u, v)
            is_edge_center = True

    # 4. Construct the Spanning Tree
    if not is_edge_center:
        # If the center is a vertex, a simple BFS tree suffices
        mdst = nx.bfs_tree(G, absolute_center).to_undirected()
    else:
        # If the center is on an edge, we subdivide the edge with a dummy node,
        # run BFS from the dummy node, and then replace the dummy with the original edge.
        u, v = absolute_center
        G_temp = G.copy()
        G_temp.remove_edge(u, v)
        dummy_node = 'TEMP_CENTER'
        G_temp.add_edge(u, dummy_node)
        G_temp.add_edge(v, dummy_node)

        mdst_temp = nx.bfs_tree(G_temp, dummy_node).to_undirected()

        # Clean up the dummy node to restore the original graph structure in the tree
        mdst_temp.remove_node(dummy_node)
        mdst_temp.add_edge(u, v)
        mdst = mdst_temp

    return mdst, absolute_center, min_radius


if __name__ == "__main__":
    from spidercat.utils import load_solution_triplet
    from spidercat.spanning_tree import match_forest_leaves_to_marked_edges, find_min_height_roots, \
    find_min_height_degree_3_roots
    from spidercat.mdsf import constrained_mdsf_generation

    N, t = 12, 4
    grf, tree, M, matchings = load_solution_triplet(N, t, 1)
    G_alt, _ = expand_graph_and_forest(grf, tree, M, matchings, expand_flags=False)
    F_alt = constrained_mdsf_generation(G_alt, 1)
    roots = find_min_height_degree_3_roots(tree)
    draw_forest_on_graph(G_alt, F_alt)

    D = build_traversal_digraph(G_alt, F_alt, roots[0])
    _, _, dependency_graph = resolve_dag_by_removing_missing_link(D)

    circ = extract_from_expanded_graph(G_alt, F_alt, roots, dependency_graph, verbose=True)
    circ.diagram('timeline-svg')