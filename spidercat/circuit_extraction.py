from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spidercat.draw import draw_spanning_forest_solution
from spidercat.utils import ed


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


# --- 2. The Main Extractor Class ---
class CatStateExtractor:
    def __init__(self, builder: CircuitBuilder, verbose=False):
        self.builder = builder
        self.verbose = verbose

        self.node_to_qubit = {}
        self.edge_to_flag = {}
        self.tree_to_qubits = defaultdict(set)
        self.tree_of_node = {}
        self.link_measurements = {}
        self.depths = {}

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

    def extract(self, G_new, F_new, roots):
        if self.verbose: print("=== Starting Elegant Extraction (BFS) ===")

        self.next_data_idx = 0
        self.next_flag_idx = len([v for v in G_new.nodes if G_new.nodes[v].get("is_mark", False)])

        for root in roots.values():
            self._compute_depth(root, None, F_new)

        # PASS 1: Grow Trees Level-by-Level
        for tree_id, root in roots.items():
            self._grow_tree_bfs(root, tree_id, G_new, F_new)

        self._generate_detectors()
        self._generate_feedback()
        # PASS 2: Close Gaps
        # self._close_gaps(G_new, F_new)
        return self.builder.get_circuit()


    def _grow_tree_bfs(self, root_node, tree_id, G_new, F_new):
        # 1. INITIALIZE ROOT
        is_flag = G_new.nodes[root_node].get("is_flag", False)
        root_qubit = self._get_new_flag_qubit() if is_flag else self._get_new_data_qubit()
        self.builder.init_ancilla(root_qubit)
        self.builder.add_h(root_qubit)

        self.node_to_qubit[root_node] = root_qubit
        self.tree_to_qubits[tree_id].add(root_qubit)
        self.tree_of_node[root_node] = tree_id

        if self.verbose:
            print(f"Init Root {root_node} (Tree {tree_id}) -> Q{root_qubit}")

        # Queue stores: (node, current_qubit)
        queue = [root_node]

        while queue:
            node = queue.pop(0)
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
                if self.verbose and is_mark:
                    print(f"  Node {node} serves as a sink point for Q{current_qubit}")
                continue

            if is_mark:
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
                is_flag_child = G_new.nodes[child].get("is_flag", False)

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


def extract_circuit_rooted(G, forest, roots, markings, matches, verbose=False) -> stim.Circuit:
    extractor = CatStateExtractor(StimBuilder(), verbose)
    G_exp, F_exp = expand_graph_and_forest(G, forest, markings, matches)
    return extractor.extract(G_exp, F_exp, roots)


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


if __name__ == "__main__":
    from spidercat.utils import load_solution_triplet
    from spidercat.spanning_tree import find_min_height_roots, match_forest_leaves_to_marked_edges

    grf, forest, M, matchings = load_solution_triplet(12, 2, 1)
    roots = find_min_height_roots(forest)
    draw_spanning_forest_solution(grf, forest, M, matchings, roots)
    extract_circuit_rooted(grf, forest, roots, M, matchings, verbose=False)
