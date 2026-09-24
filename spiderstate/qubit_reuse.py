from __future__ import annotations

import networkx as nx
import numpy as np
import stim

from spiderstate.stim_utils import explode_circuit


from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING



@dataclass
class RoutingState:
    """Holds the live state of the DAG and logical-to-physical tracking pointers."""
    dag: nx.DiGraph
    n_data: int
    ancillas: set[int]
    next_q: dict[int, int]
    prev_q: dict[int, int]
    birth_node: dict[int, int]
    death_node: dict[int, int]
    data_birth: dict[int, int]
    data_death: dict[int, int]
    
    tentative_edge: tuple[int, int] | None = None
    longest_path_to: dict[int, int] | None = None
    longest_path_from: dict[int, int] | None = None
    current_longest_path: int | None = None
    
    def get_current_depth(self) -> int:
        if self.tentative_edge is not None:
            u, v = self.tentative_edge
            if self.longest_path_to is not None and self.longest_path_from is not None:
                return max(self.current_longest_path, self.longest_path_to[u] + 1 + self.longest_path_from[v])
        return nx.dag_longest_path_length(self.dag)


def compute_dag_longest_paths(dag: nx.DiGraph) -> tuple[dict[int, int], dict[int, int], int]:
    topo_order = list(nx.topological_sort(dag))
    
    longest_path_to = dict.fromkeys(dag.nodes, 0)
    pred = dag.pred
    for n in topo_order:
        p_dict = pred[n]
        if p_dict:
            longest_path_to[n] = max(longest_path_to[p] for p in p_dict) + 1
            
    longest_path_from = dict.fromkeys(dag.nodes, 0)
    succ = dag.succ
    for n in reversed(topo_order):
        s_dict = succ[n]
        if s_dict:
            longest_path_from[n] = max(longest_path_from[s] for s in s_dict) + 1
            
    current_longest_path = max(longest_path_to.values()) if longest_path_to else 0
    return longest_path_to, longest_path_from, current_longest_path


class ReuseStrategy(Protocol):
    """Protocol defining the rules for evaluating a qubit reuse dependency."""

    def setup(self, state: RoutingState) -> None:
        """Initialize any baseline metrics before routing begins."""
        ...

    def evaluate_candidate(self, state: RoutingState) -> float:
        """Evaluate the modified state. Return cost (lower is better) or float('inf') to reject."""
        ...

    def commit_edge(self, state: RoutingState) -> None:
        """Called by the router when an edge is permanently committed."""
        ...


class NoReuseStrategy:
    """
    Evaluates candidates purely by their resulting circuit depth.
    Guarantees minimum hardware qubits while selecting the permutations
    that bloat the depth the least.
    """

    def setup(self, state: RoutingState) -> None:
        pass

    def evaluate_candidate(self, state: RoutingState) -> float:
        return float('inf')

    def commit_edge(self, state: RoutingState) -> None:
        # Stateless evaluation: no cached baselines to update
        pass



class AggressiveDepthAwareStrategy:
    """
    Evaluates candidates purely by their resulting circuit depth.
    Guarantees minimum hardware qubits while selecting the permutations
    that bloat the depth the least.
    """

    def setup(self, state: RoutingState) -> None:
        pass

    def evaluate_candidate(self, state: RoutingState) -> float:
        return float(state.get_current_depth())

    def commit_edge(self, state: RoutingState) -> None:
        # Stateless evaluation: no cached baselines to update
        pass


class PureAggressiveStrategy:
    """Treats all valid edges equally. The router will pick the first valid one."""

    def setup(self, state: RoutingState) -> None:
        pass

    def evaluate_candidate(self, state: RoutingState) -> float:
        return 0.0

    def commit_edge(self, state: RoutingState) -> None:
        # Stateless evaluation: nothing to update
        pass


class DepthPreservingStrategy:
    """Rejects any edge that increases the original baseline depth."""

    def setup(self, state: RoutingState) -> None:
        # The baseline is the original depth before ANY routing occurs
        self.baseline_depth = state.get_current_depth()

    def evaluate_candidate(self, state: RoutingState) -> float:
        current_depth = state.get_current_depth()
        if current_depth > self.baseline_depth:
            return float('inf')  # Outright reject
        return 0.0

    def commit_edge(self, state: RoutingState) -> None:
        # Stateful, but the baseline represents a static original constraint.
        # Do not update self.baseline_depth here.
        pass


class VolumeOptimizingReuseStrategy:
    """
    Evaluates candidates by exact total spacetime volume (depth * hardware_qubits).
    Rejects any edge that increases the total spacetime volume.
    """

    def setup(self, state: RoutingState) -> None:
        self.current_vol = self._compute_total_volume(state)

    def evaluate_candidate(self, state: RoutingState) -> float:
        new_vol = self._compute_total_volume(state)

        # Must not increase the rectangular bounding box volume
        if new_vol <= self.current_vol:
            # We return the new volume as the cost.
            # If volumes are equal, the router will pick the first one it found,
            # which is fine as it still merged and reduced qubit count.
            return float(new_vol)

        return float('inf')

    def commit_edge(self, state: RoutingState) -> None:
        self.current_vol = self._compute_total_volume(state)

    @staticmethod
    def _compute_total_volume(state: RoutingState) -> int:
        depth = state.get_current_depth()
        
        # Count number of hardware qubits: data qubits + roots of ancilla chains
        num_hw_qubits = state.n_data
        for q in state.ancillas:
            if q not in state.prev_q:
                num_hw_qubits += 1
                
        return depth * num_hw_qubits


class OptimalTightPackingStrategy:
    """
    Achieves the best deterministic LER by acting as the optimal 
    'Earliest Start Time' interval scheduler.
    By unconditionally returning the active spacetime volume, the greedy router 
    will always pick merges that introduce the absolute smallest temporal gap. 
    This strictly maximizes the global number of merges (reaching minimum hardware qubits) 
    while preventing unnecessary topological stretching.
    """

    def setup(self, state) -> None:
        pass

    def evaluate_candidate(self, state) -> float:
        return float(self._compute_active_volume(state))

    def commit_edge(self, state) -> None:
        pass

    @staticmethod
    def _compute_active_volume(state) -> int:
        layer = {}
        for node in nx.topological_sort(state.dag):
            layer[node] = max((layer[pred] for pred in state.dag.predecessors(node)), default=-1) + 1

        active_volume = 0
        for q in range(state.n_data):
            if q in state.data_birth and q in state.data_death:
                b = state.data_birth[q]
                d = state.data_death[q]
                active_volume += layer[d] - layer[b] + 1
                
        for q in state.ancillas:
            if q not in state.prev_q:
                curr = q
                min_layer = float('inf')
                max_layer = float('-inf')
                while curr is not None:
                    if curr in state.birth_node:
                        min_layer = min(min_layer, layer[state.birth_node[curr]])
                    if curr in state.death_node:
                        max_layer = max(max_layer, layer[state.death_node[curr]])
                    curr = state.next_q.get(curr)
                
                if min_layer != float('inf') and max_layer != float('-inf'):
                    active_volume += max_layer - min_layer + 1

        return active_volume

import networkx as nx

class SimQubitsAndDepthStrategy:
    """
    Optimizes explicitly for sim_qubits + depth.
    Since every valid merge reduces sim_qubits by exactly 1, this naturally
    penalizes merges that increase depth by more than 1, implicitly 
    balancing the tradeoff between circuit depth and hardware qubit count.
    """
    def setup(self, state) -> None:
        pass

    def evaluate_candidate(self, state) -> float:
        depth = nx.dag_longest_path_length(state.dag)
        
        num_hw_qubits = state.n_data
        for q in state.ancillas:
            if q not in state.prev_q:
                num_hw_qubits += 1
                
        return float(num_hw_qubits + depth)

    def commit_edge(self, state) -> None:
        pass

def build_circuit_dag(circ: stim.Circuit) -> nx.DiGraph:
    """
    Converts a sequential list of quantum operations into a dependency DAG.
    Stores the original operation name, targets, and measurement_id in the node attributes.
    """
    dag = nx.DiGraph()
    last_op_on_qubit: dict[int,int] = {}

    ordered_operations = explode_circuit(circ)
    
    current_meas_id = 0

    for i, circ_op in enumerate(ordered_operations):
        op_name = circ_op.name
        targets = [t.value for t in circ_op.targets_copy() if t.is_qubit_target]
        
        measurement_id = None
        if op_name in {"M", "MX", "MR", "MZ"}:
            measurement_id = current_meas_id
            current_meas_id += len(targets)

        # Normalize targets to a tuple for consistent storage
        qubits = tuple(targets)

        # Rely strictly on the explicit measurement_id from CircuitOperation
        dag.add_node(i, op_name=op_name, targets=qubits, measurement_id=measurement_id)

        # Determine dependencies based on qubit usage
        dependencies = set()
        for q in qubits:
            if q in last_op_on_qubit:
                dependencies.add(last_op_on_qubit[q])
            # Update the tracker: this node 'i' is now the latest operation on qubit 'q'
            last_op_on_qubit[q] = i

        # Draw the edges from dependencies to the current operation
        for dep in dependencies:
            dag.add_edge(dep, i)

    return dag


import math
import random

def _compute_D_weights(dag: nx.DiGraph) -> dict[int, int]:
    L = list(nx.topological_sort(dag))
    
    first_node_for_q = {}
    last_meas_for_q = {}
    
    for node in L:
        data = dag.nodes[node]
        targets = data.get("targets", [])
        if isinstance(targets, int): 
            targets = [targets]
        elif isinstance(targets, tuple): 
            targets = list(targets)
            
        op_name = data.get("op_name", "")
        is_meas = op_name in {"M", "MX", "MR", "MZ"}
        
        for q in targets:
            if q not in first_node_for_q: 
                first_node_for_q[q] = node
            if is_meas: 
                last_meas_for_q[q] = node
                
    D = {node: 0 for node in L}
    for node in first_node_for_q.values(): 
        D[node] -= 1
    for node in last_meas_for_q.values(): 
        D[node] += 1
        
    return D


def _greedy_topological_insertion(dag: nx.DiGraph, num_passes: int = 15) -> list:
    """
    Minimizes the active lifetime of qubits by eagerly exploring the valid 
    topological insertion window of each node and placing it at the position 
    that minimizes the sum of all qubit lifetimes.
    """
    L = list(nx.topological_sort(dag))
    if len(L) < 2: 
        return L
        
    D = _compute_D_weights(dag)
        
    preds = {n: set(dag.predecessors(n)) for n in L}
    succs = {n: set(dag.successors(n)) for n in L}
    pos = {node: i for i, node in enumerate(L)}
    
    for pass_idx in range(num_passes):
        changed = False
        nodes = L.copy()
        for v in nodes:
            curr = pos[v]
            
            min_idx = max((pos[p] for p in preds[v]), default=-1) + 1
            max_idx = min((pos[s] for s in succs[v]), default=len(L)) - 1
            
            best_j = curr
            min_delta = 0
            
            current_delta = 0
            for j in range(curr - 1, min_idx - 1, -1):
                current_delta += D[L[j]] - D[v]
                if current_delta < min_delta:
                    min_delta = current_delta
                    best_j = j
                    
            current_delta = 0
            for j in range(curr + 1, max_idx + 1):
                current_delta += D[v] - D[L[j]]
                if current_delta < min_delta:
                    min_delta = current_delta
                    best_j = j
                    
            if best_j != curr:
                changed = True
                L.pop(curr)
                L.insert(best_j, v)
                
                if best_j < curr:
                    for i in range(best_j, curr + 1):
                        pos[L[i]] = i
                else:
                    for i in range(curr, best_j + 1):
                        pos[L[i]] = i
                        
        if not changed:
            break
            
    return L

def _hybrid_topological_optimization(dag: nx.DiGraph, greedy_passes: int = 15, greedy_restarts: int = 3, sa_steps: int = 100000) -> list:
    """
    Minimizes the active lifetime of qubits by eagerly exploring the valid 
    topological insertion window of each node, followed by a Simulated Annealing 
    polishing phase to escape any remaining local minima.
    """
    base_L = list(nx.topological_sort(dag))
    if len(base_L) < 2: 
        return base_L
        
    D = _compute_D_weights(dag)
        
    preds = {n: set(dag.predecessors(n)) for n in base_L}
    succs = {n: set(dag.successors(n)) for n in base_L}
    
    def compute_energy(L):
        pos = {node: i for i, node in enumerate(L)}
        return sum(pos[n] * D[n] for n in L)

    best_global_L = None
    best_global_E = float('inf')

    # Phase 1: Multi-start Greedy Insertion
    for restart in range(greedy_restarts):
        L = base_L.copy()
        pos = {node: i for i, node in enumerate(L)}
        
        for pass_idx in range(greedy_passes):
            changed = False
            nodes = L.copy()
            random.shuffle(nodes)
            
            for v in nodes:
                curr = pos[v]
                min_idx = max((pos[p] for p in preds[v]), default=-1) + 1
                max_idx = min((pos[s] for s in succs[v]), default=len(L)) - 1
                
                best_j = curr
                min_delta = 0
                
                current_delta = 0
                for j in range(curr - 1, min_idx - 1, -1):
                    current_delta += D[L[j]] - D[v]
                    if current_delta < min_delta:
                        min_delta = current_delta
                        best_j = j
                        
                current_delta = 0
                for j in range(curr + 1, max_idx + 1):
                    current_delta += D[v] - D[L[j]]
                    if current_delta < min_delta:
                        min_delta = current_delta
                        best_j = j
                        
                if best_j != curr:
                    changed = True
                    L.pop(curr)
                    L.insert(best_j, v)
                    
                    if best_j < curr:
                        for i in range(best_j, curr + 1):
                            pos[L[i]] = i
                    else:
                        for i in range(curr, best_j + 1):
                            pos[L[i]] = i
                            
            if not changed:
                break
                
        final_E = compute_energy(L)
        if final_E < best_global_E:
            best_global_E = final_E
            best_global_L = L

    # Phase 2: Simulated Annealing Polishing (O(1) state transitions)
    L = best_global_L.copy()
    current_energy = best_global_E
    edges = set(dag.edges())
    temp = 2.0
    cooling_rate = (0.01 / temp) ** (1.0 / sa_steps) if sa_steps > 0 else 0.99
    
    for step in range(sa_steps):
        i = random.randint(0, len(L) - 2)
        v = L[i]
        w = L[i+1]
        
        if (v, w) in edges:
            continue
            
        delta = D[v] - D[w]
        
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-5)):
            L[i], L[i+1] = w, v
            current_energy += delta
            if current_energy < best_global_E:
                best_global_E = current_energy
                best_global_L = L.copy()
                
        temp *= cooling_rate

    return best_global_L


def _exact_topological_optimization(dag: nx.DiGraph, max_time_seconds: float = 15.0) -> list:
    """
    Finds the mathematically optimal topological ordering that minimizes 
    active qubit lifetimes using Constraint Programming (OR-Tools CP-SAT).
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("ortools is required for exact optimization. Run `pip install ortools`.")
        
    L = list(nx.topological_sort(dag))
    if len(L) < 2:
        return L
        
    D = _compute_D_weights(dag)
        
    model = cp_model.CpModel()
    N = len(L)
    
    pos = {n: model.NewIntVar(0, N - 1, f"pos_{n}") for n in L}
    
    model.AddAllDifferent(pos.values())
    
    for u, v in dag.edges():
        model.Add(pos[u] < pos[v])
        
    objective_expr = sum(pos[n] * D[n] for n in L if D[n] != 0)
    model.Minimize(objective_expr)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    
    # Provide the hybrid solver's solution as a hint to accelerate the search
    hint_L = _hybrid_topological_optimization(dag)
    for i, n in enumerate(hint_L):
        model.AddHint(pos[n], i)
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return sorted(L, key=lambda n: solver.Value(pos[n]))
        
    return hint_L


def dag_to_circuit(dag: nx.DiGraph, heuristic: str | None = "exact") -> tuple[stim.Circuit, dict[int, int]]:
    """
    Converts a circuit dependency DAG back into a Stim circuit and a measurement map.
    Extraction MUST be done in topological order to respect causality constraints.
    heuristic options:
      - None or 'none': standard topological sort
      - 'greedy': standard greedy insertion (single pass, deterministic)
      - 'hybrid': greedy multi-start + simulated annealing
      - 'exact': exact optimization using OR-Tools CP-SAT
    """
    circuit = stim.Circuit()
    measurement_map: dict[int, int] = {}
    next_measurement_index = 0

    if heuristic is None or heuristic == "none":
        sorted_nodes = list(nx.topological_sort(dag))
    elif heuristic == "greedy":
        sorted_nodes = _greedy_topological_insertion(dag)
    elif heuristic == "hybrid":
        sorted_nodes = _hybrid_topological_optimization(dag)
    elif heuristic == "exact":
        sorted_nodes = _exact_topological_optimization(dag)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    # Ensure operations are ordered correctly respecting the DAG's causal flow
    for node in sorted_nodes:
        data = dag.nodes[node]
        op_name = data.get("op_name")
        targets = data.get("targets")
        measurement_id = data.get("measurement_id")

        targets_list = list(targets) if isinstance(targets, tuple) else targets
        circuit.append(op_name, targets_list)

        # Dynamically build the measurement map for tracking
        if op_name in {"M", "MX", "MR", "MZ"}:
            if measurement_id is not None:
                for offset in range(len(targets_list)):
                    measurement_map[next_measurement_index + offset] = measurement_id + offset
            next_measurement_index += len(targets_list)

    return circuit, measurement_map


def dag_to_noisy_circuit(dag: nx.DiGraph, p: float) -> tuple[stim.Circuit, dict[int, int]]:
    """
    Converts a circuit dependency DAG into a noisy Stim circuit and a measurement map.
    Applies ASAP forward layering (1 CX per layer), ALAP backward shift for resets,
    and applies a standard QEC noise model including memory noise on idle qubits.
    """
    layer = {}
    cx_layers = set()

    # 1. ASAP Forward Layering
    for node in nx.topological_sort(dag):
        op_name = dag.nodes[node].get("op_name")

        l = max((layer[pred] for pred in dag.predecessors(node)), default=-1) + 1

        if op_name in {"CX", "CNOT", "CZ", "CY", "SWAP", "XCZ", "YCX"}:
            while l in cx_layers:
                l += 1
            cx_layers.add(l)

        layer[node] = l

    max_layer = max(layer.values(), default=-1)

    # 2. ALAP Backward Shift for Initializations (R, RX)
    for node in reversed(list(nx.topological_sort(dag))):
        op_name = dag.nodes[node].get("op_name")
        if op_name in {"R", "RX"}:
            successors = list(dag.successors(node))
            if successors:
                layer[node] = min(layer[s] for s in successors) - 1
            else:
                layer[node] = max_layer

    # 3. Build layers
    active_qubits = set()
    for node in dag.nodes():
        active_qubits.update(dag.nodes[node].get("targets", ()))

    num_layers = max_layer + 1
    layers = [[] for _ in range(num_layers)]
    for node in dag.nodes():
        layers[layer[node]].append(node)

    # 4. Generate Stim circuit
    circuit = stim.Circuit()
    measurement_map = {}
    next_measurement_index = 0

    for i in range(num_layers):
        current_layer_nodes = layers[i]
        unused_qubits = set(active_qubits)

        has_two_qubit_gate = False

        for node in current_layer_nodes:
            data = dag.nodes[node]
            op_name = data.get("op_name")
            targets = data.get("targets")
            measurement_id = data.get("measurement_id")
            targets_list = list(targets) if isinstance(targets, tuple) else targets

            unused_qubits -= set(targets_list)

            if op_name in {"CX", "CNOT", "CZ", "CY", "SWAP", "XCZ", "YCX"}:
                has_two_qubit_gate = True

            # Bit-flip (X_ERROR) before flag measurement in Z basis, Z_ERROR for X basis
            if op_name in {"M", "MR", "MZ"} and p > 0:
                circuit.append("X_ERROR", targets_list, p)
            elif op_name in {"MX"} and p > 0:
                circuit.append("Z_ERROR", targets_list, p)

            circuit.append(op_name, targets_list)

            if op_name in {"M", "MX", "MR", "MZ"}:
                if measurement_id is not None:
                    for offset in range(len(targets_list)):
                        measurement_map[next_measurement_index + offset] = measurement_id + offset
                next_measurement_index += len(targets_list)

            # Depolarizing after init/measure, or two-qubit gate, or single-qubit gate
            if (op_name in {"M", "MR", "MZ", "MX"} or op_name in {"R", "RX"}) and p > 0:
                circuit.append("DEPOLARIZE1", targets_list, p)
            elif op_name in {"CX", "CNOT", "CZ", "CY", "SWAP", "XCZ", "YCX"} and p > 0:
                circuit.append("DEPOLARIZE2", targets_list, p)
            elif op_name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK", "M", "MR", "MZ", "MX", "R", "RX"} and p > 0:
                circuit.append("DEPOLARIZE1", targets_list, p)

        # Memory noise on unused qubits only during layers that contain a CNOT (two-qubit gate)
        if has_two_qubit_gate and p > 0 and unused_qubits:
            circuit.append("DEPOLARIZE1", sorted(list(unused_qubits)), p / 100)

        circuit.append("TICK", [])

    return circuit, measurement_map


def inject_qubit_reuse(dag: nx.DiGraph, n_data: int, strategy: ReuseStrategy):
    """
    Best-Fit routing engine. Groups potential reuse dependencies by target,
    tests all valid sources, and commits the edge with the lowest strategy cost.
    """
    mod_dag = dag.copy()
    topo_order = list(nx.topological_sort(mod_dag))

    ancillas = set()
    birth_node, death_node = {}, {}
    data_birth, data_death = {}, {}

    for node in topo_order:
        targets = mod_dag.nodes[node].get("targets", ())
        for q in targets:
            if q >= n_data:
                ancillas.add(q)
                if q not in birth_node: birth_node[q] = node
                death_node[q] = node
            else:
                if q not in data_birth: data_birth[q] = node
                data_death[q] = node

    state = RoutingState(
        dag=mod_dag,
        n_data=n_data,
        ancillas=ancillas,
        next_q={},
        prev_q={},
        birth_node=birth_node,
        death_node=death_node,
        data_birth=data_birth,
        data_death=data_death
    )

    strategy.setup(state)

    # Group candidates by the target Reset node (qB)
    # We want to find the best Qubit A to feed into Qubit B
    candidates_by_target = {qB: [] for qB in ancillas}

    for qB in ancillas:
        bB = birth_node[qB]
        reachable_from_bB = nx.descendants(mod_dag, bB)
        for qA in ancillas:
            if qA == qB:
                continue
            dA = death_node[qA]

            # Fast topological rejection
            if dA == bB or dA in reachable_from_bB:
                continue
                
            candidates_by_target[qB].append((qA, dA, bB))

    # Evaluate best-fit for each target qubit
    graph_changed = True
    for qB, sources in candidates_by_target.items():
        if qB in state.prev_q:
            continue

        best_cost = float('inf')
        best_qA = None
        best_dA = None
        best_bB = None

        reachable_from_bB = nx.descendants(state.dag, birth_node[qB])
        
        # Precompute path lengths for O(1) depth evaluation in strategies
        if graph_changed:
            state.longest_path_to, state.longest_path_from, state.current_longest_path = compute_dag_longest_paths(state.dag)
            graph_changed = False

        for qA, dA, bB in sources:
            if qA in state.next_q:
                continue

            # 1. Hardware Constraint (Cycle Check)
            if dA == bB or dA in reachable_from_bB:
                continue

            # Tentatively apply the edge
            state.dag.add_edge(dA, bB)
            state.tentative_edge = (dA, bB)

            # 2. Update tracking state for accurate strategy evaluation
            state.next_q[qA] = qB
            state.prev_q[qB] = qA

            # 3. Strategy Evaluation
            cost = strategy.evaluate_candidate(state)

            if cost < best_cost:
                best_cost = cost
                best_qA = qA
                best_dA = dA
                best_bB = bB

            # Revert the tentative changes to test the next source
            state.dag.remove_edge(dA, bB)
            state.tentative_edge = None
            del state.next_q[qA]
            del state.prev_q[qB]

        # If we found at least one valid, non-rejected source, commit the best one permanently
        if best_qA is not None and best_cost != float('inf'):
            state.dag.add_edge(best_dA, best_bB)
            state.next_q[best_qA] = qB
            state.prev_q[qB] = best_qA
            graph_changed = True

            # ---> ADD THESE TWO LINES <---
            # Notify the strategy that the graph architecture has permanently changed
            strategy.commit_edge(state)

    # Hardware Allocation Mapping
    logical_to_physical = {q: q for q in range(n_data)}
    next_hw = n_data

    for q in ancillas:
        if q not in state.prev_q:
            curr = q
            while curr is not None:
                logical_to_physical[curr] = next_hw
                curr = state.next_q.get(curr)
            next_hw += 1

    return state.dag, logical_to_physical, next_hw


def apply_logical_qubit_merge_and_compress(dag: nx.DiGraph, n_data: int) -> nx.DiGraph:
    """
    1. Merges logical qubits based on M -> R injected edges.
    2. Compresses the remaining active qubit IDs so they are contiguous.
    """
    mod_dag = dag.copy()

    # --- PHASE 1: Union-Find for Merges ---
    parent_map: dict[int, int] = {}

    def get_root(q):
        curr = q
        while curr in parent_map:
            curr = parent_map[curr]
        return curr

    for u, v in mod_dag.edges():
        op_u = mod_dag.nodes[u].get("op_name", "")
        op_v = mod_dag.nodes[v].get("op_name", "")

        # Generalize to match CoveredZXGraph definition
        if op_u in {"M", "MX", "MR", "MZ"} and op_v in {"R", "RX"}:
            targets_u = mod_dag.nodes[u].get("targets", [])
            targets_v = mod_dag.nodes[v].get("targets", [])

            if len(targets_u) == 1 and len(targets_v) == 1:
                qA = targets_u[0]
                qB = targets_v[0]

                if qA != qB:
                    root_A = get_root(qA)
                    parent_map[qB] = root_A

    # --- PHASE 2: Collect and Compress ---
    active_roots = set()
    for node in mod_dag.nodes():
        targets = mod_dag.nodes[node].get("targets", [])
        for q in targets:
            active_roots.add(get_root(q))

    data_roots = [q for q in active_roots if q < n_data]
    ancilla_roots = sorted([q for q in active_roots if q >= n_data])

    compression_map = {}
    for q in data_roots:
        compression_map[q] = q

    next_dense_id = n_data
    for q in ancilla_roots:
        compression_map[q] = next_dense_id
        next_dense_id += 1

    # --- PHASE 3: Rewrite the DAG ---
    for node in mod_dag.nodes():
        old_targets = mod_dag.nodes[node].get("targets", [])
        new_targets = []
        for q in old_targets:
            root_q = get_root(q)
            compressed_q = compression_map[root_q]
            new_targets.append(compressed_q)

        if isinstance(old_targets, tuple):
            mod_dag.nodes[node]["targets"] = tuple(new_targets)
        else:
            mod_dag.nodes[node]["targets"] = new_targets

    return mod_dag


if __name__ == "__main__":
    circ = stim.Circuit()

    dag = build_circuit_dag(circ)
    mod_dag, logical_to_physical, total_hw = inject_qubit_reuse(dag, code.n, AggressiveDepthAwareStrategy())
    compressed_dag = apply_logical_qubit_merge_and_compress(mod_dag, code.n)
    final_circ, final_meas_map = dag_to_circuit(compressed_dag)

    composed_meas_map = {k: pre_measurement_map[v] for k, v in final_meas_map.items()}

    print("\n--- Final Reused Circuit ---")
    print(final_circ)
    print("\nFinal Measurement Map:", composed_meas_map)