import networkx as nx
import numpy as np
import random
import math

def get_node_distances_to_sink(digraphs: list[nx.DiGraph]) -> list[dict[int, int]]:
    dist_dicts = []
    for D in digraphs:
        d_dict = {}
        if len(D.nodes) > 0:
            for layer_idx, layer in enumerate(nx.topological_generations(D.reverse())):
                for node in layer:
                    d_dict[node] = layer_idx
        dist_dicts.append(d_dict)
    return dist_dicts

def sequence_distance_cost(order: list[tuple[int, int]]) -> float:
    """
    Computes the proxy optimization cost:
    Average index distance between consecutive CNOTs acting on the same Z-spider.
    The order list contains tuples (i, j) representing edges from Z-spider i to X-spider j.
    """
    if not order:
        return 0.0
    
    last_seen = {}
    distances = []
    
    for idx, (i, j) in enumerate(order):
        if i in last_seen:
            distances.append(idx - last_seen[i])
        last_seen[i] = idx
        
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def greedy_depth_minimization(edge_list: list[tuple[int, int]], edge_groups: dict[tuple[int, int], int], z_dists: list[list[int]], x_dists: list[list[int]]) -> list[tuple[int, int]]:
    """
    List-scheduling approach to build an absolute ordering of edges.
    Dynamically prioritizes connections involving qubits that have the longest gap since their last operation.
    Ties are broken by prioritizing CNOTs that are on the longest remaining DAGs (highest remaining distance to sink).
    """
    edges_by_j = {}
    for (i, j) in edge_list:
        if j not in edges_by_j:
            edges_by_j[j] = []
        edges_by_j[j].append((i, j))
        
    for j in edges_by_j:
        edges_by_j[j].sort(key=lambda e: edge_groups[e])
        
    pointers = {j: 0 for j in edges_by_j}
    pointers_i = {i: 0 for i in range(len(z_dists))}
    active_js = list(edges_by_j.keys())
    
    depth_i = {}
    depth_j = {}
    
    order = []
    
    while active_js:
        best_j = None
        best_score = (float('inf'), float('inf'))
        
        # To avoid deterministic bias, shuffle active_js
        random.shuffle(active_js)
        
        for j in active_js:
            e = edges_by_j[j][pointers[j]]
            i = e[0]
            
            di = depth_i.get(i, 0)
            dj = depth_j.get(j, 0)
            
            rem_dist_i = z_dists[i][pointers_i[i]] if pointers_i[i] < len(z_dists[i]) else 0
            rem_dist_j = x_dists[j][pointers[j]] if pointers[j] < len(x_dists[j]) else 0
            
            # Primary objective: minimize execution depth
            # Secondary objective (tie-breaker): maximize remaining work (longest DAGs)
            score = (max(di, dj), -(rem_dist_i + rem_dist_j))
            
            if score < best_score:
                best_score = score
                best_j = j
                
        e = edges_by_j[best_j][pointers[best_j]]
        i = e[0]
        order.append(e)
        
        new_depth = max(depth_i.get(i, 0), depth_j.get(best_j, 0)) + 1
        depth_i[i] = new_depth
        depth_j[best_j] = new_depth
        
        pointers_i[i] += 1
        
        pointers[best_j] += 1
        if pointers[best_j] == len(edges_by_j[best_j]):
            active_js.remove(best_j)
            
    return order


def qubit_reuse_minimization(edge_list: list[tuple[int, int]], edge_groups: dict[tuple[int, int], int], z_dists: list[list[int]], x_dists: list[list[int]]) -> list[tuple[int, int]]:
    """
    List-scheduling approach to build an absolute ordering of edges prioritizing qubit reuse.
    It minimizes the peak number of simultaneously active spiders by strongly prioritizing
    edges that belong to already-active spiders, and finishing them as quickly as possible.
    """
    edges_by_j = {}
    rem_i = {}
    rem_j = {}
    for (i, j) in edge_list:
        if j not in edges_by_j:
            edges_by_j[j] = []
        edges_by_j[j].append((i, j))
        rem_i[i] = rem_i.get(i, 0) + 1
        rem_j[j] = rem_j.get(j, 0) + 1
        
    for j in edges_by_j:
        edges_by_j[j].sort(key=lambda e: edge_groups[e])
        
    pointers = {j: 0 for j in edges_by_j}
    pointers_i = {i: 0 for i in range(len(z_dists))}
    active_js = list(edges_by_j.keys())
    
    active_i_set = set()
    active_j_set = set()
    
    order = []
    
    while active_js:
        best_j = None
        best_score = (float('inf'), float('inf'))
        
        random.shuffle(active_js)
        
        for j in active_js:
            e = edges_by_j[j][pointers[j]]
            i = e[0]
            
            new_activations = (0 if i in active_i_set else 1) + (0 if j in active_j_set else 1)
            
            rem_dist_i = z_dists[i][pointers_i[i]] if pointers_i[i] < len(z_dists[i]) else 0
            rem_dist_j = x_dists[j][pointers[j]] if pointers[j] < len(x_dists[j]) else 0
            remaining_work = rem_dist_i + rem_dist_j
            
            # Primary: minimize new spider activations (keep peak footprint small)
            # Secondary: minimize remaining work (Shortest Remaining Processing Time to free spiders up)
            score = (new_activations, remaining_work)
            
            if score < best_score:
                best_score = score
                best_j = j
                
        e = edges_by_j[best_j][pointers[best_j]]
        i = e[0]
        order.append(e)
        
        active_i_set.add(i)
        active_j_set.add(best_j)
        
        rem_i[i] -= 1
        rem_j[best_j] -= 1
        
        if rem_i[i] == 0:
            active_i_set.remove(i)
        if rem_j[best_j] == 0:
            active_j_set.remove(best_j)
            
        pointers_i[i] += 1
        
        pointers[best_j] += 1
        if pointers[best_j] == len(edges_by_j[best_j]):
            active_js.remove(best_j)
            
    return order


def optimize_cnot_ordering(edge_list: list[tuple[int, int]], edge_groups: dict[tuple[int, int], int], z_dists: list[list[int]], x_dists: list[list[int]], heuristic: str = "sequence_distance", num_iterations: int = 5000) -> list[tuple[int, int]]:
    """
    Finds an absolute ordering of edges that minimizes the requested heuristic 
    while respecting the local ordering constraints defined by edge_groups.
    
    Available heuristics: 'sequence_distance', 'greedy', 'qubit_reuse'
    """
    if heuristic == "qubit_reuse":
        return qubit_reuse_minimization(edge_list, edge_groups, z_dists, x_dists)
    if heuristic == "greedy":
        return greedy_depth_minimization(edge_list, edge_groups, z_dists, x_dists)

    # 1. Generate an initial valid topological sort using greedy approach as a strong baseline
    current_order = greedy_depth_minimization(edge_list, edge_groups, z_dists, x_dists)
    
    # 2. Simulated Annealing for sequence distance
            
    # 3. Simulated Annealing
    current_cost = sequence_distance_cost(current_order)
    best_order = current_order.copy()
    best_cost = current_cost
    
    T_start = 10.0
    T_end = 0.01
    
    for step in range(num_iterations):
        T = T_start * (T_end / T_start) ** (step / max(1, num_iterations - 1))
        
        # Pick a random index to swap with the next one
        if len(current_order) < 2:
            break
            
        idx = random.randrange(len(current_order) - 1)
        e1 = current_order[idx]
        e2 = current_order[idx + 1]
        
        # Check if swap violates local constraints
        # Constraint: if they belong to the same X-spider j, e1's group must be <= e2's group.
        # So if we swap them, e2 comes before e1. This is only valid if e2's group <= e1's group.
        # But wait, they are originally in a valid order, so e1's group <= e2's group.
        # For the swap to be valid, they must have the SAME group.
        if e1[1] == e2[1]:
            if edge_groups[e1] != edge_groups[e2]:
                continue # Reject swap
                
        # Apply swap
        current_order[idx], current_order[idx + 1] = current_order[idx + 1], current_order[idx]
        
        new_cost = sequence_distance_cost(current_order)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = current_order.copy()
        else:
            # Reject (swap back)
            current_order[idx], current_order[idx + 1] = current_order[idx + 1], current_order[idx]
            
    return best_order


def match_edges(H: np.ndarray, non_pivots: list[int],
                z_digraphs: list[nx.DiGraph], x_digraphs: list[nx.DiGraph],
                z_candidates: list[list[int]],
                x_candidates: list[list[int]] | list[list[list[int]]],
                edge_groups: dict[tuple[int, int], int] | None = None,
                pivots: dict[int, int] | None = None,
                x_splits: list[list[list[int]]] | None = None,
                routing_heuristic: str = "sequence_distance") -> list[
    tuple[tuple[int, int], tuple[int, int]]]:
    
    # 1. Identify all required logical connections
    edge_list = [
        (i, j)
        for i, r in enumerate(H)
        for j, x in enumerate(r[non_pivots])
        if x == 1
    ]

    # Resolve edge_groups mapping if not provided
    if edge_groups is None:
        if x_splits is not None:
            if pivots is None:
                from spiderstate.utils import find_pivots_in_matrix
                pivots, _ = find_pivots_in_matrix(H)
            edge_groups = {}
            for (i, j) in edge_list:
                pivot_q = pivots[i]
                edge_groups[(i, j)] = next(
                    k for k, piece in enumerate(x_splits[j]) if pivot_q in piece
                )
        else:
            edge_groups = {e: 0 for e in edge_list}

    # Format candidate pools
    z_pools = [[c for c in pool] for pool in z_candidates]
    if x_candidates and isinstance(x_candidates[0], list) and x_candidates[0] and isinstance(x_candidates[0][0], list):
        x_pools = [[[c for c in group] for group in spider_groups] for spider_groups in x_candidates]
    else:
        x_pools = [[[c for c in pool]] for pool in x_candidates]

    # Precompute distance to sink for actual nodes
    z_dist_dicts = get_node_distances_to_sink(z_digraphs)
    x_dist_dicts = get_node_distances_to_sink(x_digraphs)
    
    z_cnot_dists = []
    for i, pool in enumerate(z_candidates):
        z_cnot_dists.append([z_dist_dicts[i].get(node, 0) for node in pool])
        
    x_cnot_dists = []
    for j, spider_groups in enumerate(x_candidates):
        if spider_groups and isinstance(spider_groups[0], list):
            nodes = [node for grp in spider_groups for node in grp]
        else:
            nodes = spider_groups
        x_cnot_dists.append([x_dist_dicts[j].get(node, 0) for node in nodes])

    # 2. Optimize the absolute ordering of CNOT edges
    optimized_order = optimize_cnot_ordering(
        edge_list, edge_groups, z_cnot_dists, x_cnot_dists, heuristic=routing_heuristic, num_iterations=10000
    )
    
    # 3. Match edges purely based on the absolute ordering (earliest available leg)
    final_matching = []
    
    for edge in optimized_order:
        i, j = edge
        grp = edge_groups[edge]
        
        # Because we pick the earliest available legs, no global cycles can be formed
        z_val = z_pools[i].pop(0)
        x_val = x_pools[j][grp].pop(0)
        
        final_matching.append((edge, (z_val, x_val)))
        
    return final_matching
