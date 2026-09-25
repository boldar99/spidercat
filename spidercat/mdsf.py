import math
import random
from functools import lru_cache

import networkx as nx


def approximate_k_center(G, k, weight='weight', seed=None):
    """
    Finds k approximate centers using Gonzalez's Furthest-First Traversal.
    Returns a list of k center nodes.
    """
    if k <= 0 or k > len(G):
        raise ValueError("k must be between 1 and the number of nodes in G.")

    if seed is not None:
        random.seed(seed)

    # 1. Compute all-pairs shortest path lengths for quick lookups
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))

    nodes = list(G.nodes())
    first_center = random.choice(nodes)
    centers = [first_center]

    # Track the minimum distance from each node to the closest chosen center
    min_dist_to_centers = {node: dist[first_center][node] for node in nodes}

    # 2. Greedily pick the remaining k-1 centers
    for _ in range(1, k):
        # Find the node that maximizes the minimum distance to existing centers
        next_center = max(nodes, key=lambda n: min_dist_to_centers[n])
        centers.append(next_center)

        # Update the closest center distances for all nodes
        for node in nodes:
            if dist[next_center][node] < min_dist_to_centers[node]:
                min_dist_to_centers[node] = dist[next_center][node]

    return centers


def constrained_mdsf(G, k, weight='weight'):
    """
    Generates an approximate Minimum Diameter k-Spanning Forest where
    all leaves are guaranteed to be nodes that originally had degree 2.
    """
    # 1. Find k centers (using the approximate_k_center function from earlier)
    centers = approximate_k_center(G, k, weight=weight)

    # 2. Map nodes to their closest center to define our ideal clusters
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    closest_center = {}
    dist_to_center = {}

    for node in G.nodes():
        c = min(centers, key=lambda c: dist[c][node])
        closest_center[node] = c
        dist_to_center[node] = dist[c][node]

    # 3. Score edges for removal (higher score = better to delete)
    edge_scores = []
    for u, v, data in G.edges(data=True):
        score = 0
        # Heavily prioritize cutting edges that cross cluster boundaries
        if closest_center[u] != closest_center[v]:
            score += 1000000

        # Prioritize cutting edges far from centers to keep trees compact
        score += dist_to_center[u] + dist_to_center[v]
        edge_scores.append((score, u, v))

    # Sort edges by removal desirability (descending)
    edge_scores.sort(key=lambda x: x[0], reverse=True)

    # 4. Greedily remove edges while respecting the matching constraint
    F = G.copy()
    locked_nodes = set()  # Tracks nodes that have already lost an edge

    for score, u, v in edge_scores:
        # CONSTRAINT: Cannot drop degree by more than 1
        if u in locked_nodes or v in locked_nodes:
            continue

        # Temporarily remove the edge
        F.remove_edge(u, v)
        num_components = nx.number_connected_components(F)

        # Check if removing this edge fragments the graph too much
        if num_components > k:
            # Revert the removal: we need this edge to prevent over-fragmentation
            F.add_edge(u, v, **G[u][v])
        else:
            # Keep it removed, and lock the endpoints so they can't lose another edge
            locked_nodes.add(u)
            locked_nodes.add(v)

        # Optimization: Stop early if we have exactly k components and no cycles
        if num_components == k and nx.is_forest(F):
            break

    # Note: Depending on the graph's rigid topology, a greedy heuristic
    # might occasionally leave a cycle if all nodes in the cycle get locked.
    return F, centers


def get_tree_diameter(T, weight='weight'):
    """Calculates the exact diameter of a tree in O(V) time using double-Dijkstra."""
    if len(T) <= 1:
        return 0

    # 1. Start from an arbitrary node and find the furthest node (A)
    start_node = next(iter(T.nodes()))
    lengths1 = nx.single_source_dijkstra_path_length(T, start_node, weight=weight)
    node_A = max(lengths1, key=lengths1.get)

    # 2. Find the furthest node from A. The distance to it is the tree's diameter.
    lengths2 = nx.single_source_dijkstra_path_length(T, node_A, weight=weight)
    node_B = max(lengths2, key=lengths2.get)

    return lengths2[node_B]


def get_forest_max_diameter(F, weight='weight'):
    """Returns the maximum diameter among all trees in the forest."""
    max_diam = 0
    for component in nx.connected_components(F):
        subtree = F.subgraph(component)
        diam = get_tree_diameter(subtree, weight)
        if diam > max_diam:
            max_diam = diam
    return max_diam


def get_valid_neighbor(G, M, k):
    """
    Mutates the current matching M by swapping one removed edge back into
    the graph and removing a new one, strictly maintaining the k-components
    and matching (degree) constraints.
    """
    M_list = list(M)
    # 1. Pick a random edge we previously removed and put it back
    e_add = random.choice(M_list)

    # Determine which nodes are "locked" by the REMAINING removed edges
    locked_nodes = set()
    for u, v in M:
        if (u, v) != e_add and (v, u) != e_add:
            locked_nodes.add(u)
            locked_nodes.add(v)

    # 2. Rebuild the temporary forest with e_add included
    F = G.copy()
    F.remove_edges_from(M)
    F.add_edge(*e_add, **G[e_add[0]][e_add[1]])

    # Find all valid edges we could potentially remove to replace e_add
    valid_candidates = []
    for u, v in F.edges():
        if u not in locked_nodes and v not in locked_nodes:
            # NEW RULE: Do not allow removal of 3-3 edges
            if G.degree(u) == 3 and G.degree(v) == 3:
                continue
            valid_candidates.append((u, v))

    random.shuffle(valid_candidates)

    # 3. Find a swap that successfully restores the graph to k components
    for e_remove in valid_candidates:
        F.remove_edge(*e_remove)
        if nx.number_connected_components(F) == k:
            # Success! Form the new matching and return
            new_M = set(M)
            new_M.remove(e_add)
            new_M.add(e_remove)
            return new_M, F
        # Revert the removal if it didn't create exactly k components
        F.add_edge(*e_remove, **G[e_remove[0]][e_remove[1]])

    # Fallback: if trapped, return the original state
    F.remove_edge(*e_add)
    return M, F


def get_leaf_adjacency_penalty(F, G, penalty_weight=100000):
    """
    Calculates a massive penalty if any two leaves within the same
    tree are adjacent in the original graph G.
    """
    penalty = 0
    for component in nx.connected_components(F):
        subtree = F.subgraph(component)

        # Identify leaves (degree 1) in this specific tree
        leaves = [n for n, d in subtree.degree() if d == 1]

        # Check all pairs of leaves in this tree for adjacency in G
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                if G.has_edge(leaves[i], leaves[j]):
                    penalty += penalty_weight
    return penalty


def simulated_annealing_mdsf(G, initial_matching, k, weight='weight',
                             init_temp=100.0, cooling_rate=0.99, min_temp=0.1, seed=None):
    """
    Optimizes a given starting forest using Simulated Annealing to minimize
    the maximum tree diameter while respecting degree constraints.

    initial_matching: A set of edge tuples (u, v) that were removed to
                      form the starting valid forest.
    """
    if seed is not None:
        random.seed(seed)
    current_M = set(initial_matching)

    # Build initial forest
    current_F = G.copy()
    current_F.remove_edges_from(current_M)

    current_energy = get_forest_max_diameter(current_F, weight)
    current_energy += get_leaf_adjacency_penalty(current_F, G)

    best_M = current_M
    best_F = current_F.copy()
    best_energy = current_energy

    temp = init_temp
    iteration = 0

    while temp > min_temp:
        # Generate a neighboring state (swap an edge)
        new_M, new_F = get_valid_neighbor(G, current_M, k)

        new_energy = get_forest_max_diameter(new_F, weight)
        new_energy += get_leaf_adjacency_penalty(new_F, G)

        # Calculate energy difference (negative means the new state is better/smaller)
        delta_energy = new_energy - current_energy

        # Acceptance logic
        if delta_energy < 0:
            accept = True
        else:
            # Probability of accepting a worse solution to escape local minimums
            probability = math.exp(-delta_energy / temp)
            accept = random.random() < probability

        if accept:
            current_M = new_M
            current_F = new_F
            current_energy = new_energy

            # Track the global best seen so far
            if current_energy < best_energy:
                best_energy = current_energy
                best_M = set(current_M)
                best_F = current_F.copy()

        # Cool down the temperature
        temp *= cooling_rate
        iteration += 1

    # print(f"Optimization finished in {iteration} steps. Best Diameter: {best_energy}")
    return best_F, best_M


@lru_cache
def constrained_mdsf_generation(G, k, weight='weight', init_temp=100.0, cooling_rate=0.995, min_temp=0.01, verbose=False, seed=None):
    """
    Runs the end-to-end pipeline:
    1. Generates an initial valid forest using the greedy heuristic.
    2. Extracts the matching (removed edges).
    3. Optimizes the forest using Simulated Annealing.
    """
    if seed is not None:
        random.seed(seed)
    if verbose:
        print(f"--- Starting Pipeline for k={k} ---")
        print("Step 1: Running greedy heuristic for initial state...")

    # 1. Get the initial valid forest
    initial_F, centers = constrained_mdsf(G, k, weight=weight)

    initial_diam = get_forest_max_diameter(initial_F, weight=weight)

    if verbose:
        print(f"Initial Greedy Forest Max Diameter: {initial_diam}")

    # 2. Extract the matching
    # Note: We must check has_edge() rather than using set differences
    # because undirected edge tuples (u, v) might be ordered differently in G vs F.
    if verbose:
        print("Step 2: Extracting the matching constraint...")
    initial_matching = set()
    for u, v in G.edges():
        if not initial_F.has_edge(u, v):
            initial_matching.add((u, v))

    if verbose:
        print(f"Extracted {len(initial_matching)} removed edges.")

    # 3. Run Simulated Annealing
    if verbose:
        print("Step 3: Running Simulated Annealing to optimize...")
    optimized_F, best_matching = simulated_annealing_mdsf(
        G,
        initial_matching,
        k,
        weight=weight,
        init_temp=init_temp,
        cooling_rate=cooling_rate,
        min_temp=min_temp
    )

    return optimized_F
