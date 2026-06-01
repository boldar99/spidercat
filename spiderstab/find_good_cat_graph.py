import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pysat.card import CardEnc
from pysat.formula import IDPool, CNF
from pysat.solvers import Glucose42


# ---------------------------------------------------------
# 1. Parameterized 2n Tree Generation (BFS Expansion)
# ---------------------------------------------------------
def generate_balanced_2n_tree(n: int) -> tuple[nx.Graph, list]:
    """
    Generates an unrooted tree with exactly 2n leaves.
    Internal nodes have degree 3, leaves have degree 1.
    Uses BFS expansion to ensure the tree is as 'well-balanced' as mathematically possible.
    """
    if n < 2:
        raise ValueError("n must be >= 2 to form a valid 3-regular internal structure.")

    G = nx.Graph()
    G.add_edge("core_1", "core_2")

    # Base structure: 4 leaves
    G.add_edge("core_1", "L_1")
    G.add_edge("core_1", "L_2")
    G.add_edge("core_2", "L_3")
    G.add_edge("core_2", "L_4")

    leaves = ["L_1", "L_2", "L_3", "L_4"]
    expand_queue = deque(leaves)

    node_counter = 5
    target_leaves = 2 * n

    # Each expansion removes 1 leaf and adds 2 (Net gain: +1 leaf)
    # We expand until we reach the target 2n leaves.
    while len(leaves) < target_leaves:
        # Pop the shallowest leaf to expand, maintaining structural balance
        leaf_to_expand = expand_queue.popleft()
        leaves.remove(leaf_to_expand)

        new_l1 = f"L_{node_counter}"
        new_l2 = f"L_{node_counter + 1}"
        node_counter += 2

        G.add_edge(leaf_to_expand, new_l1)
        G.add_edge(leaf_to_expand, new_l2)

        leaves.extend([new_l1, new_l2])
        expand_queue.extend([new_l1, new_l2])

    return G, leaves


# ---------------------------------------------------------
# 2. Visualization
# ---------------------------------------------------------
def visualize_cat_graph(G: nx.Graph, M_edges: list[tuple]):
    """
    Visualizes the graph G = T U M.
    Tree edges are black/solid. Matching edges are red/dashed.
    """
    G_combined = G.copy()
    G_combined.add_edges_from(M_edges)

    # Kamada-Kawai layout handles unrooted symmetric graphs exceptionally well
    pos = nx.kamada_kawai_layout(G_combined)

    plt.figure(figsize=(12, 10))

    # Draw Tree (T)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=2.0)

    # Draw Matching (M)
    nx.draw_networkx_edges(
        G_combined, pos,
        edgelist=M_edges,
        edge_color="red",
        style="dashed",
        width=2.5,
        alpha=0.7
    )

    # Separate nodes for coloring
    leaves = set(u for u, v in M_edges).union(set(v for u, v in M_edges))
    internal_nodes = set(G.nodes()) - leaves

    nx.draw_networkx_nodes(G, pos, nodelist=list(internal_nodes), node_color="lightgray", node_size=300)
    nx.draw_networkx_nodes(G, pos, nodelist=list(leaves), node_color="lightgreen", node_size=400)

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.title(
        f"Cat State Fusion Network\nTree $T$ (Black) $\\cup$ Matching $M$ (Red Dashed) | $2n = {len(leaves)}$ Leaves",
        fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def is_bad_cat_graph(G: nx.Graph, M_edges: list[tuple]) -> tuple[bool, dict | None]:
    """
    Determines if a graph fails the V >= c criteria (only tree edges can be cut).

    Returns:
        (is_bad, counterexample)
        - is_bad: True if the graph fails the fault-tolerance criteria.
        - counterexample: A dictionary detailing the exact cut and partitions if bad, None otherwise.
    """
    v_list = list(G.nodes())
    T_edges = list(G.edges())
    num_T = len(T_edges)

    vpool = IDPool()

    def x(u):
        return vpool.id(f"x_{u}")

    def nc(u, v):
        return vpool.id(f"nc_{min(u, v)}_{max(u, v)}")

    def m1(u, v):
        return vpool.id(f"m1_{min(u, v)}_{max(u, v)}")

    def m2(u, v):
        return vpool.id(f"m2_{min(u, v)}_{max(u, v)}")

    cnf = CNF()

    # --- 1. Cut Edge Logic (ONLY for Tree Edges) ---
    nc_vars = []
    for u, v in T_edges:
        nc_uv = nc(u, v)
        nc_vars.append(nc_uv)
        cnf.append([-nc_uv, -x(u), x(v)])
        cnf.append([-nc_uv, x(u), -x(v)])

    # --- 2. Matching Edge Logic (NO CUTS ALLOWED) ---
    m1_vars, m2_vars = [], []
    M_set = {(min(u, v), max(u, v)) for u, v in M_edges}

    for u, v in M_set:
        # M edges cannot be cut: endpoints must share the same partition
        cnf.append([-x(u), x(v)])
        cnf.append([x(u), -x(v)])

        m1_uv, m2_uv = m1(u, v), m2(u, v)
        m1_vars.append(m1_uv)
        m2_vars.append(m2_uv)

        cnf.append([-m1_uv, -x(u)])
        cnf.append([m1_uv, x(u)])
        cnf.append([-m2_uv, x(u)])
        cnf.append([m2_uv, -x(u)])

    # --- 3. Cardinality Constraints (V >= c) ---
    cnf.extend(CardEnc.atleast(lits=m1_vars + nc_vars, bound=num_T, vpool=vpool))
    cnf.extend(CardEnc.atleast(lits=m2_vars + nc_vars, bound=num_T, vpool=vpool))

    # --- 4. Symmetry Breaking ---
    if v_list:
        # 1. Force the first node into V1 (False) to halve the search space
        cnf.append([-x(v_list[0])])

        # 2. Force at least one node anywhere in the graph into V2 (True).
        # This prevents the trivial c=0 cut where all nodes are in V1.
        # Since V1 and V2 are now both non-empty, and the graph is connected,
        # at least one tree edge MUST be cut (c >= 1).
        cnf.append([x(u) for u in v_list])

    # --- 5. Solve and Extract Model ---
    with Glucose42(bootstrap_with=cnf) as solver:
        if solver.solve():
            # The solver found a cut that satisfies the failure conditions
            model_lits = set(solver.get_model())

            # Extract Partitions (False -> V1, True -> V2)
            V1 = [u for u in v_list if -x(u) in model_lits]
            V2 = [u for u in v_list if x(u) in model_lits]

            # Extract Cut Tree Edges
            cut_edges = [(u, v) for u, v in T_edges if -nc(u, v) in model_lits]

            # Extract Logical Matching Values
            M1 = [(u, v) for u, v in M_set if m1(u, v) in model_lits]
            M2 = [(u, v) for u, v in M_set if m2(u, v) in model_lits]

            counterexample = {
                "cut_size_c": len(cut_edges),
                "V1_value": len(M1),
                "V2_value": len(M2),
                "cut_edges": cut_edges,
                "V1_nodes": V1,
                "V2_nodes": V2,
                "M1_edges": M1,
                "M2_edges": M2
            }

            return True, counterexample

        return False, None


# ---------------------------------------------------------
# 4. Search Algorithm
# ---------------------------------------------------------
def get_allowed_leaf_graph(T: nx.Graph, leaves: list) -> nx.Graph:
    """
    Builds a complete graph of the leaves, but removes any edges
    between siblings (leaves that share the same parent in T).
    """
    allowed_G = nx.Graph()
    allowed_G.add_nodes_from(leaves)

    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            u, v = leaves[i], leaves[j]

            # Since u and v are leaves, they have exactly 1 neighbor in T
            parent_u = list(T.neighbors(u))[0]
            parent_v = list(T.neighbors(v))[0]

            # Only allow the edge if they do not share a parent
            if parent_u != parent_v:
                allowed_G.add_edge(u, v)

    return allowed_G


def generate_valid_random_matching(allowed_G: nx.Graph) -> list[tuple]:
    """
    Finds a random perfect matching from the allowed edges using
    Edmonds' blossom algorithm with randomized edge weights.
    """
    # Assign random weights to force max_weight_matching to pick a novel perfect matching
    for u, v in allowed_G.edges():
        allowed_G[u][v]['weight'] = random.random()

    # maxcardinality=True ensures we get a perfect matching if one exists
    matching = nx.max_weight_matching(allowed_G, maxcardinality=True)
    return list(matching)


def generate_optimal_spread_matching(T: nx.Graph, leaves: list) -> list[tuple]:
    """
    Deterministically constructs a perfect matching that maximizes the topological
    distance between paired leaves, forcing logical connections to cross the graph.
    """
    # 1. Calculate all-pairs shortest paths in the tree
    path_lengths = dict(nx.all_pairs_shortest_path_length(T))

    G_match = nx.Graph()

    # 2. Build the weighted allowed graph
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            u = leaves[i]
            v = leaves[j]
            dist = path_lengths[u][v]

            # Strictly forbid distance-2 (siblings)
            if dist > 2:
                # Squaring the distance heavily incentivizes the algorithm
                # to pick global crossing edges over local ones.
                weight = dist ** 2
                G_match.add_edge(u, v, weight=weight)

    # 3. Find the deterministic maximum-weight perfect matching
    matching = nx.max_weight_matching(G_match, maxcardinality=True)
    return list(matching)


def single_thread_search_no_siblings(n: int, max_attempts: int = 10000):
    """Executes a single-threaded stochastic search strictly avoiding sibling matchings."""
    G, leaves = generate_balanced_2n_tree(n)

    # Pre-calculate the allowed matching space once
    allowed_G = get_allowed_leaf_graph(G, leaves)

    print(f"Tree generated. Parameter n={n}, Leaves: {len(leaves)}")
    print(f"Total possible leaf edges: {len(leaves) * (len(leaves) - 1) // 2}")
    print(f"Allowed edges (excluding siblings): {len(allowed_G.edges())}")
    print(f"Starting single-threaded search. Max attempts: {max_attempts}")

    for attempt in range(1, max_attempts + 1):
        # 1. Generate a valid random matching from the allowed space
        # M_edges = generate_optimal_spread_matching(G, leaves)
        # M_edges = generate_valid_random_matching(allowed_G)# Replace the 1-factor generator with the new 2-factor generator
        M_edges = generate_valid_random_matching(allowed_G)

        # 2. Run the PySAT Oracle (Ensure you are using the relaxed +2 bound)
        is_bad, _ = is_bad_cat_graph(G, M_edges)

        if not is_bad:
            print(f"\nSUCCESS! Found a strictly valid fault-tolerant graph after {attempt} attempts.")
            return G, M_edges

        # Optional progress tracker
        if attempt % 50 == 0:
            print(f"Checked {attempt} matchings...")

    print(f"\nFAILURE. Checked {max_attempts} matchings. No valid topology found.")
    return G, None


# if __name__ == "__main__":
#     G, M_opt = single_thread_search_no_siblings(n=11, max_attempts=1000)
#     print(M_opt)
#     if M_opt:
#         visualize_cat_graph(G, M_opt)

# Another idea is as follows, it seems that if the graph is 3 regular, n=11 is the limit. Now, change the implementation so that the root node can have an arbitrary degree. Can we go beyond n=

def generate_k_augmented_matching(allowed_G: nx.Graph, k: int) -> list[tuple]:
    """
    Generates a base perfect matching (n edges), and then adds exactly
    k extra edges from a disjoint perfect matching to increase connectivity
    without giving the adversary too many wildcards.
    """
    # 1. Find the base perfect matching (M1)
    for u, v in allowed_G.edges():
        allowed_G[u][v]['weight'] = random.random()
    M1 = nx.max_weight_matching(allowed_G, maxcardinality=True)
    M1_clean = {(min(u, v), max(u, v)) for u, v in M1}

    if k == 0:
        return list(M1_clean)

    # 2. Restrict the graph to find a strictly disjoint matching
    remaining_G = allowed_G.copy()
    remaining_G.remove_edges_from(M1_clean)

    # 3. Find the second perfect matching (M2)
    for u, v in remaining_G.edges():
        remaining_G[u][v]['weight'] = random.random()
    M2 = nx.max_weight_matching(remaining_G, maxcardinality=True)
    M2_clean = list({(min(u, v), max(u, v)) for u, v in M2})

    # 4. Inject exactly k random edges from M2 into M1
    random.shuffle(M2_clean)
    extra_edges = M2_clean[:k]

    return list(M1_clean.union(extra_edges))


def sweep_augmented_matchings(n: int, attempts_per_k: int = 500):
    """
    Sweeps the number of extra edges k from 1 up to n-1 to find the Goldilocks zone.
    """
    G, leaves = generate_balanced_2n_tree(n)
    allowed_G = get_allowed_leaf_graph(G, leaves)

    print(f"Tree generated. Parameter n={n}, Leaves: {len(leaves)}")
    print(f"Sweeping extra boundary edges (k) from 0 to {n}...")

    # Sweep k from 0 (perfect matching) up to n (full 2-factor)
    for k in range(n + 1):
        print(f"\n--- Testing k={k} extra edges (Total |M| = {n + k}) ---")

        for attempt in range(1, attempts_per_k + 1):
            M_proposed = generate_k_augmented_matching(allowed_G, k)

            is_bad, _ = is_bad_cat_graph(G, M_proposed)

            if not is_bad:
                print(f"[SUCCESS] Found a valid fault-tolerant graph at k={k} after {attempt} attempts!")
                return G, M_proposed

            if attempt % 100 == 0:
                print(f"  Checked {attempt} configurations...")

        print(f"[-] No valid graphs found for k={k} after {attempts_per_k} attempts.")

    print(
        "\nFAILURE. Swept all k values. The tree expansion limit is structurally fatal for this adversarial threshold.")
    return G, None


# if __name__ == "__main__":
#     # Test on the n=12 bottleneck
#     G, M_opt = sweep_augmented_matchings(n=14, attempts_per_k=500)
#
#     if M_opt:
#         visualize_cat_graph(G, M_opt)


import networkx as nx
import random
from collections import deque
from pysat.card import CardEnc
from pysat.formula import IDPool, CNF
from pysat.solvers import Glucose42


# ---------------------------------------------------------
# 1. Generalized Arbitrary-Degree Tree Generator
# ---------------------------------------------------------
def generate_d_regular_internal_tree(n: int, max_d: int) -> tuple[nx.Graph, list]:
    """
    Generates a dynamically balanced tree with exactly 2n leaves.
    Internal nodes will expand up to 'max_d' degree.
    """
    if max_d < 3:
        raise ValueError("max_d must be >= 3 to allow any expansion.")

    G = nx.Graph()
    root = "root"
    G.add_node(root)

    leaves = []
    expand_queue = deque()

    target_leaves = 2 * n

    # Phase 1: Expand the root up to max_d
    initial_children = min(max_d, target_leaves)
    for i in range(initial_children):
        child = f"N_{i}"
        G.add_edge(root, child)
        leaves.append(child)
        expand_queue.append(child)

    node_counter = initial_children

    # Phase 2: BFS Expansion for all other internal nodes
    while len(leaves) < target_leaves:
        leaf_to_expand = expand_queue.popleft()
        leaves.remove(leaf_to_expand)

        # Calculate how many children we need to add.
        # Adding 'k' children gives a net gain of 'k-1' leaves (since we popped 1).
        needed_net_gain = target_leaves - len(leaves)

        # We can add up to max_d - 1 children (leaving 1 edge for the parent connection).
        children_to_add = min(max_d - 1, needed_net_gain + 1)

        for _ in range(children_to_add):
            new_leaf = f"N_{node_counter}"
            node_counter += 1
            G.add_edge(leaf_to_expand, new_leaf)
            leaves.append(new_leaf)
            expand_queue.append(new_leaf)

    return G, leaves


# ---------------------------------------------------------
# 2. SAT Oracle & Safe Subspace Filter (Unchanged)
# ---------------------------------------------------------
def get_allowed_leaf_graph(T: nx.Graph, leaves: list) -> nx.Graph:
    allowed_G = nx.Graph()
    allowed_G.add_nodes_from(leaves)
    path_lengths = dict(nx.all_pairs_shortest_path_length(T))

    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            u, v = leaves[i], leaves[j]
            # Sibling constraint seamlessly scales: distance must be > 2
            if path_lengths[u][v] > 2:
                allowed_G.add_edge(u, v)
    return allowed_G


def generate_valid_random_matching(allowed_G: nx.Graph) -> list[tuple]:
    for u, v in allowed_G.edges():
        allowed_G[u][v]['weight'] = random.random()
    matching = nx.max_weight_matching(allowed_G, maxcardinality=True)
    return list(matching)


def is_bad_cat_graph(G: nx.Graph, M_edges: list[tuple]) -> tuple[bool, dict | None]:
    v_list = list(G.nodes())
    T_edges = list(G.edges())
    num_T = len(T_edges)

    vpool = IDPool()

    def x(u):
        return vpool.id(f"x_{u}")

    def nc(u, v):
        return vpool.id(f"nc_{min(u, v)}_{max(u, v)}")

    def m1(u, v):
        return vpool.id(f"m1_{min(u, v)}_{max(u, v)}")

    def m2(u, v):
        return vpool.id(f"m2_{min(u, v)}_{max(u, v)}")

    cnf = CNF()
    nc_vars = []

    for u, v in T_edges:
        nc_uv = nc(u, v)
        nc_vars.append(nc_uv)
        cnf.append([-nc_uv, -x(u), x(v)])
        cnf.append([-nc_uv, x(u), -x(v)])

    m1_vars, m2_vars = [], []
    M_set = {(min(u, v), max(u, v)) for u, v in M_edges}

    for u, v in M_set:
        cnf.append([-x(u), x(v)])
        cnf.append([x(u), -x(v)])

        m1_uv, m2_uv = m1(u, v), m2(u, v)
        m1_vars.append(m1_uv);
        m2_vars.append(m2_uv)

        cnf.append([-m1_uv, -x(u)]);
        cnf.append([m1_uv, x(u)])
        cnf.append([-m2_uv, x(u)]);
        cnf.append([m2_uv, -x(u)])

    cnf.extend(CardEnc.atleast(lits=m1_vars + nc_vars, bound=num_T, vpool=vpool))
    cnf.extend(CardEnc.atleast(lits=m2_vars + nc_vars, bound=num_T, vpool=vpool))

    if v_list:
        cnf.append([-x(v_list[0])])
        cnf.append([x(u) for u in v_list])

    with Glucose42(bootstrap_with=cnf) as solver:
        if solver.solve():
            return True, {"cut_size": sum(1 for u, v in T_edges if -nc(u, v) in set(solver.get_model()))}
        return False, None


# ---------------------------------------------------------
# 3. Execution: Sweeping the Topological Limit
# ---------------------------------------------------------
def sweep_topological_limits():
    """Tests the resilience of internal expansion against the SAT adversary."""
    test_sizes = [12, 13, 14, 16]
    test_degrees = [3, 4, 5]
    max_attempts = 500

    for n in test_sizes:
        print(f"\n{'=' * 40}")
        print(f" Testing Scalability for n={n} ({2 * n} leaves)")
        print(f"{'=' * 40}")

        for d in test_degrees:
            print(f"-> Generating internal topology with max degree = {d}...")
            G, leaves = generate_d_regular_internal_tree(n, max_d=d)
            allowed_G = get_allowed_leaf_graph(G, leaves)

            found_solution = False

            for attempt in range(1, max_attempts + 1):
                M_proposed = generate_valid_random_matching(allowed_G)
                is_bad, _ = is_bad_cat_graph(G, M_proposed)

                if not is_bad:
                    print(f"   [SUCCESS] Fault-Tolerant matching found at degree={d} after {attempt} attempts!")
                    found_solution = True
                    visualize_cat_graph(G, M_proposed)
                    break

            if not found_solution:
                print(f"   [FAILURE] Hit {max_attempts} attempts. degree={d} is mathematically too constrained.")


if __name__ == "__main__":
    sweep_topological_limits()
