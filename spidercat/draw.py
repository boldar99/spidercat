import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_qubit_lines_state(G, path_cover, markings, matching, pos = None):
    """
    Draws the graph with path cover, markings, and matchings.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 5. Visualization
    # pos = nx.shell_layout(G, nlist=[range(5), range(5,10)])
    pos = pos or nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    # Draw all edges faint
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1)

    # Draw Marked Edges
    marked_edge_list = [(u, v) for (u, v), m in markings.items() if m > 0]
    nx.draw_networkx_edges(G, pos, edgelist=marked_edge_list, edge_color='black', style='solid', width=1.5)

    # Draw Edge Labels (Mark counts)
    edge_labels = {e: "|" * m for e, m in markings.items() if m > 0}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_weight='bold', font_size=20,
        bbox=dict(facecolor='none', edgecolor='none', alpha=0)
    )

    # Draw Nodes & Labels
    nx.draw_networkx_labels(G, pos)

    # Draw Path Edges
    colors = plt.cm.tab10.colors
    for i, path in enumerate(path_cover):
        color = colors[i % len(colors)]
        if len(path) > 1:
            edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[color], width=3, label=f'Path {i}')
        # Draw path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=[color] * len(path), node_size=500)

    # Highlight Matching
    for end_node, neighbor in matching.items():
        # Reconstruct edge for clarification if needed, but we mostly need neighbor position

        # Determine path color for this end node
        path_color = 'green'  # fallback
        for i, path in enumerate(path_cover):
            if end_node in path:
                path_color = colors[i % len(colors)]
                break

        # Calculate Midpoint of the edge (end_node, neighbor)
        # Note: 'neighbor' here is the node across the marked edge
        pos_u = np.array(pos[end_node])
        pos_v = np.array(pos[neighbor])
        midpoint = (pos_u + pos_v) / 2
        pos_end = np.array(pos[end_node])

        # Draw line from End Node to Midpoint
        plt.plot([pos_end[0], midpoint[0]], [pos_end[1], midpoint[1]],
                 color=path_color, linewidth=4, linestyle='-')

    plt.axis('off')
    plt.savefig("qubit_lines_state.png")
    plt.show()
    plt.close()


def draw_path_cover(ax, G_base, pos, cover_paths, markings=None, matching=None, node_size=200, label_font_size=8):
    # Base faint background
    edges = list(G_base.edges())
    nx.draw_networkx_nodes(G_base, pos, node_color="#ecf0f1", node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G_base, pos, font_size=label_font_size, ax=ax)
    nx.draw_networkx_edges(G_base, pos, edgelist=edges, edge_color="gray", alpha=0.2, ax=ax)

    # Draw each path with a distinct color
    colors = plt.cm.tab10.colors
    for idx, path in enumerate(cover_paths):
        color = colors[idx % len(colors)]
        if len(path) >= 2:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G_base, pos, edgelist=path_edges, edge_color=[color], width=3, ax=ax)
        nx.draw_networkx_nodes(G_base, pos, nodelist=path, node_color=[color], node_size=int(node_size * 1.2), ax=ax)

    # Overlay tick marks from `markings` (if provided)
    if markings:
        edge_labels = {}
        for u, v in G_base.edges():
            num_marks = markings.get((u, v), markings.get((v, u), 0))
            if num_marks:
                edge_labels[(u, v)] = "  |  " * num_marks
        if edge_labels:
            nx.draw_networkx_edge_labels(G_base, pos, edge_labels=edge_labels, font_size=14, font_weight="bold",
                                         bbox=dict(alpha=0), ax=ax)

    # Draw Matching Lines (Half-edges to midpoints)
    if matching:
        for end_node, neighbor in matching.items():
            # Determine path color for this end node
            path_color = 'black'  # fallback
            for i, path in enumerate(cover_paths):
                if end_node in path:
                    path_color = colors[i % len(colors)]
                    break

            # Calculate Midpoint
            pos_u = np.array(pos[end_node])
            pos_v = np.array(pos[neighbor])
            midpoint = (pos_u + pos_v) / 2
            pos_end = np.array(pos[end_node])

            # Draw solid line from end_node to midpoint
            ax.plot([pos_end[0], midpoint[0]], [pos_end[1], midpoint[1]],
                    color=path_color, linewidth=4, linestyle='-')


def visualize_cat_state_base(G, ham_path, markings, pos=None):
    plt.figure(figsize=(5, 5))
    pos = pos or nx.spring_layout(G)  # Kamada-Kawai usually looks best for regular graphs
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: "  |  " * num_marks for e, num_marks in markings.items()},
                                 font_size=18, font_weight='bold', bbox=dict(alpha=0))
    nx.draw_networkx_edges(
        G, pos=pos,
        edgelist=ham_path,
        edge_color='red', width=1.5
    )
    plt.show()

def draw_spanning_forest_solution(
        G: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        matches: dict[int, list[tuple[int, int]]] | None = None,
        roots: dict[int, int] | None = None,
        figsize=(14, 10),
        pos = None,
):
    """
    Visualizes the graph state, highlighting the spanning forest,
    drawing connections to marked edges, and circling tree roots
    with the color corresponding to their tree.
    """
    # 1. Setup Layout
    pos = pos or nx.spring_layout(G, seed=42)
    plt.figure(figsize=figsize)

    # 2. Draw Background (All edges faint)
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1, alpha=0.5)

    # 3. Draw Markings (Black, dashed lines)
    marked_edge_list = [(u, v) for (u, v), m in markings.items() if m > 0]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=marked_edge_list,
        edge_color='black',
        style='dashed',
        width=0.1,
        alpha=0.6
    )

    bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.1)

    for (u, v), m in markings.items():
        if m == 0:
            continue

        pos_u = np.array(pos[u])
        pos_v = np.array(pos[v])

        # Calculate angle to rotate the "|" so it aligns with the edge
        angle_rad = np.arctan2(pos_v[1] - pos_u[1], pos_v[0] - pos_u[0])
        angle_deg = np.degrees(angle_rad)

        if m == 1:
            # Single mark at 1/2
            mid = (pos_u + pos_v) / 2.0
            plt.text(mid[0], mid[1], "|", ha='center', va='center',
                     rotation=angle_deg, fontweight='bold', fontsize=15,
                     bbox=bbox_props, zorder=5)
        elif m >= 2:
            # Double marks at 1/3 and 2/3
            p1 = pos_u + (pos_v - pos_u) / 3.0
            p2 = pos_u + 2.0 * (pos_v - pos_u) / 3.0

            plt.text(p1[0], p1[1], "|", ha='center', va='center',
                     rotation=angle_deg, fontweight='bold', fontsize=15,
                     bbox=bbox_props, zorder=5)
            plt.text(p2[0], p2[1], "|", ha='center', va='center',
                     rotation=angle_deg, fontweight='bold', fontsize=15,
                     bbox=bbox_props, zorder=5)

    # 4. Draw Forest Trees
    cmap = plt.cm.tab10
    colors = cmap.colors
    node_color_map = {}

    for i, component in enumerate(nx.connected_components(forest)):
        color = colors[i % len(colors)]
        color = "#30A08E"
        color = "#69D3BE"
        tree = forest.subgraph(component)

        # Draw Tree Edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist=tree.edges(),
            edge_color=[color],
            width=3.5,
            alpha=0.6
        )

        # Draw Tree Nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(component),
            node_color=[color] * len(component),
            node_size=600,
            edgecolors='black'
        )

        # Store color for later lookups (roots & matches)
        for node in component:
            node_color_map[node] = color

    # --- NEW: Draw Root Highlights (Color Matched) ---
    if roots:
        roots_l = roots.values() if isinstance(roots, dict) else roots
        for root_node in roots_l:
            # Lookup the color of this specific root
            root_color = node_color_map.get(root_node, 'black')

            # Draw distinct circle for this root
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[root_node],
                node_size=1200,
                node_color='none',  # Transparent inside
                edgecolors=[root_color],  # Border matches tree color
                linewidths=3
            )

    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')

    # 5. Draw Matches
    from collections import Counter  # Ensure this is imported at the top of your file

    matches = matches if matches is not None else {}

    for node, assigned_edges in matches.items():
        node_color = node_color_map.get(node, 'gray')
        start_pos = np.array(pos[node])

        # Group assignments to see if a node claims an edge 1 or 2 times
        # Sort the tuple so (6,7) and (7,6) are counted as the same edge
        edge_counts = Counter([tuple(sorted(e)) for e in assigned_edges])

        for norm_edge, count in edge_counts.items():
            # Fallback to (v, u) if (u, v) isn't in markings
            lookup_edge = norm_edge if norm_edge in markings else (norm_edge[1], norm_edge[0])
            capacity = markings.get(lookup_edge, 1)

            # Identify the target node at the other end of the edge
            other_node = norm_edge[1] if node == norm_edge[0] else norm_edge[0]
            pos_u = np.array(pos[node])
            pos_v = np.array(pos[other_node])

            # Determine Target based on Capacity AND Count
            if capacity <= 1:
                # Standard 1/2 logic for single capacity
                target = (pos_u + pos_v) / 2.0
            else:
                if count == 1:
                    # Node claims 1 out of 2 marks -> Go 1/3 of the way
                    target = pos_u + (pos_v - pos_u) / 3.0
                elif count >= 2:
                    # Node claims 2 out of 2 marks -> Go 2/3 of the way
                    target = pos_u + 2.0 * (pos_v - pos_u) / 3.0

            plt.plot(
                [start_pos[0], target[0]],
                [start_pos[1], target[1]],
                color=node_color,
                linewidth=3.5,
                alpha=0.6,
                linestyle='-',
                zorder=1  # FIX: Forces the line to draw strictly underneath the nodes
            )

    plt.title("Spanning Forest with Marked Assignments & Roots")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def draw_forest_on_graph(
        G: nx.Graph,
        F: nx.Graph,
        figsize: tuple[int, int] = (10, 8)
) -> None:
    """
    Draws the spanning forest F over the graph G.
    Differentiates original nodes, marked nodes, and flagged nodes by color.
    """
    plt.figure(figsize=figsize)

    cmap = plt.cm.tab10
    colors = cmap.colors

    # 1. Compute a single, locked layout based on G
    # A fixed seed ensures the graph looks the same every time you run it
    pos = nx.spring_layout(G)

    # 2. Map node colors based on the metadata we injected earlier
    node_colors = []
    for node, data in G.nodes(data=True):
        if data.get("is_flag"):
            node_colors.append(colors[1])    # Implicit Flags (edge_diff)
        elif data.get("is_mark"):
            node_colors.append(colors[2])       # Explicit Marks
        else:
            node_colors.append(colors[0]) # Original Forest/Graph Nodes
    node_edge_colors = []
    for node, data in G.nodes(data=True):
        spider_type = data.get("spider_type")
        if spider_type == "X":
            node_edge_colors.append("red")
        elif spider_type == "Z":
            node_edge_colors.append("green")
        else:
            node_edge_colors.append("black")


    # 3. Draw the Background: Graph G
    # Draw the nodes first with our computed colors
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=400,
        edgecolors=node_edge_colors,
        linewidths=3,
    )

    # Draw G's edges faintly in the background
    nx.draw_networkx_edges(
        G, pos,
        edge_color="gray",
        width=1.5,
        alpha=0.8,
        style="dashed" # Helps distinguish from F
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'cnot'],
        edge_color="orange",
        width=3.0,
        alpha=0.5,
    )

    # 4. Draw the Foreground: Forest F
    # Draw F's edges thickly and prominently
    nx.draw_networkx_edges(
        F, pos,
        edge_color=(colors[0], ),
        width=3.0,
        alpha=0.8
    )

    # 5. Add Labels
    # For a cleaner look, you might only want to label original nodes,
    # but here we label everything to help you debug.
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight="bold"
    )

    plt.title("Spanning Forest F (Solid/Black) on Graph G (Dashed/Gray)")
    plt.axis("off") # Hide the bounding box
    plt.tight_layout()
    plt.show()


def display_digraph(di_graph: nx.DiGraph, figsize=(12, 12)):
    """
    Displays the directed graph, distinguishing between tree edges and cycle closures.
    """

    if nx.is_directed_acyclic_graph(di_graph):
        plt.figure(figsize=figsize)
        # 1. Calculate the hierarchical layers mathematically
        for layer, nodes in enumerate(nx.topological_generations(di_graph)):
            for node in nodes:
                # 2. Explicitly assign the layer attribute to each node
                di_graph.nodes[node]["layer"] = layer

        # 3. Use the newly created "layer" attribute for the layout
        # align="horizontal" makes it a top-down tree.
        # (Remove align="horizontal" if you prefer left-to-right)
        pos = nx.multipartite_layout(di_graph, subset_key="layer", align="vertical")
    else:
        plt.figure(figsize=(10, 8))
        # Kamada-Kawai handles graphs with cycles by treating edges like springs
        pos = nx.kamada_kawai_layout(di_graph)

    cmap = plt.cm.tab10
    colors = cmap.colors
    node_colors = []
    for node, data in di_graph.nodes(data=True):
        if data.get("is_flag"):
            node_colors.append(colors[1])    # Implicit Flags (edge_diff)
        elif data.get("is_mark"):
            node_colors.append(colors[2])       # Explicit Marks
        else:
            node_colors.append(colors[0]) # Original Forest/Graph Nodes

    nx.draw_networkx_nodes(
        di_graph, pos,
        node_color=node_colors,
        node_size=400,
        edgecolors="black" # Gives nodes a clean border
    )
    nx.draw_networkx_labels(di_graph, pos, font_size=10, font_weight='bold')

    # Filter edges by type
    tree_edges = [(u, v) for u, v, d in di_graph.edges(data=True) if d.get('edge_type') == 'tree']
    missing_edges = [(u, v) for u, v, d in di_graph.edges(data=True) if d.get('edge_type') == 'missing_link']
    cnot_edges = [(u, v) for u, v, d in di_graph.edges(data=True) if d.get('edge_type') == 'cnot']

    # Draw tree edges (Solid Black)
    nx.draw_networkx_edges(di_graph, pos, edgelist=tree_edges, edge_color='black', arrows=True, arrowsize=15)

    # Draw cycle closure edges (Dashed Red, l -> t)
    nx.draw_networkx_edges(di_graph, pos, edgelist=missing_edges, edge_color='red', style='dashed', arrows=True, arrowsize=15)
    nx.draw_networkx_edges(di_graph, pos, edgelist=cnot_edges, edge_color='orange', style='dashed', arrows=True, arrowsize=15)

    plt.title("Spanning Tree Traversal with Directed Cycle Closures")
    plt.axis('off')
    plt.show()

