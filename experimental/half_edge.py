"""
Script to split edges in graphs (including ZX-diagrams) by inserting
hidden vertices, with a half-edge colouring on the directed edges.
"""

from dataclasses import dataclass, field
import random
import networkx as nx
from typing import Dict, Tuple, Optional

try:
    import pyzx as zx
    from pyzx import VertexType
    from pyzx.rewrite_rules import recursive_unfuse_FE
except ModuleNotFoundError:
    zx = None
    VertexType = None
    recursive_unfuse_FE = None


def is_hidden(node) -> bool:
    """Return True if a node is a hidden vertex (created by splitting)."""
    return str(node).startswith("hidden_")


def assign_half_edge_coloring(
    G: nx.Graph,
    attr: str = "half_edge_color",
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Take an undirected graph of maximum degree <= len(colors) and return a
    directed graph where each undirected edge {u, v} is replaced by two
    directed edges (u, v) and (v, u).

    For each vertex v, all outgoing directed edges (v, u) get distinct colors
    chosen from `colors`, stored as edge attribute `attr`.
    """
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **data)
        H.add_edge(v, u, **data)

    rng = random.Random(seed) if seed is not None else random
    for v in H.nodes():
        out_edges = list(H.out_edges(v))
        rng.shuffle(out_edges)
        colors = ("green", "red", "blue")
        if len(out_edges) == 1:
            colors = ("green",) if rng.random() < 0.5 else ("red",)

        if len(out_edges) > len(colors):
            raise ValueError(
                f"Vertex {v} has {len(out_edges)} outgoing edges, "
                f"but only {len(colors)} colors are available."
            )
        for (src, dst), color in zip(out_edges, colors):
            H[src][dst][attr] = color
    return H


def split_directed_edges(
    G: nx.Graph,
    pos: Optional[Dict] = None,
    color_seed: Optional[int] = None,
) -> Tuple[nx.DiGraph, Dict]:
    """Split each directed edge in a (colored) graph by adding a hidden
    vertex at the midpoint."""
    H = (G if isinstance(G, nx.DiGraph)
         else assign_half_edge_coloring(G, seed=color_seed))
    new_G = nx.DiGraph()
    new_G.add_nodes_from(H.nodes(data=True))
    new_pos = pos.copy()
    edge_to_hidden = {}
    hidden_counter = 0

    for u, v in H.edges():
        if (u, v) not in edge_to_hidden:
            hidden_name = f"hidden_{hidden_counter}"
            hidden_counter += 1
            new_pos[hidden_name] = (
                (pos[u][0] + pos[v][0]) / 2,
                (pos[u][1] + pos[v][1]) / 2
            )
            edge_to_hidden[(u, v)] = hidden_name
            new_G.add_node(hidden_name)
        hidden = edge_to_hidden[(u, v)]
        edge_attrs = H[u][v].copy()
        new_G.add_edge(u, hidden, **edge_attrs)
        new_G.add_edge(hidden, v, **edge_attrs)
    return new_G, new_pos


def generate_demo_graph(
    N: int,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]], Dict[int, str]]:
    """
    Generate an unrooted binary tree with N leaves.

    Every internal node has degree 3, which matches the assumptions made by
    the extraction logic below. This keeps the script runnable without the
    previously imported rewrite helpers.
    """
    if N < 3:
        raise ValueError("N must be at least 3")

    rng = random.Random(seed)
    G = nx.Graph()

    next_id = 0

    def new_node(**attrs):
        nonlocal next_id
        node = next_id
        next_id += 1
        G.add_node(node, **attrs)
        return node

    center = new_node(basis="Z" if rng.random() < 0.5 else "X")
    leaves = [new_node(), new_node(), new_node()]
    for leaf in leaves:
        G.add_edge(center, leaf)

    while len(leaves) < N:
        leaf = rng.choice(leaves)
        neighbor = next(G.neighbors(leaf))
        G.remove_edge(leaf, neighbor)

        internal = new_node(basis="Z" if rng.random() < 0.5 else "X")
        fresh_leaf = new_node()
        G.add_edge(neighbor, internal)
        G.add_edge(internal, leaf)
        G.add_edge(internal, fresh_leaf)

        leaves.remove(leaf)
        leaves.extend([leaf, fresh_leaf])

    pos = nx.spring_layout(G, seed=seed)
    pos = {node: tuple(coords) for node, coords in pos.items()}
    node_types = {
        node: ("boundary" if G.degree(node) == 1 else "spider")
        for node in G.nodes()
    }
    return G, pos, node_types


def generate_zx_graph(N: int, W: int):
    """Generates a ZX-diagram graph for the given N and W parameters."""
    if zx is None or VertexType is None or recursive_unfuse_FE is None:
        raise ImportError(
            "pyzx is not installed. Install it with `python3 -m pip install pyzx` "
            "to build a pyzx graph."
        )
    g = zx.Graph()
    v = g.add_vertex(VertexType.Z, (N - 1) / 2, 0)
    for i in range(N):
        g.add_edge((v, g.add_vertex(VertexType.BOUNDARY, i, 1)))
    return g if recursive_unfuse_FE(g, v, w=W) else None


def zx_diagram_to_networkx_graph(graph):
    graph_dict = graph.to_dict()
    G = nx.Graph()
    pos = {}
    node_types = {}
    for v_data in graph_dict["vertices"]:
        G.add_node(v_data["id"])
        pos[v_data["id"]] = tuple(v_data["pos"])
        node_types[v_data["id"]] = v_data["t"]
    for u, v, _ in graph_dict["edges"]:
        G.add_edge(u, v)
    # Assign basis="Z" only to nodes with degree > 1
    for node in G.nodes():
        if G.degree(node) == 3:
            G.nodes[node]["basis"] = "Z" if random.random() < 0.5 else "X"
    return G, pos, node_types


def visualize_split_graph(G: nx.DiGraph, pos: Dict):
    """Visualize a directed graph with split edges."""
    import matplotlib.pyplot as plt

    # Define ZX colors
    zx_red = (232/255, 165/255, 165/255)
    zx_green = (216/255, 248/255, 216/255)

    plt.figure(figsize=(12, 8))
    original_nodes = [n for n in G.nodes() if not is_hidden(n)]
    hidden_nodes = [n for n in G.nodes() if is_hidden(n)]
    node_sizes = [500 if n in original_nodes else 0 for n in G.nodes()]

    # Separate nodes by basis
    z_nodes = [n for n in original_nodes if G.nodes[n].get("basis") == "Z"]
    x_nodes = [n for n in original_nodes if G.nodes[n].get("basis") == "X"]
    other_nodes = [
        n for n in original_nodes if n not in z_nodes and n not in x_nodes
    ]
    print(z_nodes, x_nodes, other_nodes)

    red_edges, green_edges, blue_edges = [], [], []
    for u, v, data in G.edges(data=True):
        if u in original_nodes and v in hidden_nodes:
            color = data.get("half_edge_color")
            if color == "blue":
                blue_edges.append((u, v))
            elif color == "red":
                red_edges.append((u, v))
            elif color == "green":
                green_edges.append((v, u))

    base_kwargs = {"width": 2.0, "alpha": 0.8, "node_size": node_sizes}
    nx.draw_networkx_edges(
        G, pos, edgelist=red_edges, edge_color="red",
        arrows=True, arrowsize=20, arrowstyle="->", **base_kwargs
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=green_edges, edge_color="green",
        arrows=True, arrowsize=20, arrowstyle="->", **base_kwargs
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=blue_edges, edge_color="blue",
        arrows=False, **base_kwargs
    )
    # Draw Z basis nodes in green
    nx.draw_networkx_nodes(
        G, pos, nodelist=z_nodes, node_color="green",
        node_size=500, alpha=1
    )
    # Draw X basis nodes in red
    nx.draw_networkx_nodes(
        G, pos, nodelist=x_nodes, node_color="red",
        node_size=500, alpha=1
    )
    # Draw nodes without basis attribute (fallback)
    nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes, node_color='lightblue',
        node_size=500, alpha=1
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=hidden_nodes, node_size=0, alpha=0.0
    )
    nx.draw_networkx_labels(
        G, pos, {n: str(n) for n in original_nodes},
        font_size=10, font_weight='bold'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def print_random_degree_ordering(
    G: nx.Graph, seed: Optional[int] = None
) -> list:
    """Return a random ordering of the vertices, sorted by degree."""
    rng = random.Random(seed) if seed is not None else random
    nodes = [n for n in G.nodes() if not is_hidden(n)]
    ordering = sorted(nodes, key=lambda n: (-G.degree(n), rng.random()))
    print("Vertex ordering by degree (descending), ties randomized:")
    print(ordering)
    return ordering


@dataclass
class CircuitBuilder:
    n_inputs: int
    qubit_alive: list[bool] = field(default=None)
    gates: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def __post_init__(self):
        self.qubit_alive = self.qubit_alive or [True] * self.n_inputs

    def add_qubit(self):
        self.qubit_alive.append(True)
        return len(self.qubit_alive) - 1

    def end_qubit(self, qubit: int):
        self.append_gate('Measure', [qubit])
        self.qubit_alive[qubit] = False

    def append_gate(self, op_type: str, qubits: list[int]):
        if any(not self.qubit_alive[q] for q in qubits):
            raise ValueError(
                f"Gate {op_type} applied to dead qubits: {qubits}"
            )

        self.gates.append((op_type, tuple(qubits)))

    def add_bell_state(self):
        qubit1 = self.add_qubit()
        qubit2 = self.add_qubit()
        self.append_gate('H', [qubit1])
        self.append_gate('CX', [qubit1, qubit2])
        return qubit1, qubit2

    def end_with_bell(self, qubit1: int, qubit2: int):
        self.append_gate('CX', [qubit1, qubit2])
        self.append_gate('H', [qubit1])
        self.end_qubit(qubit1)
        self.end_qubit(qubit2)

    def to_pyzx(self):
        """Convert the CircuitBuilder to a pyzx Circuit."""
        if zx is None:
            raise ImportError(
                "pyzx is not installed. Install it with `python3 -m pip install pyzx` "
                "to convert the extracted circuit to a pyzx Circuit."
            )
        circ = zx.Circuit(self.n_inputs)
        for i in range(self.n_inputs, len(self.qubit_alive)):
            circ.add_gate("InitAncilla", label=i)
            circ.add_gate("H", i)
        gate_map = {'H': 'H', 'CX': 'CNOT'}
        for op_type, qubits in self.gates:
            if op_type in gate_map:
                circ.add_gate(gate_map[op_type], *qubits)
            elif op_type == 'Measure':
                circ.add_gate('H', *qubits)
                circ.add_gate('PostSelect', *qubits)
            else:
                raise ValueError(f"Unknown gate type: {op_type}")
        return circ


def extract_circuit(graph: nx.DiGraph, ordering: list):
    """Extract a circuit from a directed graph. Assume that graph already
    has a half-edge coloring."""
    qubit_lines = {}
    v2q = {}
    v1s = [v for v in ordering if graph.degree(v) == 2]
    v3s = [v for v in ordering if graph.degree(v) == 6]
    if len(v1s) + len(v3s) != len(ordering):
        raise ValueError(
            "Invalid ordering: not all vertices are 1-degree or 3-degree"
        )

    n_inputs = 0
    for v in v1s:
        nv = list(graph.neighbors(v))[0]
        colour = graph[v][nv].get("half_edge_color")
        if colour == 'blue':
            raise ValueError(
                f"Invalid colouring: blue edge found in 1-degree vertex {v}"
            )
        if colour == 'red':
            qubit_lines[(v, nv)] = n_inputs
            n_inputs += 1
    builder = CircuitBuilder(n_inputs=n_inputs)

    for i, v in enumerate(v3s):
        nvs = {color: None for color in ['green', 'blue', 'red']}
        for n in graph.neighbors(v):
            color = graph[v][n].get("half_edge_color")
            if color in nvs:
                nvs[color] = n

        if (nvs['green'], v) in qubit_lines:
            main_qubit = qubit_lines[(nvs['green'], v)]
        else:
            g_qubit, main_qubit = builder.add_bell_state()
            qubit_lines[(v, nvs['green'])] = g_qubit

        # TODO this bit needs to be changed for general phase-free diagrams
        basis1 = graph.nodes[nvs['blue']].get("basis")
        basis2 = graph.nodes[v].get("basis")
        # assert basis1 in {'Z', 'X'}, basis1
        # assert basis2 in {'Z', 'X'}, basis2
        follows = v3s[i - 1] == nvs['blue']
        both_blue = graph[nvs['blue']][v].get("half_edge_color") == 'blue' and graph[v][nvs['blue']].get("half_edge_color") == 'blue'
        direct_cnot = follows and {basis1, basis2} == {'Z', 'X'} and both_blue
        # print(f'follows: {follows}, direct_cnot: {direct_cnot}, basis1: {basis1}, basis2: {basis2}')
        # print(nvs['blue'], v)
        if (nvs['blue'], v) in qubit_lines:
            if direct_cnot:
                qbs = [main_qubit, qubit_lines[(nvs['blue'], v)]] if basis2 == 'Z' else [qubit_lines[(nvs['blue'], v)], main_qubit]
                builder.append_gate('CX', qbs)
            else:
                b_qubit = qubit_lines[(nvs['blue'], v)]
                # this line needs fixing bc of none
                qbs = [main_qubit, b_qubit] if basis2 == 'Z' else [b_qubit, main_qubit]
                builder.append_gate('CX', qbs)
                if basis2 == 'X':
                    builder.append_gate('H', [b_qubit])
                builder.end_qubit(b_qubit)
        else:
            if not direct_cnot:
                b_qubit = builder.add_qubit()
                # add hadamard if necessary
                if basis2 == 'X':
                    builder.append_gate('H', [b_qubit])
                qbs = [main_qubit, b_qubit] if basis2 == 'Z' else [b_qubit, main_qubit]
                builder.append_gate('CX', qbs)
                qubit_lines[(v, nvs['blue'])] = b_qubit
            else:
                qubit_lines[(v, nvs['blue'])] = main_qubit

        if (nvs['red'], v) in qubit_lines:
            r_qubit = qubit_lines[(nvs['red'], v)]
            builder.end_with_bell(main_qubit, r_qubit)
        else:
            qubit_lines[(v, nvs['red'])] = main_qubit
        v2q[v] = main_qubit

    final_mapping = {}
    for v in v1s:
        nv = list(graph.neighbors(v))[0]
        if graph[v][nv].get("half_edge_color") == 'green':
            final_mapping[v] = qubit_lines[(nv, v)]

    print("Vertex to main qubit mapping:", v2q)
    assert sum(map(int, builder.qubit_alive)) == len(final_mapping), f"{sum(map(int, builder.qubit_alive))} != {len(final_mapping)}"
    return builder, final_mapping


def main():
    """Example usage of the split_directed_edges function on a ZX-diagram."""
    zx_graph = generate_zx_graph(8, None)
    if zx_graph is None:
        raise RuntimeError("Failed to generate ZX graph")

    G, pos, _ = zx_diagram_to_networkx_graph(zx_graph)
    ordering = print_random_degree_ordering(G, seed=42)
    colored_G = assign_half_edge_coloring(G, seed=42)
    builder, final_mapping = extract_circuit(colored_G, ordering)
    print("\nExtracted circuit:")
    print(f"  Total qubits: {len(builder.qubit_alive)}")
    print(f"  Total gates: {len(builder.gates)}")
    print(f"  Final mapping: {final_mapping}")
    new_G, new_pos = split_directed_edges(G, pos, color_seed=42)
    visualize_split_graph(new_G, new_pos)


if __name__ == "__main__":
    main()
