import networkx as nx
import numpy as np

class MatchEdgesRecorder:
    def __init__(self, record, G, F, D):
        self.record = record
        self.G = G
        self.F = F
        self.D = D
        self.frames_data = []

    def capture_frame(self, active_graph_edges=None, active_digraph_edges=None):
        if not self.record: return
        
        active_graph_edges = set(active_graph_edges) if active_graph_edges else set()
        active_digraph_edges = set(active_digraph_edges) if active_digraph_edges else set()
        
        self.frames_data.append((
            nx.Graph(self.G), 
            nx.DiGraph(self.D), 
            active_graph_edges, 
            active_digraph_edges
        ))

    def export_gif(self, filename="match_edges.gif"):
        if not self.frames_data:
            return
            
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import imageio
        from spidercat.draw import draw_forest_on_graph_state, display_digraph

        final_G = self.G
        final_D = self.D
        
        layout_pos = nx.spring_layout(final_G, weight="weight")
        if nx.is_directed_acyclic_graph(final_D):
            for layer, nodes in enumerate(nx.topological_generations(final_D)):
                for node in nodes:
                    final_D.nodes[node]["layer"] = layer
            digraph_pos = nx.multipartite_layout(final_D, subset_key="layer", align="vertical")
        else:
            digraph_pos = nx.kamada_kawai_layout(final_D)

        rendered_frames = []
        print(f"Rendering {len(self.frames_data)} frames for match_edges...")

        for G_state, D_state, act_G_edges, act_D_edges in self.frames_data:
            fig_graph, ax_graph = plt.subplots(figsize=(12, 9.6))
            draw_forest_on_graph_state(
                G_state, self.F, 
                pos=layout_pos, 
                processed_edges=set(),
                ax=ax_graph
            )
            
            if act_G_edges:
                nx.draw_networkx_edges(
                    G_state, layout_pos,
                    edgelist=list(act_G_edges),
                    edge_color="magenta",
                    width=3.0,
                    ax=ax_graph
                )

            buf_graph = io.BytesIO()
            fig_graph.savefig(buf_graph, format="png", bbox_inches='tight')
            plt.close(fig_graph)
            buf_graph.seek(0)
            graph_img = Image.open(buf_graph)

            fig_digraph, ax_digraph = plt.subplots(figsize=(12, 9.6))
            display_digraph(D_state, pos=digraph_pos, ax=ax_digraph)
            
            if act_D_edges:
                nx.draw_networkx_edges(
                    D_state, digraph_pos,
                    edgelist=list(act_D_edges),
                    edge_color="magenta",
                    width=3.0,
                    ax=ax_digraph
                )
                
            buf_digraph = io.BytesIO()
            fig_digraph.savefig(buf_digraph, format="png", bbox_inches='tight')
            plt.close(fig_digraph)
            buf_digraph.seek(0)
            digraph_img = Image.open(buf_digraph)

            max_width = graph_img.width + digraph_img.width
            max_height = max(graph_img.height, digraph_img.height)
            
            canvas = Image.new('RGB', (max_width, max_height), (255, 255, 255))
            canvas.paste(graph_img, (0, 0))
            canvas.paste(digraph_img, (graph_img.width, 0))
            rendered_frames.append(np.array(canvas))

        imageio.mimsave(filename, rendered_frames, fps=2)


def match_edges(H: np.ndarray, non_pivots: list[int],
                z_digraphs: list[nx.DiGraph], x_digraphs: list[nx.DiGraph],
                z_candidates: list[list[int]], x_candidates: list[list[int]],
                record=False, global_G=None, global_F=None, global_D=None,
                z_node_mapping=None, x_node_mapping=None) -> list[
    tuple[tuple[int, int], tuple[int, int]]]:
    
    recorder = MatchEdgesRecorder(record, global_G, global_F, global_D)
    recorder.capture_frame()

    edge_list = [
        (i, j)
        for i, r in enumerate(H)
        for j, x in enumerate(r[non_pivots])
        if x == 1
    ]

    tracker = nx.DiGraph()

    for i, D in enumerate(z_digraphs):
        tracker.add_edges_from((f"Z_{i}_{u}", f"Z_{i}_{v}") for u, v in D.edges())
    for j, D in enumerate(x_digraphs):
        tracker.add_edges_from((f"X_{j}_{u}", f"X_{j}_{v}") for u, v in D.edges())

    z_pools = [[c for c in pool] for pool in z_candidates]
    x_pools = [[c for c in pool] for pool in x_candidates]

    def backtrack(remaining_edges, current_matches):
        if not remaining_edges:
            return current_matches

        remaining_edges = sorted(remaining_edges, key=lambda e: len(z_pools[e[0]]) * len(x_pools[e[1]]))
        i, j = remaining_edges[0]

        for z_val in list(z_pools[i]):
            for x_val in list(x_pools[j]):

                new_edges = []
                for u, _ in z_digraphs[i].in_edges(z_val):
                    new_edges.append((f"Z_{i}_{u}", f"X_{j}_{x_val}"))
                for u, _ in x_digraphs[j].in_edges(x_val):
                    new_edges.append((f"X_{j}_{u}", f"Z_{i}_{z_val}"))

                added_edges_this_step = []
                cycle_found = False

                for src, dst in new_edges:
                    if src == dst or nx.has_path(tracker, dst, src):
                        cycle_found = True
                        break
                    tracker.add_edge(src, dst)
                    added_edges_this_step.append((src, dst))
                    
                added_global_edges = []
                if record:
                    u_global = z_node_mapping[(i, z_val)]
                    v_global = x_node_mapping[(j, x_val)]
                    global_G.add_edge(u_global, v_global, edge_type="cnot", weight=0.1)
                    
                    for u, _ in z_digraphs[i].in_edges(z_val):
                        global_D.add_edge(z_node_mapping[(i, u)], x_node_mapping[(j, x_val)], edge_type="cnot")
                        added_global_edges.append((z_node_mapping[(i, u)], x_node_mapping[(j, x_val)]))
                    for u, _ in x_digraphs[j].in_edges(x_val):
                        global_D.add_edge(x_node_mapping[(j, u)], z_node_mapping[(i, z_val)], edge_type="cnot")
                        added_global_edges.append((x_node_mapping[(j, u)], z_node_mapping[(i, z_val)]))
                        
                    recorder.capture_frame(
                        active_graph_edges={(u_global, v_global)},
                        active_digraph_edges=added_global_edges
                    )

                if not cycle_found:
                    z_pools[i].remove(z_val)
                    x_pools[j].remove(x_val)

                    result = backtrack(remaining_edges[1:], current_matches + [((i, j), (z_val, x_val))])
                    if result is not None:
                        return result

                    z_pools[i].append(z_val)
                    x_pools[j].append(x_val)
                    
                    if record:
                        global_G.remove_edge(u_global, v_global)
                        global_D.remove_edges_from(added_global_edges)
                        recorder.capture_frame()

                tracker.remove_edges_from(added_edges_this_step)
                if record and cycle_found:
                    global_G.remove_edge(u_global, v_global)
                    global_D.remove_edges_from(added_global_edges)
                    recorder.capture_frame()

        return None

    final_matching = backtrack(edge_list, [])

    if final_matching is None:
        raise ValueError("No valid cycle-free edge matching exists for this matrix topology.")

    if record:
        recorder.export_gif("match_edges.gif")
        
    return final_matching
