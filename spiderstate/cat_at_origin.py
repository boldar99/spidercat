import networkx as nx
import numpy as np
import stim
from tqdm import tqdm

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder
from spidercat.draw import draw_forest_on_graph, display_digraph
from spiderstate.between_shor_and_steane import measure_stabilizers_scheme_B, measure_stabilizers_scheme_A
from spiderstate.circuit_finder import find_circuit
from spiderstate.spider_leg_matcher import match_edges
from spiderstate.utils import find_pivots_in_matrix, load_qecc, count_operations, flatten
from spiderstate.well_ordered_cat_state import well_ordered_ft_cat_state_data
from spiderstate.optimize_parity_matrix import has_unique_ones_property, optimize_fault_tolerant_matrix, \
    row_optimize_matrix, minimum_number_of_flags, cnot_cost
from spidercat.syndrome_measurement import fao_se_circuit, bare_se_circuit
from spiderstate.verification import find_lookahead_verification_stabilizers, compute_unitary_fault_set_1, compute_bare_injected_faults
from typing import Literal
from spiderstate.circuit_merger import synthesize_and_merge_layer



def col_reduced_cat_at_origin(H: np.ndarray, d: int, max_col_ops: int = 0, max_basis_tries: int = 5000, record: bool = False):
    t = (d - 1) // 2
    _, final_matrix_after_col_ops, col_ops_performed = optimize_fault_tolerant_matrix(H, t, max_col_ops, max_basis_tries)
    circ = cat_at_origin(final_matrix_after_col_ops, d, record=record)
    for (c, n) in col_ops_performed:
        circ.append("CX", [c, n])

    return circ


def row_optimized_cat_at_origin(H: np.ndarray, d: int, max_basis_tries: int = 10_000, record: bool = False):
    t = (d - 1) // 2
    best_row_op_cost, matrix_after_row_ops = row_optimize_matrix(H, t, max_basis_tries)
    return cat_at_origin(matrix_after_row_ops, d, record=record)


def cat_at_origin(H: np.ndarray, d: int, draw_solutions=False, record=False) -> stim.Circuit:
    if not has_unique_ones_property(H):
        raise ValueError(f"H is not representing a bipartite graph state.")

    N = H.shape[1]
    t = d // 2

    pivots, rows_without_pivots = find_pivots_in_matrix(H)
    pivots_perm = [row for row, col in sorted(pivots.items(), key=lambda item: item[1])]
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(H, axis=1)
    x_spiders = np.sum(H[:, non_pivots], axis=0) + 1

    z_data = [well_ordered_ft_cat_state_data(zs, t) for zs in z_spiders]
    x_data = [well_ordered_ft_cat_state_data(xs, t) for xs in x_spiders]
    z_graphs, x_graphs, z_trees, x_trees, z_mains, x_mains = [], [], [], [], [], []
    z_digraphs, x_digraphs = [], []
    z_candidates, x_candidates = [], []
    z_roots, x_roots = [], []
    for (G, F, roots, D, e) in z_data:
        nx.set_node_attributes(G, "Z", 'spider_type')
        z_graphs.append(G);
        z_trees.append(F);
        z_roots.append(roots)
        z_digraphs.append(D);
        z_mains.append(e)

        # Flatten topological generations into prioritized 1D candidate pools
        cands = []
        for layer in nx.topological_generations(D):
            cands.extend([l for l in layer if l != e and G.nodes[l].get("is_mark", False)])
        z_candidates.append(cands)

    for (G, F, roots, D, e) in x_data:
        nx.set_node_attributes(G, "X", 'spider_type')
        x_graphs.append(G);
        x_trees.append(F);
        x_roots.append(roots)
        x_digraphs.append(D);
        x_mains.append(e)

        cands = []
        for layer in nx.topological_generations(D):
            cands.extend([l for l in layer if l != e and G.nodes[l].get("is_mark", False)])
        x_candidates.append(cands)

    matched_edges = match_edges(H, non_pivots, z_digraphs, x_digraphs, z_candidates, x_candidates)

    # Build global graphs
    z_node_mapping: dict[tuple[int, int], int] = {}
    x_node_mapping: dict[tuple[int, int], int] = {}
    global_G = nx.Graph()
    global_F = nx.Graph()
    global_roots = {}
    global_D = nx.DiGraph()
    global_primary_paths = {}
    i, j = 0, 0
    k = 0
    non_pivots_set = set(non_pivots)

    # Phase 1: Adding each individual cat state to the global graphs
    while i + j < N:
        curr_col = i + j
        is_non_pivot = curr_col in non_pivots_set

        if is_non_pivot:
            graph, trees, digraph = x_graphs[i], x_trees[i], x_digraphs[i]
            node_mapping, root, index = x_node_mapping, x_roots[i], i
        else:
            row = pivots_perm[j]
            graph, trees, digraph = z_graphs[row], z_trees[row], z_digraphs[row]
            node_mapping, root, index = z_node_mapping, z_roots[row], row
        for node, data in graph.nodes(data=True):
            node_mapping[(index, node)] = k
            global_G.add_node(k, **data)
            global_F.add_node(k)
            global_D.add_node(k)
            k += 1
        for u, v, data in graph.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_G.add_edge(u_prime, v_prime, **data)
        for u, v, data in trees.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_F.add_edge(u_prime, v_prime, **data)
        for u, v, data in digraph.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_D.add_edge(u_prime, v_prime, **data)

        global_roots[i + j] = node_mapping[(index, root[0])]
        global_primary_paths[i + j] = nx.shortest_path(
            global_F,
            source=global_roots[i + j],
            target=node_mapping[(index, x_mains[i] if i + j in non_pivots else z_mains[pivots_perm[j]])]
        )
        if i + j in non_pivots:
            i += 1
        else:
            j += 1

    # Phase 1: Connecting the cat states in the global graphs
    while matched_edges:
        (z_graph, x_graph), (z_val, x_val) = matched_edges.pop(0)
        global_G.add_edge(z_node_mapping[(z_graph, z_val)], x_node_mapping[(x_graph, x_val)], edge_type="cnot")
        global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_mark"] = False
        global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_mark"] = False

        if global_F.degree(z_node_mapping[(z_graph, z_val)]) == 1:
            global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_flag"] = True
        if global_F.degree(x_node_mapping[(x_graph, x_val)]) == 1:
            global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_flag"] = True

        for u, _ in z_digraphs[z_graph].in_edges(z_val):
            global_D.add_edge(z_node_mapping[(z_graph, u)], x_node_mapping[(x_graph, x_val)], edge_type="cnot")
        for u, _ in x_digraphs[x_graph].in_edges(x_val):
            global_D.add_edge(x_node_mapping[(x_graph, u)], z_node_mapping[(z_graph, z_val)], edge_type="cnot")

    # Extract circuit using the global graphs
    extractor = CatStateExtractor(StimBuilder(), verbose=False, record=record)
    if draw_solutions:
        draw_forest_on_graph(global_G, global_F, figsize=(8, 8))
        display_digraph(global_D, figsize=(8, 8))
    circ = extractor.extract(global_G, global_F, global_roots, global_D, global_primary_paths)
    if record:
        print("Exporting...")
        extractor.export_gif("cat_at_origin.gif")
    return circ


def _print_violations(violations: list[dict], name: str):
    if not violations:
        return
    print(f"WARNING: {name} Faults Verification has {len(violations)} violations.")
    for v in violations:
        if v.get("type") == "target_coverage":
            print(f"  - Target Coverage Violation: Qubit {v['q']}, Amount: {v['violation_amount']}")
        else:
            print(f"  - Layer {v['layer']}, Qubits {v['Q']}, Injection Points {v['J']}, M_E_Q: {v['M_E_Q']}")

def _non_ft_cost_fn(w: int, t: int, non_ft_penalty_factor: float) -> float:
    return w + non_ft_penalty_factor * 2 * minimum_number_of_flags(w, t)

def _run_verification(
    single_faults, stabs, H_filter, t, top_n, verbose, is_ft, name, non_ft_penalty_factor=0.01
):
    if verbose:
        kind = "FT" if is_ft else "Non-FT"
        print(f"\n--- {name} Faults Verification ({kind}) ---")
        
    cost_fn = None if is_ft else lambda w, t_val: _non_ft_cost_fn(w, t_val, non_ft_penalty_factor)
    
    ver_stabs_layers, dfs, ticks, violations = find_lookahead_verification_stabilizers(
        single_faults=single_faults, stabs=stabs, H_filter=H_filter, t=t, top_n=top_n, 
        verbose=verbose, cost_fn=cost_fn
    )
    _print_violations(violations, name)
    return ver_stabs_layers, dfs, ticks, violations

def _inject_faults(ver_stabs_layers, ticks, num_qubits, target_fault_set, name, verbose):
    injected = compute_bare_injected_faults(ver_stabs_layers, ticks, num_qubits)
    target_fault_set.add_faults(injected.faults)
    target_fault_set.remove_duplicates()
    if verbose:
        print(f"Injected {len(injected.faults)} {name} faults into {name}-verification pool.")

def _synthesize_verification_layer(
    current_circ, v_layers, t_ticks, viol, num_qubits, t_val, basis, name, verbose
):
    ancilla_start = num_qubits
    phase_circ = stim.Circuit()
    if verbose: 
        print(f"\nSynthesizing {name} fault verification circuits...")
    
    for i, layer in enumerate(v_layers):
        stabs_qubits = []
        for stab in layer:
            qubits = np.where(stab)[0].tolist()
            stabs_qubits.append(qubits)
            
        layer_violations = [v for v in viol if v["layer"] == i]
        phase_circ = synthesize_and_merge_layer(
            phase_circ,
            stabs_qubits, 
            t_ticks[i],
            t=t_val,
            ancilla_start=ancilla_start,
            basis=basis,
            layer_violations=layer_violations
        )
    return current_circ + phase_circ

def _evaluate_configuration(
    final_M, col_ops, H_x, stabs_x, stabs_z, H_reduce_x, H_reduce_z, 
    t, d, top_n, first_layer, verbose, non_ft_penalty_factor, record=False
):
    circ = cat_at_origin(final_M, d, record=record)
    circ.append("TICK", [])
    for c, n in col_ops:
        circ.append("CX", [c, n])
    circ.append("TICK", [])
    
    if len(col_ops) == 0:
        return circ
        
    num_qubits = H_x.shape[1]
    
    if verbose:
        print("Computing initial single faults...")
    single_faults_x = compute_unitary_fault_set_1(col_ops, num_qubits=num_qubits, kind="X")
    single_faults_z = compute_unitary_fault_set_1(col_ops, num_qubits=num_qubits, kind="Z")
    
    if verbose:
        print(f"\nRunning verification stabilizers search for t={t} layers (top_n={top_n}, first_layer={first_layer})...")

    if first_layer == "X":
        ver_x_stabs_layers, _, ticks_x, violations_x = _run_verification(
            single_faults_x, stabs_x, H_reduce_x, t, top_n, verbose, is_ft=False, name="X", non_ft_penalty_factor=non_ft_penalty_factor
        )
        _inject_faults(ver_x_stabs_layers, ticks_x, num_qubits, single_faults_z, name="Z", verbose=verbose)
        ver_z_stabs_layers, _, ticks_z, violations_z = _run_verification(
            single_faults_z, stabs_z, H_reduce_z, t, top_n, verbose, is_ft=True, name="Z"
        )
    elif first_layer == "Z":
        ver_z_stabs_layers, _, ticks_z, violations_z = _run_verification(
            single_faults_z, stabs_z, H_reduce_z, t, top_n, verbose, is_ft=False, name="Z", non_ft_penalty_factor=non_ft_penalty_factor
        )
        _inject_faults(ver_z_stabs_layers, ticks_z, num_qubits, single_faults_x, name="X", verbose=verbose)
        ver_x_stabs_layers, _, ticks_x, violations_x = _run_verification(
            single_faults_x, stabs_x, H_reduce_x, t, top_n, verbose, is_ft=True, name="X"
        )
    else:
        ver_x_stabs_layers, _, ticks_x, violations_x = _run_verification(
            single_faults_x, stabs_x, H_reduce_x, t, top_n, verbose, is_ft=True, name="X"
        )
        ver_z_stabs_layers, _, ticks_z, violations_z = _run_verification(
            single_faults_z, stabs_z, H_reduce_z, t, top_n, verbose, is_ft=True, name="Z"
        )

    if first_layer == "Z":
        circ = _synthesize_verification_layer(
            circ, ver_z_stabs_layers, ticks_z, violations_z, num_qubits, 0, "X", "Z", verbose
        )
        circ = _synthesize_verification_layer(
            circ, ver_x_stabs_layers, ticks_x, violations_x, num_qubits, t, "Z", "X", verbose
        )
    else:
        circ = _synthesize_verification_layer(
            circ, ver_x_stabs_layers, ticks_x, violations_x, num_qubits, 0 if first_layer == "X" else t, "Z", "X", verbose
        )
        circ = _synthesize_verification_layer(
            circ, ver_z_stabs_layers, ticks_z, violations_z, num_qubits, t, "X", "Z", verbose
        )
        
    return circ

def cat_at_origin_with_verification(
    H_x: np.ndarray, H_z: np.ndarray, L_x: np.ndarray, L_z: np.ndarray, d: int,
    state: str = "0", max_col_ops: int = 100, top_n: int = 50, max_basis_tries: int = 10_000,
    first_layer: Literal["X", "Z", "none"] = "none", verbose: bool = False,
    heuristic: str = "overlap", non_ft_penalty_factor: float = 0.01,
    num_circuits: int = 1, record: bool = False
) -> stim.Circuit:
    t = d // 2

    if state == "0":
        pass
    elif state == "+":
        H_x, H_z = H_z, H_x
        L_x, L_z = L_z, L_x
    else:
        raise ValueError(f"Unknown state: {state}")

    # For verifying X faults, we measure Z-type observables (H_z + L_z)
    stabs_x = np.concatenate((H_z, L_z))
    # To reduce X faults, we use X-type stabilizers of |0>_L (H_x)
    H_reduce_x = H_x
    # For verifying Z faults, we measure X-type observables (H_x + L_x)
    stabs_z = H_x
    # To reduce Z faults, we use Z-type stabilizers of |0>_L (H_z + Z_L)
    H_reduce_z = np.concatenate((H_z, L_z)) if len(L_z) > 0 else H_z
        
    if verbose:
        print(f"Optimizing parity matrix (max_col_ops={max_col_ops}, num_circuits={num_circuits})...")
        
    unique_matrices = {}

    if verbose and num_circuits > 1:
        pbar = tqdm(total=num_circuits, desc="Finding unique matrix configuration")
    if verbose and num_circuits == 1:
        print("Finding unique matrix configuration")
    
    for run_idx in range(num_circuits):
        row_M, final_M, col_ops = optimize_fault_tolerant_matrix(
            H_x, t=t, max_col_ops=max_col_ops, H_x=H_x, H_z=H_z, max_basis_tries=max_basis_tries,
            stabs_X=stabs_z, stabs_Z=stabs_x,  # stabs_X expects X-type, stabs_Z expects Z-type
            H_reduce_X=H_reduce_x, H_reduce_Z=H_reduce_z, heuristic=heuristic
        )
        
        m_tup = tuple(final_M.flatten())
        if m_tup not in unique_matrices:
            unique_matrices[m_tup] = (final_M, col_ops)
            if verbose and num_circuits > 1:
                pbar.update(1)

    if verbose and num_circuits > 1:
        print(f"Found {len(unique_matrices)} unique configurations out of {num_circuits} runs. Evaluating them...")

    best_circ = None
    best_cnot_count = float('inf')

    for final_M, col_ops in unique_matrices.values():
        if verbose and num_circuits > 1:
            print(f"\nEvaluating configuration with Cost: {cnot_cost(final_M, t)}")
        elif verbose:
            print(f"Cost of final M: {cnot_cost(final_M, t)}")
            print(f"Chosen {len(col_ops)} CNOT gates: {col_ops}")
            
        circ = _evaluate_configuration(
            final_M, col_ops, H_x, stabs_x, stabs_z, H_reduce_x, H_reduce_z, 
            t, d, top_n, first_layer, verbose, non_ft_penalty_factor, record=record
        )
        
        if max_col_ops == 0:
            return circ
            
        current_cnots = count_operations(circ)[0]
        if current_cnots < best_cnot_count:
            best_cnot_count = current_cnots
            best_circ = circ.copy()
            if verbose and num_circuits > 1:
                print(f"  -> New best circuit found with {best_cnot_count} CNOTs!")
                
    return best_circ


if __name__ == "__main__":
    code = "12_2_4"

    print(f"Loading QECC: {code}")
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code)

    final_circ = row_optimized_cat_at_origin(
        H_z, d=d, record=True
    )

    print("\n--- Final Fault Tolerant Verification Circuit ---")
    print(f"Total Qubits: {final_circ.num_qubits}")
    print(f"Num CX: {count_operations(final_circ)[0]}")
    print(f"Total instructions: {sum(count_operations(final_circ))}")

