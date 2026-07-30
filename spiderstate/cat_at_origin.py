import networkx as nx
import numpy as np
import stim
from tqdm import tqdm

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder
from spidercat.draw import draw_forest_on_graph, display_digraph
from spiderstate.circuit_extraction import ExtractionPolicy, extract_stabiliser_state
from spiderstate.stabiliser_decomposition import decompose_stabiliser_state
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



def col_reduced_cat_at_origin(H: np.ndarray, d: int, max_col_ops: int = 0, max_basis_tries: int = 5000):
    t = (d - 1) // 2
    _, final_matrix_after_col_ops, col_ops_performed = optimize_fault_tolerant_matrix(H, t, max_col_ops, max_basis_tries)
    circ = cat_at_origin(final_matrix_after_col_ops, d)
    for (c, n) in col_ops_performed:
        circ.append("CX", [c, n])

    return circ


def row_optimized_cat_at_origin(H: np.ndarray, d: int, max_basis_tries: int = 10_000):
    t = (d - 1) // 2
    best_row_op_cost, matrix_after_row_ops = row_optimize_matrix(H, t, max_basis_tries)
    return cat_at_origin(matrix_after_row_ops, d)


def cat_at_origin(H: np.ndarray, d: int, draw_solutions=False) -> stim.Circuit:
    decomposition = decompose_stabiliser_state(
        H,
        d,
        unique_ones_validator=has_unique_ones_property,
        pivot_finder=find_pivots_in_matrix,
        cat_state_factory=well_ordered_ft_cat_state_data,
        edge_matcher=match_edges,
    )
    if draw_solutions:
        draw_forest_on_graph(
            decomposition.graph,
            decomposition.forest,
            figsize=(8, 8),
        )
        display_digraph(decomposition.dependency_graph, figsize=(8, 8))
    result = extract_stabiliser_state(
        decomposition,
        policy=ExtractionPolicy(),
        builder=StimBuilder(),
        extractor_factory=CatStateExtractor,
    )
    return result.circuit


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
    t, d, top_n, first_layer, verbose, non_ft_penalty_factor
):
    circ = cat_at_origin(final_M, d)
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
    num_circuits: int = 1
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
            t, d, top_n, first_layer, verbose, non_ft_penalty_factor
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
    max_col_ops = 100

    print(f"Loading QECC: {code}")
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code)

    final_circ = cat_at_origin_with_verification(
        H_x=H_x, H_z=H_z, L_x=L_x, L_z=L_z, d=d,
        max_col_ops=max_col_ops, verbose=True
    )

    print("\n--- Final Fault Tolerant Verification Circuit ---")
    print(f"Total Qubits: {final_circ.num_qubits}")
    print(f"Num CX: {count_operations(final_circ)[0]}")
    print(f"Total instructions: {sum(count_operations(final_circ))}")
