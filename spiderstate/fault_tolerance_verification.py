from typing import Union, List, Tuple, FrozenSet

import mip
import numpy as np
import stim
import re

from spiderstate.utils import SPECIAL_GATES, count_operations


# ==============================================================================
# PHASE 1: Circuit Evaluation Setup
# ==============================================================================
def append_ideal_measurements(
    noisy_prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_matrix: np.ndarray,
    basis_char: str
) -> stim.Circuit:
    """Appends ideal transversal measurements, detectors, and logical observables to the circuit."""
    eval_circ = noisy_prep_circ.copy()
    
    num_data_qubits = H_check.shape[1]
    
    # Transversally measure all data qubits
    eval_circ.append("MX" if basis_char == 'X' else "M", range(num_data_qubits))
    
    # Append code-space checks as detectors
    for row in H_check:
        detector_targets = []
        for i, val in enumerate(row):
            if val:
                detector_targets.append(stim.target_rec(-num_data_qubits + i))
        if detector_targets:
            eval_circ.append("DETECTOR", detector_targets)

    # Append logical observables
    if L_matrix is not None and L_matrix.size > 0:
        for m, L_row in enumerate(L_matrix):
            obs_targets = []
            for i, val in enumerate(L_row):
                if val:
                    obs_targets.append(stim.target_rec(-num_data_qubits + i))
            if obs_targets:
                eval_circ.append("OBSERVABLE_INCLUDE", obs_targets, m)

    return eval_circ


def extract_unique_fault_mechanisms(
    eval_circ: stim.Circuit
) -> List[FrozenSet[Tuple[str, int]]]:
    """Flattens circuit, extracts DEM, and returns a list of unique fault target combinations."""
    flat_circ = eval_circ.flattened()
    dem = flat_circ.detector_error_model(decompose_errors=False)

    fault_mechanisms = set()
    for instruction in dem:
        if instruction.type == "error":
            targets = []
            for tgt in instruction.targets_copy():
                if tgt.is_relative_detector_id():
                    targets.append(("D", tgt.val))
                elif tgt.is_logical_observable_id():
                    targets.append(("L", tgt.val))
            if targets:
                fault_mechanisms.add(frozenset(targets))

    return list(fault_mechanisms)


# ==============================================================================
# PHASE 2: The Mathematical Engine
# ==============================================================================
def solve_ftsp_ilp(
    unique_mechanisms: List[FrozenSet[Tuple[str, int]]],
    n_data_qubits: int,
    num_internal_detectors: int,
    H_check: np.ndarray,
    L_matrix: np.ndarray,
    d: int,
    t: int
) -> Tuple[mip.OptimizationStatus, List[mip.Var], List[mip.Var]]:
    """Builds and solves the exact GF(2) parity requirements for FTSP failure."""
    model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    model.verbose = 0

    num_e_init = len(unique_mechanisms)
    num_final_stabilizers = H_check.shape[0]
    num_logicals = L_matrix.shape[0] if L_matrix is not None else 0

    x = [model.add_var(var_type=mip.BINARY, name=f"x_{i}") for i in range(num_e_init)]
    y = [model.add_var(var_type=mip.BINARY, name=f"y_{j}") for j in range(n_data_qubits)]

    D_triggers = {k: [] for k in range(num_internal_detectors + num_final_stabilizers)}
    L_triggers = {m: [] for m in range(num_logicals)}

    for i, mech in enumerate(unique_mechanisms):
        for t_type, t_val in mech:
            if t_type == "D":
                D_triggers[t_val].append(x[i])
            elif t_type == "L":
                L_triggers[t_val].append(x[i])

    # A. Internal Flags Parity (Must be 0)
    for k in range(num_internal_detectors):
        if D_triggers[k]:
            q = model.add_var(var_type=mip.INTEGER, lb=0)
            model += mip.xsum(D_triggers[k]) - 2 * q == 0

    # B. Final Stabilizers Parity (Must be 0)
    for r in range(num_final_stabilizers):
        k = num_internal_detectors + r
        terms = D_triggers[k].copy()
        for j in range(n_data_qubits):
            if H_check[r, j]: terms.append(y[j])
        if terms:
            q = model.add_var(var_type=mip.INTEGER, lb=0)
            model += mip.xsum(terms) - 2 * q == 0

    # C. Logical Observable Parity (At least one must flip to 1)
    logical_flip_vars = []
    for m in range(num_logicals):
        terms = L_triggers[m].copy()
        for j in range(n_data_qubits):
            if L_matrix[m, j]: terms.append(y[j])
        if terms:
            l_m = model.add_var(var_type=mip.BINARY, name=f"l_{m}")
            q = model.add_var(var_type=mip.INTEGER, lb=0)
            model += mip.xsum(terms) - 2 * q - l_m == 0
            logical_flip_vars.append(l_m)

    if logical_flip_vars:
        model += mip.xsum(logical_flip_vars) >= 1
    else:
        # It is structurally impossible to flip the logicals
        return None, [], []

    # D. FTSP Thresholds
    model += mip.xsum(x) <= t
    model += mip.xsum(x) + mip.xsum(y) <= d - 1

    model.objective = 0
    status = model.optimize()
    return status, x, y


# ==============================================================================
# PHASE 3: Extraction & Rebuild (Direct Noisy Index Mapping)
# ==============================================================================
def extract_deterministic_failure(
    flat_noisy_prep: stim.Circuit,
    flat_eval_circ: stim.Circuit,
    failed_mechs: List[FrozenSet[Tuple[str, int]]],
    failed_y_indices: List[int],
    basis_char: str
) -> stim.Circuit:
    """
    Rebuilds the failed circuit by iterating through the flattened noisy circuit,
    swapping probabilistic noise channels for deterministic Pauli faults at the
    exact matched instruction index.
    """
    # 1. Build the DEM filter mask for E_init
    dem_filter = stim.DetectorErrorModel()
    for mech in failed_mechs:
        targets = []
        for t_type, t_val in sorted(mech):
            if t_type == "D":
                targets.append(stim.target_relative_detector_id(t_val))
            elif t_type == "L":
                targets.append(stim.target_logical_observable_id(t_val))
        dem_filter.append('error', [1.0], targets)

    explanations = flat_eval_circ.explain_detector_error_model_errors(
        dem_filter=dem_filter,
        reduce_to_one_representative_error=True
    )

    # 2. Extract E_init faults
    faults_to_inject = []
    for exp in explanations:
        loc = exp.circuit_error_locations[0]
        ins_offset = loc.stack_frames[0].instruction_offset

        paulis = []
        for p in loc.flipped_pauli_product:
            target = p.gate_target if hasattr(p, 'gate_target') else p
            if target.is_x_target:
                paulis.append(('X', target.value))
            elif target.is_y_target:
                paulis.append(('Y', target.value))
            elif target.is_z_target:
                paulis.append(('Z', target.value))

        faults_to_inject.append({
            'instruction_offset': ins_offset,
            'paulis': paulis,
            'injected': False
        })

    # 3. Iterate through the noisy circuit to rebuild the deterministic one
    failed_circ = stim.Circuit()

    for idx, inst in enumerate(flat_noisy_prep):
        # Inject the deterministic fault at the exact noisy index match
        for fault in faults_to_inject:
            if not fault['injected'] and fault['instruction_offset'] == idx:
                for p_type, p_val in fault['paulis']:
                    failed_circ.append(p_type, [p_val])
                fault['injected'] = True

        # We explicitly keep TICKs to preserve time slices
        if inst.name == "TICK":
            failed_circ.append(inst)
            continue

        # Strip out special gates (DETECTOR, OBSERVABLE_INCLUDE, etc.)
        if inst.name in SPECIAL_GATES:
            continue

        # Strip out the background noise channels
        if "ERROR" in inst.name or "DEPOLARIZE" in inst.name:
            continue

        # Append the standard quantum gate
        failed_circ.append(inst)

    # 4. Fallback (safety net)
    for fault in faults_to_inject:
        if not fault['injected']:
            for p_type, p_val in fault['paulis']:
                failed_circ.append(p_type, [p_val])
            fault['injected'] = True

    # 5. Inject E_data completion (Virtual Logical Faults)
    if failed_y_indices:
        failed_circ.append("TICK")  # Isolation barrier
        y_pauli = 'X' if basis_char == 'Z' else 'Z'
        for q in failed_y_indices:
            failed_circ.append(y_pauli, [q])

    return failed_circ


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
def verify_ftsp_ilp(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_op: np.ndarray,
    d: int,
    t: int,
    basis_char: str = "Z",
    verbose: bool = False
) -> Union[bool, stim.Circuit]:
    """
    Main entry point. Coordinates preparation, exact ILP math, and error extraction.
    """
    if verbose:
        print(f"\nStarting Exact ILP Verification for {basis_char}-basis (d={d}, t={t}).")

    from spiderstate.utils import make_stim_circ_noisy
    noisy_prep = make_stim_circ_noisy(prep_circ.copy(), p=0.001)

    n_data_qubits = H_check.shape[1]
    num_internal_detectors = noisy_prep.num_detectors

    L_matrix = np.atleast_2d(L_op) if L_op is not None else None

    # Step 1: Prep and Flatten
    eval_circ = append_ideal_measurements(noisy_prep, H_check, L_matrix, basis_char)
    unique_mechanisms = extract_unique_fault_mechanisms(eval_circ)

    if L_matrix is None or L_matrix.size == 0:
        if verbose: print(f"  [{basis_char} Basis] No logical observable. Skipping.")
        return True

    # Step 2: Solve the Math
    status, x_vars, y_vars = solve_ftsp_ilp(
        unique_mechanisms, n_data_qubits, num_internal_detectors,
        H_check, L_matrix, d, t
    )

    # Step 3: Extract the Verdict
    if status == mip.OptimizationStatus.OPTIMAL:
        w_init = sum(1 for var in x_vars if var.x >= 0.9)
        w_data = sum(1 for var in y_vars if var.x >= 0.9)

        if verbose:
            print(f"    [FAIL] Catastrophic cascade found!")
            print(f"    W(E_init)={w_init} faults bypassed flags to require only W(E_data)={w_data} to fail.")
            print("    Extracting unified failure circuit (Prep Faults + Data Faults)...")

        # Extract failed mechanisms and indices
        failed_mechs = [unique_mechanisms[i] for i, var in enumerate(x_vars) if var.x >= 0.9]
        failed_y_indices = [j for j, var in enumerate(y_vars) if var.x >= 0.9]

        # Pass E_data indices and basis to extraction function
        failed_circ = extract_deterministic_failure(
            prep_circ,
            eval_circ.flattened(),
            failed_mechs,
            failed_y_indices,
            basis_char
        )
        return failed_circ

    elif status is None:
        if verbose: print(f"    [PASS] Structurally impossible to flip Logical.")
        return True
    else:
        if verbose: print(f"    [PASS] UNSAT. The state preparation is strictly Fault-Tolerant.")
        return True


if __name__ == "__main__":
    import random
    random.seed(42)

    from spiderstate.utils import load_qecc, make_stim_circ_noisy
    from spiderstate.cat_at_origin import cat_at_origin_with_verification

    code = "12_2_4"
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code)
    t = (d - 1) // 2

    print(f"Generating circuit for {code} (d={d}, t={t})...")
    circ = cat_at_origin_with_verification(
        H_x=H_x, H_z=H_z, L_x=L_x, L_z=L_z, d=d,
        state="0", max_col_ops=1000, top_n=50, verbose=False
    )
    num_cx, num_meas = count_operations(circ)
    print(f"  [{code}] circuit generated!\n"
          f"  #CX: {num_cx}, #Meas: {num_meas}, Total: {num_cx + num_meas}")

    print("\nRunning Fast DEM/SAT FT Verification...")
    res_exhaustive = verify_ftsp_ilp(circ, H_z, np.atleast_2d(L_z), d, t, verbose=True)
    print("Verification Result:", res_exhaustive is True)
    # if res_exhaustive is True:
    #     print(circ)
