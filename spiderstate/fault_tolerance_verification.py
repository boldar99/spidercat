import itertools
from typing import Union, List, Tuple, FrozenSet

import mip
import numpy as np
import stim

from spiderstate.cat_at_origin import cat_at_origin_with_verification
from spiderstate.stim_utils import get_circuit_depth
from spiderstate.utils import count_operations, NOISE_GATES, get_project_root


# ==============================================================================
# PHASE 1: Circuit Evaluation Setup & Utilities
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


def build_primary_eval_circ(
    noisy_prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_matrix: np.ndarray,
    basis_char: str
) -> stim.Circuit:
    """Builds the native FTSP evaluation circuit for the primary basis."""
    eval_circ = append_ideal_measurements(
        noisy_prep_circ=noisy_prep_circ,
        H_check=H_check,
        L_matrix=L_matrix,
        basis_char=basis_char
    )
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

    D_triggers: dict[int, list] = {k: [] for k in range(num_internal_detectors + num_final_stabilizers)}
    L_triggers: dict[int, list] = {m: [] for m in range(num_logicals)}

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
    faults_to_inject = {}
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

        faults_to_inject[ins_offset] = paulis

    # 3. Iterate through the noisy circuit to rebuild the deterministic one
    failed_circ = stim.Circuit()

    for idx, inst in enumerate(flat_eval_circ):
        # Inject the deterministic fault at the exact noisy index match
        if idx in faults_to_inject:
            for p_type, p_val in faults_to_inject[idx]:
                failed_circ.append(p_type, [p_val])
            del faults_to_inject[idx]
            continue

        if inst.name in NOISE_GATES:
            continue

        failed_circ.append(inst)

    # assert len(faults_to_inject) == 0

    # 5. Inject E_data completion (Virtual Logical Faults)
    if failed_y_indices:
        failed_circ.append("TICK")  # Isolation barrier
        y_pauli = 'X' if basis_char == 'Z' else 'Z'
        for q in failed_y_indices:
            failed_circ.append(y_pauli, [q])

    return failed_circ


# ==============================================================================
# COMBINATORICAL Z FAULT VERIFIER
# ==============================================================================
def precompute_conjugate_syndromes(H_check: np.ndarray, t: int) -> dict:
    """Precomputes the minimum weight data error required to satisfy a given syndrome."""
    valid_syndromes: dict[int, int] = {}
    n_qubits = H_check.shape[1]

    for w in range(t + 1):
        for combo in itertools.combinations(range(n_qubits), w):
            err = np.zeros(n_qubits, dtype=int)
            if w > 0:
                err[list(combo)] = 1

            # Convert binary array syndrome to a fast integer bitmask
            syn_array = (H_check @ err) % 2
            syn_int = sum(int(val) << i for i, val in enumerate(syn_array))

            if syn_int not in valid_syndromes or valid_syndromes[syn_int] > w:
                valid_syndromes[syn_int] = w

    return valid_syndromes


def solve_ftsp_combinatorial_fast(
    unique_mechanisms: List[FrozenSet[Tuple[str, int]]],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    H_check: np.ndarray,
    t: int
) -> Union[bool, List[FrozenSet[Tuple[str, int]]]]:
    """Evaluates the conjugate basis using a Bitwise Breadth-First Search (BFS)."""
    valid_syndromes = precompute_conjugate_syndromes(H_check, t)

    # Convert DEM mechanisms into fast integer bitmasks
    mech_masks = []
    for mech in unique_mechanisms:
        int_mask = 0
        ext_mask = 0
        for t_type, t_val in mech:
            if t_type == "D":
                if t_val < num_internal_detectors:
                    int_mask ^= (1 << t_val)
                else:
                    ext_mask ^= (1 << (t_val - num_internal_detectors))
        mech_masks.append((int_mask, ext_mask, mech))

    # BFS State Tracker: {(internal_mask, external_mask): [path_of_mechanisms]}
    reachable_states = {(0, 0): []}

    for k in range(1, t + 1):
        next_states = {}

        for (curr_int, curr_ext), path in reachable_states.items():
            for m_int, m_ext, mech in mech_masks:
                new_int = curr_int ^ m_int
                new_ext = curr_ext ^ m_ext
                state_key = (new_int, new_ext)

                # Prune degenerate physical faults that produce the same syndrome
                if state_key in reachable_states or state_key in next_states:
                    continue

                new_path = path + [mech]
                next_states[state_key] = new_path

                # Check FTSP Bounds
                if new_int == 0:  # Evades FTSP internal flags
                    if new_ext not in valid_syndromes or valid_syndromes[new_ext] > k:
                        return new_path  # Return the catastrophic cascade

        reachable_states.update(next_states)

    return True


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
def precompute_primary_syndromes(H_check: np.ndarray, L_matrix: np.ndarray, d: int) -> dict:
    """Precomputes the minimum weight data error required to produce a combined (syndrome, logical) signature."""
    valid_syndromes: dict[int, Tuple[int, List[int]]] = {}
    n_qubits = H_check.shape[1]
    
    if L_matrix is not None and L_matrix.size > 0:
        combined_matrix = np.vstack([H_check, L_matrix])
    else:
        combined_matrix = H_check
        
    for w in range(d):
        for combo in itertools.combinations(range(n_qubits), w):
            err = np.zeros(n_qubits, dtype=int)
            if w > 0:
                err[list(combo)] = 1
                
            syn_array = (combined_matrix @ err) % 2
            syn_int = sum(int(val) << i for i, val in enumerate(syn_array))
            
            if syn_int not in valid_syndromes or valid_syndromes[syn_int][0] > w:
                valid_syndromes[syn_int] = (w, list(combo))
                
    return valid_syndromes


def solve_ftsp_combinatorial_primary_fast(
    unique_mechanisms: List[FrozenSet[Tuple[str, int]]],
    num_internal_detectors: int,
    num_final_stabilizers: int,
    num_logicals: int,
    H_check: np.ndarray,
    L_matrix: np.ndarray,
    d: int,
    t: int
) -> Union[bool, Tuple[List[FrozenSet[Tuple[str, int]]], List[int]]]:
    """Evaluates the primary basis using a Bitwise Breadth-First Search (BFS)."""
    valid_syndromes = precompute_primary_syndromes(H_check, L_matrix, d)
    
    # Convert DEM mechanisms into fast integer bitmasks
    mech_masks = []
    for mech in unique_mechanisms:
        int_mask = 0
        ext_mask = 0
        for t_type, t_val in mech:
            if t_type == "D":
                if t_val < num_internal_detectors:
                    int_mask ^= (1 << t_val)
                else:
                    ext_mask ^= (1 << (t_val - num_internal_detectors))
            elif t_type == "L":
                ext_mask ^= (1 << (num_final_stabilizers + t_val))
        mech_masks.append((int_mask, ext_mask, mech))

    # BFS State Tracker: {(internal_mask, external_mask): [path_of_mechanisms]}
    reachable_states = {(0, 0): []}

    for k in range(1, t + 1):
        next_states = {}

        for (curr_int, curr_ext), path in reachable_states.items():
            for m_int, m_ext, mech in mech_masks:
                new_int = curr_int ^ m_int
                new_ext = curr_ext ^ m_ext
                state_key = (new_int, new_ext)

                # Prune degenerate physical faults that produce the same syndrome
                if state_key in reachable_states or state_key in next_states:
                    continue

                new_path = path + [mech]
                next_states[state_key] = new_path

                # Check FTSP Bounds
                if new_int == 0:  # Evades FTSP internal flags
                    stab_mask = (1 << num_final_stabilizers) - 1
                    for syn_data, (w_data, err_indices) in valid_syndromes.items():
                        if k + w_data <= d - 1:
                            if (new_ext ^ syn_data) & stab_mask == 0:
                                if (new_ext ^ syn_data) >> num_final_stabilizers != 0:
                                    return (new_path, err_indices)  # Return the catastrophic cascade

        reachable_states.update(next_states)

    return True


def verify_ftsp_primary_exact(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_op: np.ndarray,
    d: int,
    t: int,
    basis_char: str,
    verbose: bool = False
) -> Union[bool, stim.Circuit]:
    """Verifies the primary FTSP error basis using the Fast Combinatorial Tracker."""
    if verbose:
        print(f"\n  [Primary] Evaluating {basis_char}-basis via Combinatorics (d={d}, t={t})...")

    from spiderstate.utils import make_stim_circ_noisy
    noisy_prep = make_stim_circ_noisy(prep_circ.copy(), p=0.001)

    num_internal_detectors = noisy_prep.num_detectors
    num_final_stabilizers = H_check.shape[0]

    L_matrix = np.atleast_2d(L_op) if L_op is not None else None
    num_logicals = L_matrix.shape[0] if L_matrix is not None else 0

    # Step 1: Prep and Flatten
    eval_circ = build_primary_eval_circ(noisy_prep, H_check, L_matrix, basis_char)
    flat_eval_circ = eval_circ.flattened()

    unique_mechanisms = extract_unique_fault_mechanisms(flat_eval_circ)

    # Step 2: Solve the Math
    result = solve_ftsp_combinatorial_primary_fast(
        unique_mechanisms, num_internal_detectors, num_final_stabilizers, num_logicals,
        H_check, L_matrix, d, t
    )

    # Step 3: Extract the Verdict
    if isinstance(result, tuple):
        failed_mechs, failed_y_indices = result
        if verbose:
            print(f"    [FAIL] Catastrophic cascade found!")
            print(f"    W(E_init)={len(failed_mechs)} faults bypassed flags to require only W(E_data)={len(failed_y_indices)} to fail.")
            print("    Extracting unified failure circuit (Prep Faults + Data Faults)...")

        # Pass E_data indices and basis to extraction function
        fault_example = extract_deterministic_failure(
            flat_eval_circ,
            failed_mechs,
            failed_y_indices,
            basis_char
        )

        return fault_example
    else:
        if verbose:
            print(f"    [PASS] No uncorrectable conjugate cascades found.")
        return True


def verify_ftsp_primary_ilp(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_op: np.ndarray,
    d: int,
    t: int,
    basis_char: str,
    verbose: bool = False
) -> Union[bool, stim.Circuit]:
    """Verifies the primary FTSP error basis."""
    if verbose:
        print(f"\n  [Primary] Evaluating {basis_char}-basis (d={d}, t={t})...")

    from spiderstate.utils import make_stim_circ_noisy
    noisy_prep = make_stim_circ_noisy(prep_circ.copy(), p=0.001)

    n_data_qubits = H_check.shape[1]
    num_internal_detectors = noisy_prep.num_detectors

    L_matrix = np.atleast_2d(L_op) if L_op is not None else None

    # Step 1: Prep and Flatten
    eval_circ = build_primary_eval_circ(noisy_prep, H_check, L_matrix, basis_char)
    flat_eval_circ = eval_circ.flattened()

    unique_mechanisms = extract_unique_fault_mechanisms(flat_eval_circ)

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
        fault_example = extract_deterministic_failure(
            flat_eval_circ,
            failed_mechs,
            failed_y_indices,
            basis_char
        )
        # if verbose:
        #     print(fault_example)

        return fault_example
    else:
        if verbose:
            print(f"    [PASS] UNSAT. The state preparation is strictly Fault-Tolerant.")
        return True


def verify_ftsp_conjugate_exact(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    t: int,
    basis_char: str,
    verbose: bool = False
) -> Union[bool, stim.Circuit]:
    """Verifies the conjugate FTSP error basis using the Fast Combinatorial Tracker."""
    if verbose:
        print(f"\n  [Conjugate] Evaluating {basis_char}-basis via Combinatorics (t={t})...")

    from spiderstate.utils import make_stim_circ_noisy
    noisy_prep = make_stim_circ_noisy(prep_circ.copy(), p=0.001)

    num_internal_detectors = noisy_prep.num_detectors
    num_final_stabilizers = H_check.shape[0]

    # Conjugate tracking does not use L_matrix; we track pure syndrome mass
    eval_circ = append_ideal_measurements(noisy_prep, H_check, None, basis_char)
    flat_eval_circ = eval_circ.flattened()

    unique_mechanisms = extract_unique_fault_mechanisms(flat_eval_circ)

    result = solve_ftsp_combinatorial_fast(
        unique_mechanisms, num_internal_detectors, num_final_stabilizers, H_check, t
    )

    if isinstance(result, list):
        if verbose:
            print(f"    [FAIL] Bad Conjugate Cascade Detected!")
            print("    Extracting deterministic failure circuit...")

        fault_example = extract_deterministic_failure(flat_eval_circ, result, [], basis_char)
        # if verbose:
        #     print(fault_example)

        return fault_example

    if verbose:
        print(f"    [PASS] No uncorrectable conjugate cascades found.")
    return True


def verify_ftsp(
    prep_circ: stim.Circuit,
    H_primary: np.ndarray,
    L_primary: np.ndarray,
    H_conjugate: np.ndarray,
    L_conjugate: np.ndarray,
    d: int,
    t: int,
    primary_basis: str = "Z",
    conjugate_basis: str = "X",
    verbose: bool = False
) -> bool:
    """Comprehensive FT Verification mapping both primary and conjugate error channels."""
    res_primary = verify_ftsp_primary_ilp(
        prep_circ, H_primary, L_primary, d, t, basis_char=primary_basis, verbose=verbose
    )
    if res_primary is not True:
        return res_primary

    res_conjugate = verify_ftsp_conjugate_exact(
        prep_circ, H_conjugate, t, basis_char=conjugate_basis, verbose=verbose
    )
    if res_conjugate is not True:
        return res_conjugate

    return True


if __name__ == "__main__":
    import argparse
    from spiderstate.utils import load_qecc
    import random
    
    parser = argparse.ArgumentParser(description="Fault Tolerance Verification")
    parser.add_argument("--code", type=str, default="16_6_4", help="Code string (default: 17_1_5)")
    parser.add_argument("--state", type=str, default="0", help="State (default: 0)")
    parser.add_argument("--max_col_ops", type=int, default=100, help="Max column operations (default: 100)")
    parser.add_argument("--top_n", type=int, default=50, help="Top N (default: 50)")
    parser.add_argument("--num_circuits", type=int, default=50, help="Number of portfolio circuits (default: 1)")
    parser.add_argument("--first_layer", type=str, default="X", choices=["X", "Z", "none", "interleaved"], help="First layer (default: X)")
    parser.add_argument("--heuristic", type=str, default="overlap", choices=["overlap", "zero_tolerance", "weighted_syndrome", "global_sparsity", "max_contention", "soft_cover"], help="Heuristic for fault tracker")
    parser.add_argument("--seed", type=str, default="100", help="The random seed")
    parser.add_argument("--save_circuit", type=str, default="", help="Path to save the circuit if FT")

    args = parser.parse_args()
    
    random.seed(args.seed)

    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(args.code)
    t = d // 2

    print(f"Generating circuit for {args.code} (d={d}, t={t})...")
    circ = cat_at_origin_with_verification(
        H_x=H_x, H_z=H_z, L_x=L_x, L_z=L_z, d=d,
        state=args.state, max_col_ops=args.max_col_ops, top_n=args.top_n,
        first_layer=args.first_layer, verbose=True, heuristic=args.heuristic, num_circuits=args.num_circuits,
    )
    # circ = stim.Circuit(get_project_root().joinpath("good_circuits", f"{args.code}.stim").read_text())


    num_cx, num_meas = count_operations(circ)
    print(f"  [{args.code}] circuit generated!\n"
          f"  #CX: {num_cx}, #Meas: {num_meas}, Total: {num_cx + num_meas}\n"
          f"  Depth: {get_circuit_depth(circ)}")

    print("\nRunning Comprehensive ExRec/SAT FT Verification...")

    is_ft = verify_ftsp(
        prep_circ=circ,
        H_primary=H_z, L_primary=np.atleast_2d(L_z),
        H_conjugate=H_x, L_conjugate=np.atleast_2d(L_x),
        d=d, t=t,
        primary_basis="Z", conjugate_basis="X",
        verbose=True
    )

    print("\nFinal Verification Result:", is_ft is True)
    
    if is_ft is True and args.save_circuit:
        with open(args.save_circuit, "w") as f:
            f.write(str(circ))
