import stim
import numpy as np
from itertools import combinations
import math


def insert_noise(circ: stim.Circuit, p: float = 0.001, perfect_verification: bool = False) -> stim.Circuit:
    noisy = stim.Circuit()
    in_verification = False
    for inst in circ:
        if inst.name == "SHIFT_COORDS" and list(inst.gate_args_copy()) == [999.0]:
            in_verification = True
            noisy.append(inst)
            continue

        if inst.name in ["DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"]:
            noisy.append(inst)
            continue

        noisy.append(inst)

        if perfect_verification and in_verification:
            if inst.name in ["M", "MX"]:
                noisy.append("X_ERROR", inst.targets_copy(), p)
            continue

        if inst.name in ["H", "R", "RX", "X", "Z"]:
            noisy.append("DEPOLARIZE1", inst.targets_copy(), p)
        elif inst.name in ["CX", "CZ"]:
            noisy.append("DEPOLARIZE2", inst.targets_copy(), p)
        elif inst.name in ["M", "MX"]:
            noisy.append("X_ERROR", inst.targets_copy(), p)

    return noisy


def get_valid_syndromes(H_check: np.ndarray, L_check: np.ndarray | None, t: int):
    valid_syndromes = set()
    num_qubits = H_check.shape[1]

    for k in range(t + 1):
        for err_locs in combinations(range(num_qubits), k):
            err = np.zeros(num_qubits, dtype=int)
            err[list(err_locs)] = 1

            syn = tuple((H_check @ err) % 2)
            if L_check is not None:
                l_val = (L_check @ err) % 2
                valid_syndromes.add((syn, tuple(int(x) for x in l_val)))
            else:
                valid_syndromes.add(syn)

    return valid_syndromes


def verify_ft_exhaustive_basis(
    prep_circ: stim.Circuit,
    H_check: np.ndarray,
    L_check: np.ndarray | None,
    t: int,
    basis: str,
    verbose: bool = False,
    perfect_verification: bool = False
) -> bool:
    data_qubits = list(range(H_check.shape[1]))

    valid_syndromes = get_valid_syndromes(H_check, L_check, t)
    num_flags = prep_circ.num_detectors

    circ = prep_circ.copy()
    if basis == "X":
        for q in data_qubits:
            circ.append("H", q)
    circ.append("M", data_qubits)

    for row in H_check:
        targets = [stim.target_rec(i - len(data_qubits)) for i, val in enumerate(row) if val]
        circ.append("DETECTOR", targets)

    if L_check is not None:
        for o_idx, row in enumerate(L_check):
            targets = [stim.target_rec(i - len(data_qubits)) for i, val in enumerate(row) if val]
            circ.append("OBSERVABLE_INCLUDE", targets, o_idx)

    noisy = insert_noise(circ, perfect_verification=perfect_verification)
    dem = noisy.detector_error_model(decompose_errors=False)

    mechanisms = []
    for inst in dem:
        if inst.type == "error":
            dets = []
            obs = []
            for tgt in inst.targets_copy():
                if tgt.is_relative_detector_id():
                    dets.append(tgt.val)
                elif tgt.is_logical_observable_id():
                    obs.append(tgt.val)
            mechanisms.append((dets, obs))

    num_checks = H_check.shape[0]

    if verbose:
        print(f"Checking {len(mechanisms)} fault mechanisms up to {t} faults for {basis} basis...")

    for k in range(1, t + 1):
        for comb in combinations(mechanisms, k):
            all_dets = set()
            all_obs = set()
            for dets, obs in comb:
                for d in dets:
                    if d in all_dets:
                        all_dets.remove(d)
                    else:
                        all_dets.add(d)
                for o in obs:
                    if o in all_obs:
                        all_obs.remove(o)
                    else:
                        all_obs.add(o)

            flags_triggered = any(d < num_flags for d in all_dets)
            if flags_triggered:
                continue

            syn = tuple(1 if d in all_dets else 0 for d in range(num_flags, num_flags + num_checks))
            if L_check is not None:
                l_val = tuple(1 if o in all_obs else 0 for o in range(L_check.shape[0]))
                if (syn, l_val) not in valid_syndromes:
                    print(f"[{basis} Basis] FT FAILED! Fault comb length: {k}")
                    print(f"Produced syndrome: {syn}, L: {l_val}")
                    print(f"Fault combination: {comb}")
                    return False
            else:
                if syn not in valid_syndromes:
                    print(f"[{basis} Basis] FT FAILED! Fault comb length: {k}")
                    print(f"Produced syndrome: {syn}")
                    print(f"Fault combination: {comb}")
                    return False

    if verbose:
        print(f"[{basis} Basis] FT SUCCESS!")
    return True


def verify_ft_exhaustive(
    prep_circ: stim.Circuit,
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    t: int,
    state: str = "0",
    verbose: bool = False,
    perfect_verification: bool = False
) -> bool:
    """
    Verifies state preparation circuit fault tolerance up to t faults exhaustively.
    """
    if verbose:
        print(f"Starting exhaustive FT verification for state |{state}> up to t={t} faults.")
        if perfect_verification:
            print("Running in perfect verification mode (only M errors).")

    if state == "0":
        # Check Z-basis (detects X faults)
        ft_x = verify_ft_exhaustive_basis(prep_circ, H_z, L_z, t, "Z", verbose, perfect_verification)
        # Check X-basis (detects Z faults)
        ft_z = verify_ft_exhaustive_basis(prep_circ, H_x, None, t, "X", verbose, perfect_verification)

    elif state == "+":
        # Check Z-basis (detects X faults)
        ft_x = verify_ft_exhaustive_basis(prep_circ, H_z, None, t, "Z", verbose, perfect_verification)
        # Check X-basis (detects Z faults)
        ft_z = verify_ft_exhaustive_basis(prep_circ, H_x, L_x, t, "X", verbose, perfect_verification)

    else:
        raise ValueError("State must be '0' or '+'")

    return ft_x and ft_z


def verify_ft_stim(
    prep_circ: stim.Circuit,
    H_x: np.ndarray,
    H_z: np.ndarray,
    L_x: np.ndarray,
    L_z: np.ndarray,
    t: int,
    state: str = "0",
    verbose: bool = False
) -> bool:
    """
    Verifies state preparation circuit fault tolerance using stim's internal DEM search.
    This method only supports finding shortest undetectable logical errors for bases
    with an observable. For checking distance, this relies on search_for_undetectable_logical_errors.
    """
    if verbose:
        print(f"Starting stim heuristic FT verification for state |{state}> up to t={t} faults.")

    data_qubits = list(range(H_x.shape[1]))

    def check_basis(H_check, L_check, basis):
        circ = prep_circ.copy()
        if basis == "X":
            for q in data_qubits:
                circ.append("H", q)
        circ.append("M", data_qubits)
        for row in H_check:
            targets = [stim.target_rec(i - len(data_qubits)) for i, val in enumerate(row) if val]
            circ.append("DETECTOR", targets)

        if L_check is not None:
            for o_idx, row in enumerate(L_check):
                targets = [stim.target_rec(i - len(data_qubits)) for i, val in enumerate(row) if val]
                circ.append("OBSERVABLE_INCLUDE", targets, o_idx)
        else:
            # stim DEM requires an observable to define a logical error.
            # Without it, we cannot use search_for_undetectable_logical_errors.
            if verbose:
                print(f"[{basis} Basis] Skipping stim heuristic check (no logical observable).")
            return True

        noisy = insert_noise(circ)

        errs = noisy.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=10,
            dont_explore_edges_with_degree_above=10,
            dont_explore_edges_increasing_symptom_degree=False,
        )

        min_weight = len(errs) if errs else float('inf')
        if verbose:
            print(f"[{basis} Basis] Shortest undetectable logical error weight: {min_weight}")

        # But this doesn't check if the resulting syndrome is uncorrectable if it triggers final detectors!
        # stim's search_for_undetectable_logical_errors ASSUMES ideal decoding is not present.
        # It ONLY checks errors that trigger NO detectors. This makes it strictly less powerful
        # than the exhaustive method for state prep verification.

        if min_weight <= t:
            return False
        return True

    if state == "0":
        ft_x = check_basis(H_z, L_z, "Z")
        ft_z = check_basis(H_x, None, "X")
    elif state == "+":
        ft_x = check_basis(H_z, None, "Z")
        ft_z = check_basis(H_x, L_x, "X")
    else:
        raise ValueError("State must be '0' or '+'")

    return ft_x and ft_z


if __name__ == "__main__":
    from spiderstate.utils import load_qecc
    from spiderstate.cat_at_origin import cat_at_origin_with_verification
    import sys

    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc("15_7_3", "MQT")
    t = (d - 1) // 2

    print(f"Generating circuit for 17_1_5 (d={d}, t={t})...")
    circ = cat_at_origin_with_verification(
        H_x=H_x, H_z=H_z, L_x=L_x, L_z=L_z, d=d,
        state="0", max_col_ops=10, top_n=50, verbose=False
    )

    print("\nRunning Exhaustive FT Verification...")
    res_ex = verify_ft_exhaustive(circ, H_x, H_z, L_x, L_z, t, state="0", verbose=True)
    print("Exhaustive Result:", res_ex)

    print("\nRunning Stim Heuristic FT Verification...")
    res_stim = verify_ft_stim(circ, H_x, H_z, L_x, L_z, t, state="0", verbose=True)
    print("Stim Heuristic Result:", res_stim)

