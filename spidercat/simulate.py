import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import stim
import tqdm
from joblib import Parallel, delayed
from mypy.checkexpr import defaultdict
from numpy import dtype, ndarray
from scipy.stats import norm
from spidercss.stim_utils import get_circuit_depth, get_num_cnots
from stim import Circuit

from spidercat.utils import qasm_to_stim, make_stim_circ_noisy, get_project_root

import csv

cwd = Path(__file__).parent

from typing import TypedDict, Iterable, Any


class SimulationInfo(TypedDict):
    circuit: stim.Circuit
    method: str
    num_samples: int
    num_flags: int
    p: float
    n: int
    t: int
    basis: str


class SimulationResults(TypedDict):
    p: float
    method: str
    t: int
    n: int
    k: int
    circuit_hash: str


def load_stim_circuit(t: int, n: int, method: str, p: int = 1):
    method_to_file = {
        "spider-cat": Path(f"{cwd}/circuits/cat_state_t{t}_n{n}_p{p}.stim"),
        "flag-at-origin": Path(f"{cwd}/flag_at_origin_circuits/d{t * 2 + 1}-q{n}-GHZ.qasm"),
        "MQT": Path(f"{cwd}/MQT_circuits/ft_ghz_{n}_{t}.stim"),
    }
    my_file = method_to_file[method]
    if not my_file.is_file():
        return None

    if method == "flag-at-origin":
        qasm_txt = my_file.read_text()
        circuit = qasm_to_stim(qasm_txt)
        circuit.append("M", range(circuit.num_qubits - n))
        return circuit

    return stim.Circuit(my_file.read_text())


def save_simulation_data(n, t, samples: np.ndarray):
    # now = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = f"simulation_data/samples_{n}_{t}.npz"
    stim.write_shot_data_file(data=samples, path=file_name, format="b8")
    # np.savez_compressed(file_name, samples)
    print(f"Saved samples to {file_name}")


def calculate_wilson_interval(k, n, confidence=0.95):
    """
    Calculates the Wilson Score Interval for a binomial proportion.
    Ideal for QEC where error rates (p) are often close to 0.
    k: number of successes (or errors)
    n: total number of trials (accepted samples)
    """
    if n == 0:
        return 0.0, 0.0

    p = k / n
    z = norm.ppf(1 - (1 - confidence) / 2).tolist()

    denominator = 1 + z ** 2 / n
    center_adjusted_prob = p + z ** 2 / (2 * n)
    adjusted_std_dev = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)

    lower_bound = (center_adjusted_prob - adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_prob + adjusted_std_dev) / denominator

    return max(0.0, lower_bound), min(1.0, upper_bound)


def error_is_detected(flags: np.ndarray, method: str):
    if method in ("spider-cat", "flag-at-origin"):
        return np.any(flags, axis=1)
    if method == "MQT":
        return ~np.logical_or(np.all(flags == 0, axis=1), np.all(flags == 1, axis=1))
    raise ValueError(f"Unknown error detection method: {method}")


def process_samples(samples: np.ndarray, num_flags: int, n: int, t: int, p: float, total_samples_attempted: int,
                    circuit: stim.Circuit, method: str, num_paths=1, basis="Z"):
    """
    Processes raw simulation samples to generate rich statistics.
    """
    # 1. Post-selection (Discard runs where flags triggered)
    if method == 'spider-cat':
        converter = circuit.compile_m2d_converter()
        detectors = converter.convert(measurements=samples, append_observables=False)
        error_detected = np.any(detectors, axis=1)
    else:
        error_detected = error_is_detected(samples[:, :num_flags], method)
    post_selected_samples = samples[~error_detected]

    num_accepted = post_selected_samples.shape[0]
    acceptance_rate = num_accepted / total_samples_attempted if total_samples_attempted > 0 else 0.0

    # 2. Calculate Data Errors (Distance from nearest codeword)
    if num_accepted > 0:
        raw_errors = np.sum(post_selected_samples[:, num_flags:], axis=1)
        # Fold errors: if > n/2, it means we drifted to the "other" logical state
        if basis == "Z":
            num_data_errors = np.where(raw_errors > n // 2, n - raw_errors, raw_errors)
            unique_errors, counts = np.unique(num_data_errors, return_counts=True)
            error_counts = dict(zip(unique_errors.tolist(), counts.tolist()))
        else:
            num_errors = np.sum(raw_errors % 2).tolist()
            error_counts = {0: len(raw_errors) - num_errors, 1: num_errors}
    else:
        error_counts = {}

    # 3. Generate Statistics Dictionary
    stats = {}

    # We want entries even for 0 errors if it happened, or if the dict is empty
    # If no samples accepted, return empty or a default failure record
    if num_accepted == 0:
        return {}

    for k, count in error_counts.items():
        p_hat = count / num_accepted

        # Standard Error (Wald)
        std_error = math.sqrt(p_hat * (1 - p_hat) / num_accepted)

        # Wilson Interval (Better for low prob)
        wilson_low, wilson_high = calculate_wilson_interval(count, num_accepted)

        # Wald Interval (Standard, but can be inaccurate near 0)
        z = 1.96  # 95% confidence
        wald_low = p_hat - z * std_error
        wald_high = p_hat + z * std_error

        stats[k] = {
            "n": n,
            't': t,
            'p': p,
            "k": k,
            "count": count,
            "total_samples": total_samples_attempted,
            "num_accepted": num_accepted,
            "probability": p_hat,  # P(k | accepted)
            "acceptance_rate": acceptance_rate,
            "std_error": std_error,
            "ci_wilson_lower": wilson_low,
            "ci_wilson_upper": wilson_high,
            "ci_wald_lower": max(0.0, wald_low),
            "ci_wald_upper": min(1.0, wald_high),
            "circuit": str(circuit),
            "num_cx": get_num_cnots(circuit),
            "depth": get_circuit_depth(circuit),
            "num_flags": num_flags,
            "num_qubits": circuit.num_qubits,
            "num_paths": num_paths,
        }

    return stats


def add_measurements(circ: stim.Circuit, n: int, method: str, basis = "Z"):
    if method in ("flag-at-origin"):
        circ.append("M" if basis == "Z" else "MX", range(circ.num_qubits - n, circ.num_qubits))
    if method in ("MQT", "spider-cat"):
        circ.append("M" if basis == "Z" else "MX", range(n))


# def run_simulation(n: int, t: int, p: float, num_samples: int = 1_000_000, method="spider-cat",
#                    save_samples: bool = False, num_paths=1, basis="Z"):
#     circ = load_stim_circuit(t, n, method, num_paths)
#     if circ is None:
#         return None
#
#     num_flags = circ.num_qubits - n
#     noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
#     # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=p / 100)
#     # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=0, p_mem=0)
#     add_measurements(noisy_circ, n, method, basis)
#
#     # Run the simulation
#     circuit_sampler = noisy_circ.compile_sampler()
#     samples: np.ndarray = circuit_sampler.sample(num_samples)
#
#     if save_samples:
#         save_simulation_data(n, t, samples)
#
#     # Process metrics
#     stats = process_samples(samples, num_flags, n, t, p, num_samples, noisy_circ, method, num_paths, basis)
#
#     # Optional: Print summary of counts for quick debugging
#     counts_summary = {k: v['count'] for k, v in stats.items()}
#     print(f"Stats for {t}-FT {n}-cat (p={p}): {counts_summary}")
#
#     return stats
#
#
# # 1. Define a helper function that handles a SINGLE 'n'
# def process_simulation(n, t, p, num_samples, method='spider-cat', num_paths=1, basis="Z"):
#     stats_dict = run_simulation(n, t, p, num_samples, method, num_paths=num_paths, basis=basis)
#     if stats_dict is None:
#         return []
#     return list(stats_dict.values())
#
#
# def simulate_t_n(ts, ns, method='spider-cat', num_paths=1, basis="Z"):
#     print("Starting simulation loop, varying values of t and n")
#     print(f"Method: {method}; Number of paths: {num_paths}")
#     # parallel_results = (
#     #     process_simulation(n, t, p=0.01, num_samples=1_000_000, method=method, num_paths=num_paths) for t in ts for n in
#     #     ns
#     # )
#     parallel_results = Parallel(n_jobs=-2)(
#         delayed(process_simulation)(n, t, p=0.05, num_samples=5_000_000 * min(t, 10), method=method,
#                                     num_paths=num_paths, basis=basis) for t in ts for n in ns
#     )
#     collected_data = [item for sublist in parallel_results for item in sublist]
#     basis_str = "_X" if basis == "X" else ""
#     with open(f"simulation_data/simulation_results_t_n_{method}-opt_p{num_paths}{basis_str}.json", "w") as f:
#         json.dump(collected_data, f, indent=4)
#     print("Simulation complete")
#     print()


def simulate_t_p(ts, ps, n):
    print("Starting simulation loop, varying values of t and p")
    parallel_results = Parallel(n_jobs=-3)(
        delayed(process_simulation)(n=n, t=t, p=p, num_samples=5_000_000) for t in ts for p in ps
    )
    collected_data = [item for sublist in parallel_results for item in sublist]
    with open(f"simulation_data/simulation_results_t_p_n{n}.json", "w") as f:
        json.dump(collected_data, f, indent=4)
    print("Simulation complete")
    print()


def get_simulation_info(method, n, t, p, num_samples, basis) -> SimulationInfo | None:
    circ = load_stim_circuit(t, n, method)
    if circ is None:
        return None
    noisy_circ, _ = make_stim_circ_noisy(circ, p_1=p, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    add_measurements(noisy_circ, n, method, basis)

    return SimulationInfo(
        circuit=noisy_circ,
        method=method,
        p=p,
        t=t,
        n=n,
        num_samples=num_samples,
        num_flags=circ.num_qubits - n,
        basis=basis,
    )




def run_simulation(simulation_info: SimulationInfo):
    sampler = simulation_info["circuit"].compile_sampler()
    samples = sampler.sample(simulation_info["num_samples"])
    # 1. Post-selection (Discard runs where flags triggered)
    if simulation_info["method"] == 'spider-cat':
        converter = simulation_info["circuit"].compile_m2d_converter()
        detectors = converter.convert(measurements=samples, append_observables=False)
        error_detected = np.any(detectors, axis=1)
    else:
        error_detected = error_is_detected(samples[:, :simulation_info["num_flags"]], simulation_info["method"])
    post_selected_samples = samples[~error_detected]

    num_accepted = post_selected_samples.shape[0]
    error_counts = get_error_counts(post_selected_samples, simulation_info)

    return simulation_info, num_accepted, error_counts


def get_error_counts(post_selected_samples: ndarray[tuple[int, ...], Any],
                     simulation_info: SimulationInfo) -> dict[Any, Any]:
    if len(post_selected_samples) == 0:
        return {}

    raw_errors = np.sum(post_selected_samples[:, simulation_info["num_flags"]:], axis=1)

    # Fold errors: if > n/2, it means we drifted to the "other" logical state
    if simulation_info["basis"] == "Z":
        num_data_errors = np.where(
            raw_errors > simulation_info["n"] // 2,
            simulation_info["n"] - raw_errors,
            raw_errors
        )
        unique_errors, counts = np.unique(num_data_errors, return_counts=True)
        return dict(zip(unique_errors.tolist(), counts.tolist()))
    else:
        num_errors = np.sum(raw_errors % 2).tolist()
        return {0: len(raw_errors) - num_errors, 1: num_errors}


def batch_sim_info(_batch_size: int, simulation_info: Iterable[SimulationInfo]) -> list[Any]:
    batched_sim_info = []
    for info in simulation_info:
        d, m = divmod(info["num_samples"], _batch_size)
        batched_sim_info += [info | {"num_samples": _batch_size}] * d
        if m:
            batched_sim_info.append(info | {"num_samples": m})
    return batched_sim_info


def make_simulation_info_list(ps, methods, ts, ns, basis, num_samples=5_000_000):
    circuits = []
    reported = set()
    for p, m, t, n in itertools.product(ps, methods, ts, ns):
        si = get_simulation_info(m, n, t, p, num_samples=num_samples, basis=basis)
        if si is None:
            if (m, n, t) in reported: continue
            print(f"Circuit not available for method={m}, {n=}, {t=}", file=sys.stderr)
            reported.add((m, n, t))
            continue
        circuits.append(si)
    return circuits


if __name__ == "__main__":
    benchmakrs_dir = get_project_root().joinpath("benchmarks")
    simulation_results_dir = benchmakrs_dir.joinpath("simulation_results")
    simulated_circuits_dir = benchmakrs_dir.joinpath("simulated_circuits")
    os.makedirs(simulation_results_dir, exist_ok=True)
    os.makedirs(simulated_circuits_dir, exist_ok=True)
    key_file = benchmakrs_dir.joinpath("key_file.csv")

    if not os.path.exists(key_file):
        pd.DataFrame(columns=[
            "method", "t", "n", "basis", "p", "circuit_hash"
        ]).to_csv(key_file, header=True, index=False)

    start_time = time.time()

    N = 50
    T = 2
    # # simulate_t_n(range(2, T+1), range(8, N + 1), method="spider-cat", num_paths=1)
    # simulate_t_n(range(3, T+1), range(10, N + 1), method="spider-cat", num_paths=1)
    # simulate_t_n(range(3, 7), range(8, N + 1), method="spider-cat", num_paths=2)
    # simulate_t_n(range(3, 7), range(8, N + 1), method="spider-cat", num_paths=3)
    # simulate_t_n(range(3, 7), range(8, N + 1), method="spider-cat", num_paths=4)
    # simulate_t_n(range(3, T + 1), range(8, N + 1), method="spider-cat", num_paths=5)
    # simulate_t_n(range(3, T+1), range(8, N + 1), method="spider-cat", num_paths=10)
    # simulate_t_n(range(3, T + 1), range(8, N + 1), method="spider-cat", num_paths=20)
    # simulate_t_n(range(2, T+1), range(8, N + 1), method="flag-at-origin")
    # simulate_t_n(range(2, T+1), range(8, N + 1), method="MQT")
    # simulate_t_p(range(3, 8), (10 ** np.linspace(-0.3, -2, 18)).tolist(), n=24)
    # simulate_t_p(range(3, 8), (10 ** np.linspace(-0.3, -2, 18)).tolist(), n=50)
    # simulate_t_p(range(3, 8), (10 ** np.linspace(-0.3, -2, 18)).tolist(), n=80)
    ps = [0.05]
    methods = ["spider-cat", "flag-at-origin", "MQT"]
    ts = range(2, T+1)
    ns = range(8, N + 1)
    simulation_info = make_simulation_info_list(ps, methods, ts, ns, basis="Z")
    batched_sim_info = batch_sim_info(10_000_000, simulation_info)
    # results = (run_simulation(si) for si in batched_sim_info)

    print("Simulation Information Prepared")

    results = Parallel(n_jobs=-2, return_as="generator")(delayed(run_simulation)(si) for si in batched_sim_info)
    # results = (run_simulation(si) for si in batched_sim_info)
    initialized = set()

    for info, num_accepted, error_counts in tqdm.tqdm(results, total=len(batched_sim_info)):

        circ_str = str(info["circuit"])
        circ_hash = hashlib.sha256(circ_str.encode()).hexdigest()[:16]
        circuit_file = simulated_circuits_dir.joinpath(f"{circ_hash}.stim")
        if not circuit_file.exists():
            info["circuit"].to_file(circuit_file)

        df_results = pd.DataFrame([{
            "num_samples": info["num_samples"],
            "num_accepted": num_accepted,
            "error_counts": error_counts
        }])
        csv_file = str(simulation_results_dir.joinpath(f'{circ_hash}.csv'))
        if os.path.exists(csv_file):
            df_results.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_results.to_csv(csv_file, index=False)

        p, method, t, n, basis = info["p"], info["method"], info["t"], info["n"], info["basis"]

        df_key = pd.DataFrame([{
            "method": method,
            "t": t,
            "n": n,
            "basis": basis,
            "p": p,
            "circuit_hash": circ_hash,
        }])
        if circ_hash not in initialized:
            df_key.to_csv(key_file, mode='a', header=False, index=False)
            initialized.add(circ_hash)

    print("--- %s seconds ---" % (time.time() - start_time))
