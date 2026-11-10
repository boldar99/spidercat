import hashlib
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Any
import json

import numpy as np
import pandas as pd
import stim
import tqdm
from joblib import Parallel, delayed
from numpy import ndarray
from scipy.stats import norm
from spidercat.utils import qasm_to_stim, make_stim_circ_noisy, get_project_root, get_assets_dir, get_repo_root

cwd = Path(__file__).parent


@dataclass(frozen=True)
class SimulationInfo:
    circuit: stim.Circuit
    circuit_hash: str
    method: str
    num_samples: int
    num_flags: int
    p: float
    n: int
    t: int
    basis: str

    @classmethod
    def init(cls, method, n, t, p, num_samples, basis, simulation_results_dir: Path | None = None):
        circ = load_stim_circuit(t, n, method)
        if circ is None:
            return None
        noisy_circ, _ = make_stim_circ_noisy(circ, p_1=p, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
        add_measurements(noisy_circ, n, method, basis)
        circ_hash = hashlib.sha256(str(noisy_circ).encode()).hexdigest()[:16]

        if simulation_results_dir is None:
            simulation_results_dir = get_repo_root().joinpath("benchmarks", "simulation_results")
        csv_file = simulation_results_dir.joinpath(f"{circ_hash}.csv")
        if csv_file.exists():
            df_existing = pd.read_csv(csv_file)
            existing_samples = int(df_existing["num_samples"].sum()) if not df_existing.empty else 0
            num_samples = max(0, num_samples - existing_samples)

        return cls(
            circuit=noisy_circ,
            circuit_hash=circ_hash,
            method=method,
            num_samples=num_samples,
            num_flags=circ.num_qubits - n,
            p=p,
            t=t,
            n=n,
            basis=basis,
        )

    def with_samples(self, num_samples: int) -> "SimulationInfo":
        return replace(self, num_samples=num_samples)


def load_stim_circuit(t: int, n: int, method: str, p: int = 1):
    assets = get_assets_dir()
    method_to_file = {
        "spider-cat": assets / f"circuits/cat_state_t{t}_n{n}_p{p}.stim",
        "spider-cat-optimized": assets / f"circuits/cat_state_t{t}_n{n}_p{p}.stim",
        "spider-cat-old": assets / f"circuits_old/cat_state_t{t}_n{n}_p{p}.stim",
        "flag-at-origin": assets / f"flag_at_origin_circuits/d{t * 2 + 1}-q{n}-GHZ.qasm",
        "MQT": assets / f"MQT_circuits/ft_ghz_{n}_{t}.stim",
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
    if method in ("spider-cat", "spider-cat-old", "spider-cat-optimized", "flag-at-origin"):
        return np.any(flags, axis=1)
    if method == "MQT":
        return ~np.logical_or(np.all(flags == 0, axis=1), np.all(flags == 1, axis=1))
    raise ValueError(f"Unknown error detection method: {method}")


def add_measurements(circ: stim.Circuit, n: int, method: str, basis="Z"):
    if method in ("flag-at-origin", ):
        circ.append("M" if basis == "Z" else "MX", range(circ.num_qubits - n, circ.num_qubits))
    if method in ("MQT", "spider-cat", "spider-cat-old", "spider-cat-optimized"):
        circ.append("M" if basis == "Z" else "MX", range(n))


def run_simulation(simulation_info: SimulationInfo):
    sampler = simulation_info.circuit.compile_sampler()
    samples = sampler.sample(simulation_info.num_samples)
    # 1. Post-selection (Discard runs where flags triggered)
    if simulation_info.method in ("spider-cat", "spider-cat-old", "spider-cat-optimized"):
        converter = simulation_info.circuit.compile_m2d_converter()
        detectors = converter.convert(measurements=samples, append_observables=False)
        error_detected = np.any(detectors, axis=1)
    else:
        error_detected = error_is_detected(samples[:, :simulation_info.num_flags], simulation_info.method)
    post_selected_samples = samples[~error_detected]

    num_accepted = post_selected_samples.shape[0]
    error_counts = get_error_counts(post_selected_samples, simulation_info)

    return simulation_info, num_accepted, error_counts


def get_error_counts(post_selected_samples: ndarray[tuple[int, ...], Any],
                     simulation_info: SimulationInfo) -> dict[Any, Any]:
    if len(post_selected_samples) == 0:
        return {}

    raw_errors = np.sum(post_selected_samples[:, simulation_info.num_flags:], axis=1)

    # Fold errors: if > n/2, it means we drifted to the "other" logical state
    if simulation_info.basis == "Z":
        num_data_errors = np.where(
            raw_errors > simulation_info.n // 2,
            simulation_info.n - raw_errors,
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
        d, m = divmod(info.num_samples, _batch_size)
        batched_sim_info += [info.with_samples(_batch_size)] * d
        if m:
            batched_sim_info.append(info.with_samples(m))
    return batched_sim_info


def make_simulation_info_list(ps, methods, ts, ns, basis, num_samples=10_000_000, simulation_results_dir: Path | None = None):
    circuits = []
    reported = set()
    for p, m, t, n in itertools.product(ps, methods, ts, ns):
        si = SimulationInfo.init(m, n, t, p, num_samples=num_samples, basis=basis, simulation_results_dir=simulation_results_dir)
        if si is None:
            if (m, n, t) in reported: continue
            print(f"Circuit not available for method={m}, {n=}, {t=}", file=sys.stderr)
            reported.add((m, n, t))
            continue
        if si.num_samples > 0:
            circuits.append(si)
    return circuits


if __name__ == "__main__":
    benchmarks_dir = get_repo_root().joinpath("benchmarks")
    simulation_results_dir = benchmarks_dir.joinpath("simulation_results")
    simulated_circuits_dir = benchmarks_dir.joinpath("simulated_circuits")
    os.makedirs(simulation_results_dir, exist_ok=True)
    os.makedirs(simulated_circuits_dir, exist_ok=True)
    key_file = benchmarks_dir.joinpath("key_file.csv")

    if key_file.exists():
        df_existing_keys = pd.read_csv(key_file).drop_duplicates(subset=["circuit_hash"])
        df_existing_keys.to_csv(key_file, header=True, index=False)
        initialized = set(df_existing_keys["circuit_hash"].astype(str))
    else:
        pd.DataFrame(columns=[
            "method", "t", "n", "basis", "p", "circuit_hash"
        ]).to_csv(key_file, header=True, index=False)
        initialized = set()

    start_time = time.time()

    # N = 50
    # T = 7
    methods = ["spider-cat-old", "spider-cat-optimized", "flag-at-origin", "MQT"]
    ps = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    ts = range(2, 7)
    ns = range(8, 51)
    simulation_info = make_simulation_info_list(
        ps, methods, ts, ns, basis="Z", simulation_results_dir=simulation_results_dir
    )
    batched_sim_info = batch_sim_info(10_000_000, simulation_info)

    print("Simulation Information Prepared")

    results = Parallel(n_jobs=-2, return_as="generator")(delayed(run_simulation)(si) for si in batched_sim_info)
    # results = (run_simulation(si) for si in batched_sim_info)

    for info, num_accepted, error_counts in tqdm.tqdm(results, total=len(batched_sim_info)):
        circ_hash = info.circuit_hash
        circuit_file = simulated_circuits_dir.joinpath(f"{circ_hash}.stim")
        if not circuit_file.exists():
            info.circuit.to_file(circuit_file)

        df_results = pd.DataFrame([{
            "num_samples": info.num_samples,
            "num_accepted": num_accepted,
            "error_counts": json.dumps(error_counts)
        }])
        csv_file = str(simulation_results_dir.joinpath(f'{circ_hash}.csv'))
        if os.path.exists(csv_file):
            df_results.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_results.to_csv(csv_file, index=False)

        p, method, t, n, basis = info.p, info.method, info.t, info.n, info.basis

        if circ_hash not in initialized:
            df_key = pd.DataFrame([{
                "method": method,
                "t": t,
                "n": n,
                "basis": basis,
                "p": p,
                "circuit_hash": circ_hash,
            }])
            df_key.to_csv(key_file, mode='a', header=False, index=False)
            initialized.add(circ_hash)

    print("--- %s seconds ---" % (time.time() - start_time))
