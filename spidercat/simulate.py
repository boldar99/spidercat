import json
import math
import time
from pathlib import Path

import numpy as np
import stim
from joblib import Parallel, delayed
from mypy.checkexpr import defaultdict
from scipy.stats import norm

from spidercat.circuit_extraction import make_stim_circ_noisy
from spidercat.utils import qasm_to_stim

cwd = Path.cwd()


def init_data_folder():
    Path(f"{cwd}/simulation_data").mkdir(parents=True, exist_ok=True)


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
                    circuit: stim.Circuit, method: str, num_paths=1):
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
        num_data_errors = np.where(raw_errors > n // 2, n - raw_errors, raw_errors)
        unique_errors, counts = np.unique(num_data_errors, return_counts=True)
        error_counts = dict(zip(unique_errors.tolist(), counts.tolist()))
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

        raw_cnots = [l for (name, l, _) in circuit.flattened_operations() if name == "CX"]
        cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
        layered_cnots = _layer_cnot_circuit(cnots)

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
            "num_cx": len(cnots),
            "depth": len(layered_cnots) + 2,
            "num_flags": num_flags,
            "num_qubits": circuit.num_qubits,
            "num_paths": num_paths,
        }

    return stats


def _layer_cnot_circuit(cnots):
    cnots = [(c, n) for c, n in cnots if isinstance(c, int) and isinstance(n, int)]
    num_qubits = max(map(max, cnots))
    all_qubits = range(num_qubits + 1)
    next_free_layer = {q: 0 for q in all_qubits}
    layers = defaultdict(list)
    for c, n in cnots:
        l = max(next_free_layer[c], next_free_layer[n])
        layers[l].append((c, n))
        next_free_layer[c] = l + 1
        next_free_layer[n] = l + 1
    return list(layers.values())


def add_measurements(circ: stim.Circuit, n: int, method: str):
    if method in ("flag-at-origin"):
        circ.append("M", range(circ.num_qubits - n, circ.num_qubits))
    if method in ("MQT", "spider-cat"):
        circ.append("M", range(n))


def run_simulation(n: int, t: int, p: float, num_samples: int = 1_000_000, method="spider-cat",
                   save_samples: bool = False, num_paths=1):
    circ = load_stim_circuit(t, n, method, num_paths)
    if circ is None:
        return None

    num_flags = circ.num_qubits - n
    noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=2 / 30 * p)
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=0, p_mem=0)
    add_measurements(noisy_circ, n, method)

    # Run the simulation
    circuit_sampler = noisy_circ.compile_sampler()
    samples: np.ndarray = circuit_sampler.sample(num_samples)

    if save_samples:
        save_simulation_data(n, t, samples)

    # Process metrics
    stats = process_samples(samples, num_flags, n, t, p, num_samples, noisy_circ, method, num_paths)

    # Optional: Print summary of counts for quick debugging
    counts_summary = {k: v['count'] for k, v in stats.items()}
    print(f"Stats for {t}-FT {n}-cat (p={p}): {counts_summary}")

    return stats


# 1. Define a helper function that handles a SINGLE 'n'
def process_simulation(n, t, p, num_samples, method='spider-cat', num_paths=1):
    stats_dict = run_simulation(n, t, p, num_samples, method, num_paths=num_paths)
    if stats_dict is None:
        return []
    return list(stats_dict.values())


def simulate_t_n(ts, ns, method='spider-cat', num_paths=1):
    print("Starting simulation loop, varying values of t and n")
    print(f"Method: {method}; Number of paths: {num_paths}")
    # parallel_results = (
    #     process_simulation(n, t, p=0.01, num_samples=1_000_000, method=method, num_paths=num_paths) for t in ts for n in
    #     ns
    # )
    parallel_results = Parallel(n_jobs=-2)(
        delayed(process_simulation)(n, t, p=0.05, num_samples=1_000_000 * min(t, 10), method=method,
                                    num_paths=num_paths) for t in ts for n in ns
    )
    collected_data = [item for sublist in parallel_results for item in sublist]
    with open(f"simulation_data/simulation_results_t_n_{method}-opt_p{num_paths}.json", "w") as f:
        json.dump(collected_data, f, indent=4)
    print("Simulation complete")
    print()


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


if __name__ == "__main__":
    init_data_folder()
    start_time = time.time()

    N = 50
    T = 7
    # simulate_t_n(range(2, T+1), range(8, N + 1), method="spider-cat", num_paths=1)
    simulate_t_n(range(2, T+1), range(10, N + 1), method="spider-cat", num_paths=1)
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

    print("--- %s seconds ---" % (time.time() - start_time))
