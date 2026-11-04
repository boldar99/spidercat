import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import stim

from spidercat.circuit_extraction import extract_circuit_rooted
from spidercat.simulate import SimulationInfo, add_measurements, run_simulation
from spidercat.utils import make_stim_circ_noisy


def benchmark_solution(
        G: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        matches: dict[int, list[tuple[int, int]]] | None = None,
        roots: dict[int, int] | None = None,
        num_samples: int = 100_000,
        p=0.05
):
    n = sum(markings.values())
    circ = extract_circuit_rooted(G, forest, roots, markings, matches, verbose=False)
    return benchmark_circuit(circ, n, num_samples, p)


def benchmark_circuit(circ: stim.Circuit, n, num_samples: int = 100_000, p=0.05, method="spider-cat"):
    num_flags = circ.num_qubits - n
    noisy_circ, _ = make_stim_circ_noisy(circ, p_1=p, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    add_measurements(noisy_circ, n, method)

    sim_info = SimulationInfo(
        circuit=noisy_circ,
        circuit_hash="",
        method=method,
        num_samples=num_samples,
        num_flags=num_flags,
        p=p,
        n=n,
        t=-1,
        basis="Z",
    )
    _, num_accepted, error_counts = run_simulation(sim_info)
    counts_summary = {k: count / num_accepted for k, count in error_counts.items()} if num_accepted > 0 else {}
    print(f"Stats for {n}-cat (p={p}): {counts_summary}")

    return counts_summary


def sample_circuit(circ: stim.Circuit, n, num_samples: int = 100_000, p=0.05):
    noisy_circ, _ = make_stim_circ_noisy(circ, p_1=p, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    add_measurements(noisy_circ, n, "spider-cat")

    # Run the simulation
    circuit_sampler = noisy_circ.compile_sampler()
    samples: np.ndarray = circuit_sampler.sample(num_samples)

    # Process metrics
    converter = noisy_circ.compile_m2d_converter()
    detectors = converter.convert(measurements=samples, append_observables=False)
    error_detected = np.any(detectors, axis=1)
    post_selected_samples = samples[~error_detected, -n:]

    return post_selected_samples
