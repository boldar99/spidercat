import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import stim

from spidercat.circuit_extraction import extract_circuit_rooted, make_stim_circ_noisy
from spidercat.simulate import add_measurements, process_samples


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


def benchmark_circuit(circ: stim.Circuit, n, num_samples: int = 100_000, p=0.05):
    num_flags = circ.num_qubits - n
    noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=2 / 30 * p)
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=0, p_mem=0)
    add_measurements(noisy_circ, n, "spider-cat")

    # Run the simulation
    circuit_sampler = noisy_circ.compile_sampler()
    samples: np.ndarray = circuit_sampler.sample(num_samples)

    # Process metrics
    stats = process_samples(samples, num_flags, n, -1, p, num_samples, noisy_circ, "spider-cat")

    # Optional: Print summary of counts for quick debugging
    counts_summary = {k: v['probability'] for k, v in stats.items()}
    print(f"Stats for {n}-cat (p={p}): {counts_summary}")

    return counts_summary


def sample_circuit(circ: stim.Circuit, n, num_samples: int = 100_000, p=0.05):
    num_flags = circ.num_qubits - n
    noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=2 / 30 * p)
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=0, p_mem=0)
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
