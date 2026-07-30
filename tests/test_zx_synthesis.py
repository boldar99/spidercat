"""Tests for phase-free cubic ZX-to-Stim synthesis."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pyzx as zx
from pyzx.utils import EdgeType, VertexType
import stim

from experimental.half_edge import make_four_spider_example
import spiderstate.zx_synthesis as zx_synthesis
from spiderstate.zx_synthesis import (
    HalfEdgeRole,
    SynthesisError,
    SynthesisResult,
    normalize_zx_diagram,
    synthesize_stim,
    synthesize_zx,
)
from spiderstate.stim_utils import make_stim_circ_noisy


def make_ghz_spider(num_outputs: int = 3) -> zx.graph.base.BaseGraph:
    graph = zx.Graph()
    spider = graph.add_vertex(VertexType.Z)
    outputs = []
    for qubit in range(num_outputs):
        boundary = graph.add_vertex(
            VertexType.BOUNDARY, qubit=qubit, row=1
        )
        graph.add_edge((spider, boundary), EdgeType.SIMPLE)
        outputs.append(boundary)
    graph.set_inputs(())
    graph.set_outputs(tuple(outputs))
    return graph


def make_two_spider_state() -> zx.graph.base.BaseGraph:
    graph = zx.Graph()
    left = graph.add_vertex(VertexType.Z)
    right = graph.add_vertex(VertexType.Z)
    graph.add_edge((left, right), EdgeType.HADAMARD)
    outputs = []
    for qubit, spider in enumerate((left, left, right, right)):
        boundary = graph.add_vertex(
            VertexType.BOUNDARY, qubit=qubit, row=1
        )
        graph.add_edge((spider, boundary), EdgeType.SIMPLE)
        outputs.append(boundary)
    graph.set_inputs(())
    graph.set_outputs(tuple(outputs))
    return graph


def make_cycle_state(size: int = 6) -> zx.graph.base.BaseGraph:
    graph = zx.Graph()
    spiders = [
        graph.add_vertex(
            VertexType.X if index % 3 == 1 else VertexType.Z
        )
        for index in range(size)
    ]
    outputs = []
    for index, spider in enumerate(spiders):
        graph.add_edge(
            (spider, spiders[(index + 1) % size]),
            EdgeType.HADAMARD if index % 2 == 0 else EdgeType.SIMPLE,
        )
        boundary = graph.add_vertex(
            VertexType.BOUNDARY, qubit=index, row=1
        )
        graph.add_edge((spider, boundary), EdgeType.SIMPLE)
        outputs.append(boundary)
    graph.set_inputs(())
    graph.set_outputs(tuple(outputs))
    return graph


def make_disconnected_ghz_state() -> zx.graph.base.BaseGraph:
    graph = zx.Graph()
    outputs = []
    for component in range(2):
        spider = graph.add_vertex(
            VertexType.Z if component == 0 else VertexType.X
        )
        for offset in range(3):
            qubit = 3 * component + offset
            boundary = graph.add_vertex(
                VertexType.BOUNDARY, qubit=qubit, row=1
            )
            graph.add_edge(
                (spider, boundary),
                EdgeType.HADAMARD
                if component == 0 and offset == 0
                else EdgeType.SIMPLE,
            )
            outputs.append(boundary)
    graph.set_inputs(())
    graph.set_outputs(tuple(outputs))
    return graph


def target_stabilizers(
    graph: zx.graph.base.BaseGraph,
) -> list[stim.PauliString]:
    tensor = zx.tensor.tensorfy(graph, preserve_scalar=False)
    state = tensor.reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise AssertionError("Test diagram denotes the zero tensor.")
    state = state / norm
    tableau = stim.Tableau.from_state_vector(state, endian="big")
    return [tableau.z_output(q) for q in range(len(graph.outputs()))]


def assert_prepares_target(
    case: unittest.TestCase,
    graph: zx.graph.base.BaseGraph,
    result: SynthesisResult,
    *,
    seeds: range = range(8),
) -> None:
    stabilizers = target_stabilizers(graph)
    output_qubits = [
        result.output_qubits[boundary] for boundary in graph.outputs()
    ]
    for seed in seeds:
        simulator = stim.TableauSimulator(seed=seed)
        simulator.do(result.circuit)
        for stabilizer in stabilizers:
            embedded = stim.PauliString(result.circuit.num_qubits)
            embedded.sign = stabilizer.sign
            for logical, physical in enumerate(output_qubits):
                embedded[physical] = stabilizer[logical]
            case.assertEqual(
                simulator.peek_observable_expectation(embedded),
                1,
                msg=(
                    f"seed={seed}, stabilizer={stabilizer}, "
                    f"circuit=\n{result.circuit}"
                ),
            )


class ZXSynthesisTests(unittest.TestCase):
    def test_normalizes_without_mutating_source_graph(self):
        graph = make_two_spider_state()
        before = graph.to_json()
        normalized = normalize_zx_diagram(graph)

        self.assertEqual(graph.to_json(), before)
        self.assertEqual(len(normalized.spiders), 2)
        self.assertEqual(len(normalized.outputs), 4)
        self.assertEqual(len(normalized.inputs), 0)
        self.assertEqual(len(normalized.edges), 5)
        self.assertTrue(
            any(edge.hadamard for edge in normalized.edges)
        )

    def test_rejects_non_phase_free_or_non_cubic_diagrams(self):
        graph = make_ghz_spider()
        spider = next(
            vertex
            for vertex in graph.vertices()
            if graph.type(vertex) == VertexType.Z
        )
        graph.set_phase(spider, 1)
        with self.assertRaisesRegex(SynthesisError, "nonzero phase"):
            normalize_zx_diagram(graph)

        graph = make_ghz_spider()
        graph.remove_edge(graph.edge(spider, graph.outputs()[0]))
        with self.assertRaisesRegex(SynthesisError, "degree 2"):
            normalize_zx_diagram(graph)

    def test_rejects_undeclared_and_closed_boundaries(self):
        graph = make_ghz_spider()
        graph.set_outputs(graph.outputs()[:-1])
        with self.assertRaisesRegex(SynthesisError, "undeclared boundaries"):
            normalize_zx_diagram(graph)

        graph = zx.Graph()
        a = graph.add_vertex(VertexType.Z)
        b = graph.add_vertex(VertexType.Z)
        c = graph.add_vertex(VertexType.Z)
        d = graph.add_vertex(VertexType.Z)
        for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            graph.add_edge((u, v), EdgeType.HADAMARD)
        graph.set_inputs(())
        graph.set_outputs(())
        with self.assertRaisesRegex(SynthesisError, "scalar-only"):
            normalize_zx_diagram(graph)

    def test_ghz_state_uses_only_expected_stim_operations(self):
        graph = make_ghz_spider()
        result = synthesize_zx(graph, optimizer="heuristic")

        self.assertEqual(result.metrics.cx_count, 2)
        self.assertEqual(result.metrics.measurement_count, 0)
        self.assertEqual(result.metrics.num_qubits, 3)
        self.assertEqual(result.metrics.peak_qubits, 3)
        self.assertEqual(result.detectors, ())
        self.assertEqual(
            set(result.output_qubits.values()), {0, 1, 2}
        )
        names = {
            instruction.name
            for instruction in result.circuit
            if instruction.name != "TICK"
        }
        self.assertLessEqual(names, {"R", "RX", "H", "CX", "M", "MX", "DETECTOR"})
        self.assertEqual(synthesize_stim(graph), synthesize_zx(graph).circuit)
        assert_prepares_target(self, graph, result)

    def test_hadamard_contraction_feedback_prepares_every_branch(self):
        graph = make_two_spider_state()
        for strategy in ("gate_count", "depth"):
            with self.subTest(strategy=strategy):
                result = synthesize_zx(
                    graph,
                    strategy=strategy,
                    optimizer="heuristic",
                    seed=4,
                )
                self.assertEqual(result.metrics.measurement_count, 2)
                self.assertTrue(
                    any(
                        measurement.kind == "teleportation"
                        for measurement in result.measurements
                    )
                )
                self.assertTrue(
                    any(
                        measurement.correction
                        for measurement in result.measurements
                    )
                )
                assert_prepares_target(self, graph, result)
                # Random teleportation outcomes are corrected, not mislabeled
                # as deterministic detectors.
                self.assertEqual(result.metrics.detector_count, 0)

    def test_terminal_frames_and_disconnected_components(self):
        graph = make_disconnected_ghz_state()
        result = synthesize_zx(
            graph, optimizer="heuristic", seed=12
        )

        self.assertEqual(len(result.output_qubits), 6)
        self.assertEqual(set(result.output_qubits.values()), set(range(6)))
        assert_prepares_target(self, graph, result, seeds=range(3))

    def test_strategies_expose_expected_tradeoff(self):
        graph = make_cycle_state(4)
        gate_result = synthesize_zx(
            graph, "gate_count", optimizer="heuristic", seed=2
        )
        depth_result = synthesize_zx(
            graph, "depth", optimizer="heuristic", seed=2
        )

        self.assertLessEqual(
            gate_result.metrics.gate_count_key,
            depth_result.metrics.gate_count_key,
        )
        self.assertLessEqual(
            depth_result.metrics.depth_key,
            gate_result.metrics.depth_key,
        )
        self.assertLess(
            gate_result.metrics.num_qubits,
            depth_result.metrics.num_qubits,
        )
        self.assertLess(
            depth_result.metrics.depth,
            gate_result.metrics.depth,
        )
        assert_prepares_target(self, graph, gate_result, seeds=range(3))
        assert_prepares_target(self, graph, depth_result, seeds=range(3))

    def test_exact_and_heuristic_layouts_are_deterministic(self):
        graph = make_two_spider_state()
        first = synthesize_zx(
            graph, optimizer="heuristic", seed=91
        )
        second = synthesize_zx(
            graph, optimizer="heuristic", seed=91
        )
        self.assertEqual(first.circuit, second.circuit)
        self.assertEqual(first.half_edge_roles, second.half_edge_roles)
        for roles in first.half_edge_roles.values():
            self.assertEqual(set(roles.values()), set(HalfEdgeRole))

        exact = synthesize_zx(
            graph,
            optimizer="exact",
            timeout_seconds=2,
        )
        self.assertIn(
            exact.optimizer_status.split("+")[0],
            {"optimal", "feasible_timeout"},
        )

    def test_large_diagram_uses_documented_heuristic_fallback(self):
        graph = make_cycle_state(7)
        result = synthesize_zx(
            graph,
            exact_max_spiders=2,
            timeout_seconds=1,
        )
        self.assertTrue(
            result.optimizer_status.startswith("heuristic_size_limit")
        )
        self.assertFalse(result.proven_optimal)

    def test_exact_timeout_uses_deterministic_heuristic_fallback(self):
        graph = make_two_spider_state()
        with mock.patch.object(
            zx_synthesis, "_solve_exact_layout", return_value=None
        ):
            first = synthesize_zx(graph, seed=17)
            second = synthesize_zx(graph, seed=17)

        self.assertTrue(
            first.optimizer_status.startswith(
                "heuristic_after_exact_timeout"
            )
        )
        self.assertFalse(first.proven_optimal)
        self.assertEqual(first.circuit, second.circuit)

    def test_attached_open_diagram_round_trips_as_unitary(self):
        graph = make_four_spider_example()
        for strategy in ("gate_count", "depth"):
            with self.subTest(strategy=strategy):
                result = synthesize_zx(
                    graph, strategy, optimizer="heuristic"
                )
                self.assertEqual(result.input_qubits, {4: 0, 5: 1})
                self.assertEqual(result.output_qubits, {6: 0, 7: 1})
                self.assertEqual(result.metrics.preparation_count, 0)
                self.assertEqual(result.metrics.measurement_count, 0)
                self.assertTrue(
                    result.optimizer_status.endswith("pyzx_gflow")
                )

                tensor = zx.tensor.tensorfy(
                    graph, preserve_scalar=False
                )
                expected = zx.tensor.tensor_to_matrix(tensor, 2, 2)
                expected /= np.linalg.norm(expected[:, 0])
                actual = result.circuit.to_tableau().to_unitary_matrix(
                    endian="big"
                )
                pivot = np.unravel_index(
                    np.argmax(np.abs(expected)), expected.shape
                )
                phase = expected[pivot] / actual[pivot]
                np.testing.assert_allclose(
                    expected, phase * actual, atol=1e-6
                )

    def test_stim_text_round_trip_and_noiseless_detectors(self):
        graph = make_cycle_state(4)
        result = synthesize_zx(
            graph, "depth", optimizer="heuristic", seed=5
        )
        reparsed = stim.Circuit(str(result.circuit))
        self.assertEqual(reparsed, result.circuit)
        detector_samples = reparsed.compile_detector_sampler().sample(32)
        self.assertFalse(np.any(detector_samples))
        self.assertEqual(
            len(result.detectors), result.metrics.detector_count
        )

    def test_existing_noise_helper_preserves_feedback_records(self):
        result = synthesize_zx(
            make_two_spider_state(), optimizer="heuristic", seed=5
        )
        noisy, measurement_mapping = make_stim_circ_noisy(
            result.circuit, 0.001
        )

        self.assertEqual(measurement_mapping, {0: 0, 1: 1})
        self.assertIn("CX rec[-", str(noisy))
        noisy.compile_sampler().sample(4)


if __name__ == "__main__":
    unittest.main()
