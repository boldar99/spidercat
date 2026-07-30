from dataclasses import replace
import unittest

import numpy as np
import stim

from spiderstate.circuit_extraction import (
    ExtractionPolicy,
    extract_best_stabiliser_state,
    extract_stabiliser_state,
)
from spiderstate.stabiliser_decomposition import decompose_stabiliser_state


class StabiliserDecompositionContractTests(unittest.TestCase):
    def test_decomposition_is_strictly_valid_and_extraction_reports_resources(self):
        H = np.array([[1, 1, 1, 1]], dtype=np.int8)

        decomposition = decompose_stabiliser_state(H, d=3)
        decomposition.validate(strict_target_alignment=True)
        result = extract_stabiliser_state(
            decomposition,
            policy=ExtractionPolicy(strict_target_alignment=True),
        )

        self.assertEqual(decomposition.num_data_qubits, 4)
        self.assertEqual(len(decomposition.components), 4)
        self.assertEqual(len(decomposition.couplings), 3)
        self.assertEqual(result.resources.data_qubits, 4)
        self.assertEqual(result.resources.total_qubits, result.circuit.num_qubits)
        self.assertEqual(result.resources.ancilla_qubits, 1)
        self.assertEqual(result.resources.two_qubit_gates, 5)
        self.assertEqual(result.resources.measurements, 1)
        self.assertEqual(result.resources.detectors, 1)
        self.assertGreater(result.resources.tick_depth, 0)

    def test_extraction_inputs_are_defensive_copies(self):
        decomposition = decompose_stabiliser_state(
            np.array([[1, 1]], dtype=np.int8),
            d=3,
        )

        graph, _, roots, _, primary_paths = decomposition.extraction_inputs()
        graph.remove_node(0)
        roots.clear()
        primary_paths[0].clear()

        self.assertIn(0, decomposition.graph)
        self.assertTrue(decomposition.roots)
        self.assertTrue(decomposition.primary_paths[0])

    def test_portfolio_selection_uses_measured_circuit_resources(self):
        base = decompose_stabiliser_state(
            np.array([[1, 1]], dtype=np.int8),
            d=3,
        )
        candidates = [
            replace(base, candidate_id="two-cnots"),
            replace(base, candidate_id="one-cnot"),
        ]
        circuits = iter(
            [
                stim.Circuit("R 0 1\nCX 0 1\nCX 1 0"),
                stim.Circuit("R 0 1\nCX 0 1"),
            ]
        )

        class StubExtractor:
            def __init__(self, _builder, verbose=False):
                self.verbose = verbose

            def extract(self, *_args):
                return next(circuits)

        result = extract_best_stabiliser_state(
            candidates,
            builder_factory=object,
            extractor_factory=StubExtractor,
        )

        self.assertEqual(result.candidate_id, "one-cnot")
        self.assertEqual(result.resources.two_qubit_gates, 1)


if __name__ == "__main__":
    unittest.main()
