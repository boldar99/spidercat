import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import stim

import spiderstate.cat_at_origin as pipeline


def _component_graph(label: str):
    graph = nx.Graph()
    graph.add_node(0, is_mark=False, label=f"{label}-root")
    graph.add_node(1, is_mark=True, label=f"{label}-matched")
    graph.add_node(2, is_mark=False, label=f"{label}-main")
    graph.add_edges_from([(0, 1), (0, 2)])

    forest = graph.copy()
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(graph.nodes(data=True))
    dependency_graph.add_edges_from([(0, 1), (0, 2)])
    return graph, forest, {0: 0}, dependency_graph, 2


def _assert_stabilized(test_case, circuit, observables):
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)

    for observable in observables:
        padded = observable + "I" * (circuit.num_qubits - len(observable))
        expectation = simulator.peek_observable_expectation(stim.PauliString(padded))
        test_case.assertEqual(
            expectation,
            1,
            msg=f"{observable} is not a +1 stabilizer of the prepared state",
        )


class StabiliserStateAssemblyTests(unittest.TestCase):
    def test_assembles_global_graphs_before_circuit_extraction(self):
        H = np.array([[1, 1]], dtype=np.int8)
        z_component = _component_graph("z")
        x_component = _component_graph("x")
        extracted = stim.Circuit("H 0")
        builder = object()

        with (
            patch.object(pipeline, "has_unique_ones_property", return_value=True),
            patch.object(
                pipeline,
                "find_pivots_in_matrix",
                return_value=({0: 0}, []),
            ),
            patch.object(
                pipeline,
                "well_ordered_ft_cat_state_data",
                side_effect=[z_component, x_component],
            ) as decompose,
            patch.object(
                pipeline,
                "match_edges",
                return_value=[((0, 0), (1, 1))],
            ) as match,
            patch.object(pipeline, "StimBuilder", return_value=builder) as builder_type,
            patch.object(pipeline, "CatStateExtractor") as extractor_type,
        ):
            extractor = extractor_type.return_value
            extractor.extract.return_value = extracted

            result = pipeline.cat_at_origin(H, d=3)

        self.assertIs(result, extracted)
        self.assertEqual(
            [(int(call.args[0]), call.args[1]) for call in decompose.call_args_list],
            [(2, 1), (2, 1)],
        )

        match_args = match.call_args.args
        np.testing.assert_array_equal(match_args[0], H)
        self.assertEqual(match_args[1], [1])
        self.assertEqual(match_args[4], [[1]])
        self.assertEqual(match_args[5], [[1]])

        builder_type.assert_called_once_with()
        extractor_type.assert_called_once_with(builder, verbose=False)
        global_graph, global_forest, roots, dependency_graph, primary_paths = (
            extractor.extract.call_args.args
        )

        self.assertEqual(set(global_graph.nodes), set(range(6)))
        self.assertEqual(set(global_forest.nodes), set(global_graph.nodes))
        self.assertEqual(set(dependency_graph.nodes), set(global_graph.nodes))
        self.assertEqual(
            {global_graph.nodes[q]["spider_type"] for q in range(3)},
            {"Z"},
        )
        self.assertEqual(
            {global_graph.nodes[q]["spider_type"] for q in range(3, 6)},
            {"X"},
        )

        self.assertEqual(global_graph.edges[1, 4]["edge_type"], "cnot")
        self.assertFalse(global_graph.nodes[1]["is_mark"])
        self.assertFalse(global_graph.nodes[4]["is_mark"])
        self.assertTrue(global_graph.nodes[1]["is_flag"])
        self.assertTrue(global_graph.nodes[4]["is_flag"])

        self.assertEqual(
            {frozenset(edge) for edge in global_forest.edges},
            {
                frozenset((0, 1)),
                frozenset((0, 2)),
                frozenset((3, 4)),
                frozenset((3, 5)),
            },
        )
        self.assertEqual(roots, {0: 0, 1: 3})
        self.assertEqual(primary_paths, {0: [0, 2], 1: [3, 5]})
        self.assertEqual(
            set(dependency_graph.edges),
            {(0, 1), (0, 2), (3, 4), (3, 5), (0, 4), (3, 1)},
        )
        self.assertTrue(nx.is_directed_acyclic_graph(dependency_graph))

    def test_rejects_matrix_without_a_unique_pivot_per_row(self):
        H = np.array([[1, 1], [1, 1]], dtype=np.int8)

        with self.assertRaisesRegex(ValueError, "bipartite graph state"):
            pipeline.cat_at_origin(H, d=3)

    def test_prepares_expected_small_css_states(self):
        cases = [
            (
                np.array([[1, 1]], dtype=np.int8),
                ["XX", "ZZ"],
            ),
            (
                np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
                ["XXI", "IXX", "ZZZ"],
            ),
        ]

        for H, stabilizers in cases:
            with self.subTest(H=H.tolist()):
                circuit = pipeline.cat_at_origin(H, d=3)
                self.assertIsInstance(circuit, stim.Circuit)
                self.assertGreater(len(circuit), 0)
                _assert_stabilized(self, circuit, stabilizers)

    def test_flagged_ghz_has_silent_detector_and_phase_free_output(self):
        H = np.array([[1, 1, 1, 1]], dtype=np.int8)

        circuit = pipeline.cat_at_origin(H, d=3)

        self.assertGreater(circuit.num_detectors, 0)
        detector_samples = circuit.compile_detector_sampler().sample(shots=32)
        self.assertFalse(np.any(detector_samples))
        self.assertTrue(
            {name for name, _, _ in circuit.flattened_operations()}
            <= {"R", "RX", "CX", "M", "MX", "DETECTOR", "TICK"}
        )
        _assert_stabilized(self, circuit, ["XXXX", "ZZII", "IZZI", "IIZZ"])

    def test_flagged_ghz_passes_exact_single_fault_verification(self):
        from spiderstate.fault_tolerance_verification import verify_ftsp

        H_x = np.array([[1, 1, 1, 1]], dtype=np.int8)
        H_z = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int8,
        )
        no_logicals = np.empty((0, 4), dtype=np.int8)
        circuit = pipeline.cat_at_origin(H_x, d=3)

        result = verify_ftsp(
            circuit,
            H_primary=H_z,
            L_primary=no_logicals,
            H_conjugate=H_x,
            L_conjugate=no_logicals,
            d=3,
            t=1,
        )

        self.assertIs(result, True)


if __name__ == "__main__":
    unittest.main()
