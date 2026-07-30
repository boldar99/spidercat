"""Task 2: Verify cached large-code simulation data and metadata."""

import csv
import hashlib
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "spiderstate" / "simulation_results"

EXPECTED_RESULTS = (
    ("49_1_5", "DepthPreservingStrategy", "ae4858c845b9481a"),
    ("49_1_5", "PureAggressiveStrategy", "3b635635efce8a93"),
    ("71_1_11", "DepthPreservingStrategy", "1bf89c0f90a2cce2"),
    ("71_1_11", "PureAggressiveStrategy", "526b210be5eed802"),
    ("81_1_9", "DepthPreservingStrategy", "36053276369cdb8e"),
    ("81_1_9", "PureAggressiveStrategy", "6ca68c9162b83da8"),
    ("95_1_7", "DepthPreservingStrategy", "d25a0bdcfcb8392c"),
    ("95_1_7", "PureAggressiveStrategy", "c186381787210f36"),
)


def load_result(code, strategy, circuit_hash):
    path = RESULTS_DIR / f"{code}_{strategy}_{circuit_hash}.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class SimulationResultTests(unittest.TestCase):
    """Confirm every step of the large-code simulation-data task."""

    def test_step_1_has_both_strategies_for_each_large_code(self):
        # Step 1: Provide complete metadata for four codes and two strategies.
        for code, strategy, circuit_hash in EXPECTED_RESULTS:
            with self.subTest(code=code, strategy=strategy):
                json_path = RESULTS_DIR / f"{code}_{strategy}_{circuit_hash}.json"
                csv_path = RESULTS_DIR / f"{code}_{circuit_hash}.csv"
                self.assertTrue(json_path.is_file())
                self.assertTrue(csv_path.is_file())

                result = load_result(code, strategy, circuit_hash)
                self.assertEqual(result["code"], code)
                self.assertEqual(result["strategy"], strategy)
                self.assertEqual(result["circuit_hash"], circuit_hash)
                self.assertEqual(result["p"], 0.001)
                self.assertGreater(result["num_samples"], 0)
                self.assertGreater(result["num_cx"], 0)
                self.assertGreater(result["num_flags"], 0)

    def test_step_2_csv_batches_reproduce_cached_totals(self):
        # Step 2: Match every cached batch to its JSON summary.
        expected_header = [
            "total_shots",
            "num_flagged",
            "num_discarded",
            "num_incorrect",
        ]

        for code, strategy, circuit_hash in EXPECTED_RESULTS:
            with self.subTest(code=code, strategy=strategy):
                result = load_result(code, strategy, circuit_hash)
                csv_path = RESULTS_DIR / f"{code}_{circuit_hash}.csv"
                with csv_path.open(newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    self.assertEqual(reader.fieldnames, expected_header)
                    rows = list(reader)

                self.assertEqual(len(rows), 1_000)
                self.assertEqual(
                    sum(int(row["total_shots"]) for row in rows),
                    result["num_samples"],
                )
                self.assertEqual(
                    sum(int(row["num_flagged"]) for row in rows),
                    result["total_flagged"],
                )
                self.assertEqual(
                    sum(int(row["num_discarded"]) for row in rows),
                    result["total_discarded"],
                )
                incorrect = sum(
                    int(row["num_incorrect"])
                    for row in rows
                    if row["num_incorrect"]
                )
                if result["total_incorrect"] is None:
                    self.assertFalse(any(row["num_incorrect"] for row in rows))
                else:
                    self.assertEqual(incorrect, result["total_incorrect"])

    def test_step_3_recomputes_rates_volumes_and_hashes(self):
        # Step 3: Recompute every derived value from its source fields.
        for code, strategy, circuit_hash in EXPECTED_RESULTS:
            with self.subTest(code=code, strategy=strategy):
                result = load_result(code, strategy, circuit_hash)
                samples = result["num_samples"]
                raw_acceptance = 1 - result["total_flagged"] / samples
                acceptance = (
                    samples
                    - result["total_flagged"]
                    - result["total_discarded"]
                ) / samples
                volume = result["depth"] * result["num_sim_qubits"]
                digest = hashlib.sha256(
                    result["noisy_circuit"].encode()
                ).hexdigest()[:16]

                self.assertAlmostEqual(result["raw_acceptance_rate"], raw_acceptance)
                self.assertAlmostEqual(result["acceptance_rate"], acceptance)
                self.assertEqual(result["circuit_volume"], volume)
                self.assertEqual(
                    result["expected_circuit_volume"],
                    int(volume / acceptance),
                )
                self.assertEqual(digest, circuit_hash)
                self.assertTrue(result["perfect_stim"])
                self.assertTrue(result["noisy_circuit"])

    def test_step_4_preserves_the_strategy_tradeoff(self):
        # Step 4: Aggressive reuse saves qubits while preserving CX and flags.
        for code in ("49_1_5", "71_1_11", "81_1_9", "95_1_7"):
            matching = [item for item in EXPECTED_RESULTS if item[0] == code]
            by_strategy = {
                strategy: load_result(code, strategy, circuit_hash)
                for _, strategy, circuit_hash in matching
            }
            depth = by_strategy["DepthPreservingStrategy"]
            aggressive = by_strategy["PureAggressiveStrategy"]

            with self.subTest(code=code):
                self.assertLess(
                    aggressive["num_sim_qubits"],
                    depth["num_sim_qubits"],
                )
                self.assertGreater(aggressive["depth"], depth["depth"])
                self.assertEqual(aggressive["num_cx"], depth["num_cx"])
                self.assertEqual(aggressive["num_flags"], depth["num_flags"])

    def test_step_5_corrects_the_71_qubit_code_label(self):
        # Step 5: Identify [[71, 1, 11]] as a color code.
        path = REPO_ROOT / "spiderstate" / "qeccs" / "FAO" / "71_1_11.json"
        with path.open(encoding="utf-8") as file:
            code = json.load(file)

        self.assertEqual(code["abbr_name"], "color code")
        self.assertEqual((code["n"], code["k"], code["d"]), (71, 1, 11))


if __name__ == "__main__":
    unittest.main()
