"""Task 1: Verify LaTeX table generation and 95% Wilson intervals."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "spiderstate" / "generate_table.py"


def load_generate_table():
    """Load the table module without its optional simulation dependencies."""
    fake_utils = types.ModuleType("spiderstate.utils")
    fake_utils.load_qecc = lambda *_args, **_kwargs: None
    fake_utils.load_qecc_data = lambda *_args, **_kwargs: None

    spec = importlib.util.spec_from_file_location(
        "generate_table_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"spiderstate.utils": fake_utils}):
        spec.loader.exec_module(module)
    return module


generate_table = load_generate_table()


class GenerateTableTests(unittest.TestCase):
    """Confirm every step of the revised table-generation task."""

    def test_step_1_formats_each_supported_state_label(self):
        # Step 1: Render logical plus, zero strings, and large tensor powers.
        self.assertEqual(
            generate_table.get_state("49_1_5", 1),
            r"$\ket{\overline{+}}$",
        )
        self.assertEqual(
            generate_table.get_state("20_2_6", 2),
            r"$\ket{\overline{00}}$",
        )
        self.assertEqual(
            generate_table.get_state("example", 4),
            r"$\ket{\overline0}^{\otimes 4}$",
        )

    def test_step_2_calculates_wilson_intervals(self):
        # Step 2: Calculate stable 95% confidence bounds.
        low, high = generate_table.wilson_score_interval(0.5, 100)
        self.assertAlmostEqual(low, 0.4038317, places=6)
        self.assertAlmostEqual(high, 0.5961683, places=6)
        self.assertEqual(
            generate_table.wilson_score_interval(0.25, 0),
            (0.25, 0.25),
        )

        for probability in (0.0, 0.5, 1.0):
            with self.subTest(probability=probability):
                low, high = generate_table.wilson_score_interval(probability, 100)
                self.assertGreaterEqual(low, -1e-15)
                self.assertLessEqual(low, probability + 1e-15)
                self.assertGreaterEqual(high, probability - 1e-15)
                self.assertLessEqual(high, 1.0 + 1e-15)

    def test_step_3_defines_valid_baseline_intervals(self):
        # Step 3: Store ordered LER and AR bounds for every baseline.
        expected_codes = {
            "7_1_3",
            "9_1_3",
            "17_1_5",
            "20_2_6",
            "23_1_7",
            "25_1_5",
            "31_1_7",
            "47_1_11",
            "49_1_5",
            "49_1_7",
            "49_1_9",
            "71_1_11",
            "81_1_9",
            "95_1_7",
        }
        self.assertEqual(set(generate_table.BASELINE_DATA), expected_codes)
        self.assertEqual(
            generate_table.BASELINE_DATA["7_1_3"]["ler_bounds"],
            (2.7, 2.9, -5),
        )
        self.assertEqual(
            generate_table.BASELINE_DATA["7_1_3"]["ar_bounds"],
            (0.9783, 0.9784),
        )

        for code, baseline in generate_table.BASELINE_DATA.items():
            with self.subTest(code=code):
                ler_low, ler_high, exponent = baseline["ler_bounds"]
                ar_low, ar_high = baseline["ar_bounds"]
                self.assertGreater(ler_low, 0)
                self.assertLessEqual(ler_low, ler_high)
                self.assertLess(exponent, 0)
                self.assertLessEqual(0, ar_low)
                self.assertLessEqual(ar_low, ar_high)
                self.assertLessEqual(ar_high, 1)

    def test_step_4_renders_sorted_rows_and_missing_values(self):
        # Step 4: Load results, sort strategies, and print the complete table.
        code_data = {
            "5_1_2": {"n": 5, "k": 1, "d": 2, "abbr_name": "test code"},
            "7_1_3": {"n": 7, "k": 1, "d": 3, "abbr_name": "Steane"},
        }
        rows = [
            {
                "code": "7_1_3",
                "strategy": "VolumeOptimizingReuseStrategy",
                "num_samples": 1_000_000,
                "logical_error_rate": 5e-5,
                "acceptance_rate": 0.8,
                "num_cx": 12,
                "num_flags": 3,
                "num_sim_qubits": 8,
                "depth": 10,
            },
            {
                "code": "7_1_3",
                "strategy": "DepthPreservingStrategy",
                "num_samples": 1_000_000,
                "logical_error_rate": 5e-5,
                "acceptance_rate": 0.8,
                "num_cx": 12,
                "num_flags": 3,
                "num_sim_qubits": 9,
                "depth": 7,
            },
            {
                "code": "7_1_3",
                "strategy": "PureAggressiveStrategy",
                "num_samples": 1_000_000,
                "logical_error_rate": 5e-5,
                "acceptance_rate": 0.8,
                "num_cx": 12,
                "num_flags": 3,
                "num_sim_qubits": 7,
                "depth": 12,
            },
            {
                "code": "5_1_2",
                "strategy": "VolumeOptimizingReuseStrategy",
                "num_samples": 0,
                "logical_error_rate": None,
                "acceptance_rate": None,
                "num_cx": 4,
                "num_flags": 1,
                "num_sim_qubits": 5,
                "depth": 4,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "simulation_results"
            results_dir.mkdir()
            for index, row in enumerate(reversed(rows)):
                (results_dir / f"{index}.json").write_text(
                    json.dumps(row),
                    encoding="utf-8",
                )
            (results_dir / "invalid.json").write_text("{", encoding="utf-8")

            output = io.StringIO()
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                with (
                    patch.object(
                        generate_table,
                        "load_qecc_data",
                        side_effect=lambda code, _source: code_data[code],
                    ),
                    redirect_stdout(output),
                ):
                    generate_table.main()
            finally:
                os.chdir(old_cwd)

        latex = output.getvalue()
        self.assertIn(r"\begin{tabular*}{\textwidth}", latex)
        self.assertIn(r"\end{tabular*}", latex)
        self.assertIn(r"\multirow{4}{*}", latex)
        self.assertIn(r"Sim.\@ Qubit", latex)
        self.assertIn("Depth", latex)
        self.assertIn("Volume", latex)
        self.assertIn("using Wilson confidence intervals of 95\\%", latex)
        self.assertIn("Flag at Origin", latex)
        self.assertIn("& Volume & 5 & 4 & - & - \\\\", latex)
        self.assertIn(
            r"$[2.7, \,\, 2.9]\! \times\! 10^{-5}$",
            latex,
        )
        self.assertLess(
            latex.index(r"\code{5, 1, 2}"),
            latex.index(r"\code{7, 1, 3}"),
        )

        ler_low, ler_high = generate_table.wilson_score_interval(5e-5, 800_000)
        expected_ler = (
            f"$[{generate_table.format_float(ler_low / 1e-5, 1)}, \\,\\, "
            f"{generate_table.format_float(ler_high / 1e-5, 1)}]"
            r"\! \times\! 10^{-5}$"
        )
        ar_low, ar_high = generate_table.wilson_score_interval(0.8, 1_000_000)
        expected_ar = f"$[{ar_low:.4f}, \\,\\, {ar_high:.4f}]$"
        self.assertIn(expected_ler, latex)
        self.assertIn(expected_ar, latex)

        qubit_row = latex.index(r"& Sim.\@ Qubit &")
        self.assertLess(qubit_row, latex.index("& Depth &", qubit_row))


if __name__ == "__main__":
    unittest.main()
