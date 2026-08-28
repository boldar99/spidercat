import unittest

from spiderstate.correlated_frame_decoder import analyze_correlated_frame_decoder


class CorrelatedFrameDecoderTest(unittest.TestCase):
    def test_exact_joint_syndromes_decode_steane_and_color_codes(self):
        for code in ("7_1_3", "17_1_5"):
            with self.subTest(code=code):
                report = analyze_correlated_frame_decoder(code, method="FAO")
                self.assertTrue(report.transversal_s_preserves_code)
                self.assertTrue(report.logically_decodable)
                self.assertTrue(report.exactly_resettable)

    def test_readout_errors_preserve_logical_distinguishability_but_not_exact_reset(self):
        for code in ("7_1_3", "17_1_5"):
            with self.subTest(code=code):
                report = analyze_correlated_frame_decoder(
                    code,
                    method="FAO",
                    include_measurement_errors=True,
                )
                self.assertTrue(report.logically_decodable)
                self.assertFalse(report.exactly_resettable)
                self.assertGreater(report.exact_reset_ambiguity_records, 0)


if __name__ == "__main__":
    unittest.main()
