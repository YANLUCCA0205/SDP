import unittest

from app.infrastructure.visual_history import _area_summary, _box_area


class VisualHistoryAreaTest(unittest.TestCase):
    def test_box_area_uses_width_times_height(self):
        self.assertEqual(_box_area([10, 20, 40, 60]), 1200)

    def test_box_area_is_zero_for_degenerate_box(self):
        self.assertEqual(_box_area([10, 20, 10, 60]), 0)

    def test_area_summary_reports_occupancy_and_missing_percentages(self):
        self.assertEqual(
            _area_summary(1000, 325),
            {
                "area_referencia_px": 1000,
                "area_atual_estimada_px": 675,
                "area_ausente_px": 325,
                "ocupacao_percentual": 67.5,
                "ruptura_percentual": 32.5,
            },
        )

    def test_area_summary_caps_missing_area_at_reference(self):
        summary = _area_summary(100, 150)
        self.assertEqual(summary["area_atual_estimada_px"], 0)
        self.assertEqual(summary["ruptura_percentual"], 100.0)


if __name__ == "__main__":
    unittest.main()
