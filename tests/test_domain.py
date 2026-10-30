import unittest
from datetime import datetime, timedelta

from app.domain.flow import CustomerFlow, FlowThresholds
from app.domain.rupture import RuptureEvent
from app.domain.shelf import ProductSlot, ShelfProfile, ShelfState
from app.domain.shift import Shift
from tests.fakes import T0


class ProductSlotTest(unittest.TestCase):
    def test_required_units_is_ceil_of_capacity_times_occupancy(self):
        self.assertEqual(ProductSlot("a", "c", 10).required_units(0.6), 6)
        self.assertEqual(ProductSlot("a", "c", 8).required_units(0.6), 5)
        self.assertEqual(ProductSlot("a", "c", 3).required_units(0.6), 2)

    def test_float_noise_does_not_inflate_requirement(self):
        # 10 * 0.7 == 7.000000000000001 em ponto flutuante; o mínimo correto é 7, não 8.
        self.assertEqual(ProductSlot("a", "c", 10).required_units(0.7), 7)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            ProductSlot("a", "c", 0)
        with self.assertRaises(ValueError):
            ProductSlot("  ", "c", 5)


class ShelfProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = ShelfProfile(
            "cam-01", "C10", (ProductSlot("cafe", "M", 10), ProductSlot("arroz", "M", 5)), 0.6
        )

    def test_available_when_count_reaches_minimum(self):
        evaluation = self.profile.evaluate({"cafe": 6, "arroz": 3}, {"cafe", "arroz"})
        self.assertTrue(all(r.state is ShelfState.DISPONIVEL for r in evaluation.readings))

    def test_possible_rupture_when_below_minimum(self):
        evaluation = self.profile.evaluate({"cafe": 5, "arroz": 3}, {"cafe", "arroz"})
        states = {r.product: r.state for r in evaluation.readings}
        self.assertIs(states["cafe"], ShelfState.POSSIVEL_RUPTURA)
        self.assertIs(states["arroz"], ShelfState.DISPONIVEL)

    def test_missing_class_counts_as_zero(self):
        evaluation = self.profile.evaluate({}, {"cafe", "arroz"})
        self.assertTrue(all(r.in_rupture and r.detected == 0 for r in evaluation.readings))

    def test_classes_unknown_to_the_model_are_not_evaluated(self):
        evaluation = self.profile.evaluate({}, {"person", "car"})
        self.assertEqual(evaluation.readings, ())
        self.assertEqual(set(evaluation.unmonitored), {"cafe", "arroz"})

    def test_rejects_invalid_profile(self):
        with self.assertRaises(ValueError):
            ShelfProfile("c", "A", (ProductSlot("a", "c", 1),), 0)
        with self.assertRaises(ValueError):
            ShelfProfile("c", "A", (), 0.6)


class FlowThresholdsTest(unittest.TestCase):
    def test_classification(self):
        thresholds = FlowThresholds(low_max=1.0, medium_max=3.0)
        self.assertIs(thresholds.classify(0), CustomerFlow.BAIXO)
        self.assertIs(thresholds.classify(1.0), CustomerFlow.BAIXO)
        self.assertIs(thresholds.classify(2.5), CustomerFlow.MEDIO)
        self.assertIs(thresholds.classify(3.01), CustomerFlow.ALTO)

    def test_rejects_inverted_limits(self):
        with self.assertRaises(ValueError):
            FlowThresholds(low_max=5, medium_max=2)


class RuptureEventTest(unittest.TestCase):
    def open_event(self, confirmations_required=2, people=None):
        return RuptureEvent.open(
            camera_id="cam-01", aisle="C10", product="cafe", category="M",
            at=T0, people=people, confirmations_required=confirmations_required,
        )

    def test_starts_pending_and_confirms_after_consecutive_captures(self):
        event = self.open_event(confirmations_required=3)
        self.assertFalse(event.confirmed)
        event.register_capture(None, 3)
        self.assertFalse(event.confirmed)
        event.register_capture(None, 3)
        self.assertTrue(event.confirmed)

    def test_single_confirmation_confirms_immediately(self):
        self.assertTrue(self.open_event(confirmations_required=1).confirmed)

    def test_duration_is_difference_between_start_and_end(self):
        event = self.open_event(confirmations_required=1)
        self.assertIsNone(event.duration_minutes)
        event.close(T0 + timedelta(minutes=47, seconds=30), "Ana")
        self.assertEqual(event.duration_minutes, 47.5)
        self.assertEqual(event.employee, "Ana")
        self.assertFalse(event.is_open)

    def test_pending_event_cannot_be_closed(self):
        with self.assertRaises(ValueError):
            self.open_event(confirmations_required=2).close(T0 + timedelta(minutes=1), None)

    def test_end_cannot_precede_start(self):
        with self.assertRaises(ValueError):
            self.open_event(confirmations_required=1).close(T0 - timedelta(seconds=1), None)

    def test_average_people_ignores_captures_without_measurement(self):
        event = self.open_event(people=2)
        event.register_capture(None, 2)
        event.register_capture(4, 2)
        self.assertEqual(event.average_people, 3.0)

    def test_average_people_is_none_when_never_measured(self):
        self.assertIsNone(self.open_event(people=None).average_people)


class ShiftTest(unittest.TestCase):
    def test_covers_is_half_open_interval(self):
        shift = Shift("Ana", "C10", datetime(2026, 6, 10, 8), datetime(2026, 6, 10, 16))
        self.assertTrue(shift.covers(datetime(2026, 6, 10, 8)))
        self.assertTrue(shift.covers(datetime(2026, 6, 10, 15, 59)))
        self.assertFalse(shift.covers(datetime(2026, 6, 10, 16)))

    def test_rejects_invalid_interval_and_blank_fields(self):
        with self.assertRaises(ValueError):
            Shift("Ana", "C10", datetime(2026, 6, 10, 8), datetime(2026, 6, 10, 8))
        with self.assertRaises(ValueError):
            Shift(" ", "C10", datetime(2026, 6, 10, 8), datetime(2026, 6, 10, 9))


if __name__ == "__main__":
    unittest.main()
