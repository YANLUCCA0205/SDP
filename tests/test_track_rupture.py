import unittest
from datetime import datetime, timedelta

from app.application.track_rupture import TrackRupture
from app.domain.shelf import ShelfEvaluation
from app.domain.shift import Shift
from tests.fakes import T0, InMemoryEventRepository, InMemoryShiftRepository, cafe_profile


class TrackRuptureTest(unittest.TestCase):
    def setUp(self):
        self.events = InMemoryEventRepository()
        self.shifts = InMemoryShiftRepository(
            [Shift("Ana", "C10", datetime(2026, 6, 10, 7), datetime(2026, 6, 10, 15))]
        )
        self.tracker = TrackRupture(self.events, self.shifts, confirmations_required=2)
        self.profile = cafe_profile(capacity=10)  # mínimo = 6

    def capture(self, detected: int, minutes: float, people: int | None = None):
        evaluation = self.profile.evaluate({"cafe": detected}, {"cafe"})
        at = T0 + timedelta(minutes=minutes)
        return self.tracker.execute(self.profile, evaluation, at, people)

    def test_full_lifecycle_start_confirm_end_with_responsible_employee(self):
        first = self.capture(detected=2, minutes=0, people=1)["cafe"]
        self.assertFalse(first.confirmed)
        second = self.capture(detected=1, minutes=0.5, people=3)["cafe"]
        self.assertTrue(second.confirmed)
        self.assertEqual(self.capture(detected=1, minutes=1)["cafe"].consecutive_captures, 3)

        self.assertIsNone(self.capture(detected=7, minutes=47)["cafe"])  # reposto

        [event] = self.events.list(confirmed_only=True)
        self.assertEqual(event.started_at, T0)
        self.assertEqual(event.ended_at, T0 + timedelta(minutes=47))
        self.assertEqual(event.duration_minutes, 47.0)
        self.assertEqual(event.employee, "Ana")
        self.assertEqual(event.average_people, 2.0)  # (1 + 3) / 2 capturas com contagem

    def test_pending_event_is_discarded_when_shelf_recovers(self):
        self.capture(detected=0, minutes=0)
        self.assertEqual(len(self.events.list(confirmed_only=False)), 1)
        self.capture(detected=8, minutes=0.5)
        self.assertEqual(self.events.list(confirmed_only=False), [])

    def test_available_shelf_without_open_event_changes_nothing(self):
        self.assertIsNone(self.capture(detected=9, minutes=0)["cafe"])
        self.assertEqual(self.events.list(confirmed_only=False), [])

    def test_new_event_after_previous_one_was_closed(self):
        self.capture(detected=1, minutes=0)
        self.capture(detected=1, minutes=0.5)
        self.capture(detected=8, minutes=10)
        self.capture(detected=1, minutes=20)
        self.assertEqual(len(self.events.list(confirmed_only=False)), 2)
        self.assertEqual(len(self.events.list(open_only=True, confirmed_only=False)), 1)

    def test_no_employee_when_nobody_is_on_duty(self):
        self.capture(detected=1, minutes=0)
        self.capture(detected=1, minutes=0.5)
        evaluation = self.profile.evaluate({"cafe": 8}, {"cafe"})
        night = datetime(2026, 6, 10, 22, 0)
        self.tracker.execute(self.profile, evaluation, night, None)
        [event] = self.events.list()
        self.assertIsNone(event.employee)

    def test_multiple_employees_on_duty_are_all_recorded(self):
        self.shifts.add(Shift("Bia", "C10", datetime(2026, 6, 10, 7), datetime(2026, 6, 10, 15)))
        self.capture(detected=1, minutes=0)
        self.capture(detected=1, minutes=0.5)
        self.capture(detected=8, minutes=5)
        [event] = self.events.list()
        self.assertEqual(event.employee, "Ana / Bia")

    def test_shift_from_another_aisle_is_ignored(self):
        shifts = InMemoryShiftRepository([Shift("Caio", "C99", datetime(2026, 6, 10, 7), datetime(2026, 6, 10, 15))])
        tracker = TrackRupture(self.events, shifts, confirmations_required=1)
        evaluation = self.profile.evaluate({"cafe": 0}, {"cafe"})
        tracker.execute(self.profile, evaluation, T0, None)
        tracker.execute(self.profile, self.profile.evaluate({"cafe": 9}, {"cafe"}), T0 + timedelta(minutes=3), None)
        [event] = self.events.list()
        self.assertIsNone(event.employee)

    def test_unmonitored_products_produce_no_events(self):
        evaluation = ShelfEvaluation(readings=(), unmonitored=("cafe",))
        self.assertEqual(self.tracker.execute(self.profile, evaluation, T0, None), {})
        self.assertEqual(self.events.list(confirmed_only=False), [])

    def test_rejects_invalid_confirmation_count(self):
        with self.assertRaises(ValueError):
            TrackRupture(self.events, self.shifts, confirmations_required=0)


if __name__ == "__main__":
    unittest.main()
