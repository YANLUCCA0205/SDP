"""Integração com SQLite real. Requer sqlalchemy (pulado se ausente)."""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

try:
    from app.infrastructure.sql_repositories import (
        SqlEventRepository,
        SqlShiftRepository,
        create_session_factory,
        create_sqlite_engine,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    SqlEventRepository = None
    SKIP_REASON = f"dependência ausente: {exc.name}"
else:
    SKIP_REASON = ""

from app.application.track_rupture import TrackRupture
from app.domain.rupture import RuptureEvent
from app.domain.shift import Shift
from tests.fakes import T0, cafe_profile


@unittest.skipIf(SqlEventRepository is None, SKIP_REASON)
class SqlRepositoriesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = create_sqlite_engine(f"sqlite:///{Path(tmp.name) / 'test.db'}")
        self.addCleanup(engine.dispose)
        factory = create_session_factory(engine)
        self.events = SqlEventRepository(factory)
        self.shifts = SqlShiftRepository(factory)

    def new_event(self, product="cafe", confirmations=2, at=T0):
        return RuptureEvent.open(
            camera_id="cam-01", aisle="C10", product=product, category="M",
            at=at, people=2, confirmations_required=confirmations,
        )

    def test_event_roundtrip_preserves_all_fields(self):
        saved = self.events.add(self.new_event(confirmations=1))
        self.assertIsNotNone(saved.id)
        saved.close(T0 + timedelta(minutes=5), "Ana")
        self.events.update(saved)
        [loaded] = self.events.list()
        self.assertEqual(loaded, saved)

    def test_find_open_ignores_closed_events_and_other_products(self):
        closed = self.events.add(self.new_event(confirmations=1))
        closed.close(T0 + timedelta(minutes=1), None)
        self.events.update(closed)
        self.events.add(self.new_event(product="arroz"))
        self.assertIsNone(self.events.find_open("cam-01", "cafe"))
        self.assertEqual(self.events.find_open("cam-01", "arroz").product, "arroz")

    def test_list_filters_pending_open_and_camera(self):
        self.events.add(self.new_event(confirmations=2))  # pendente
        self.events.add(self.new_event(product="arroz", confirmations=1, at=T0 + timedelta(minutes=1)))
        self.assertEqual([e.product for e in self.events.list()], ["arroz"])
        self.assertEqual(len(self.events.list(confirmed_only=False)), 2)
        self.assertEqual(self.events.list(camera_id="outra"), [])
        self.assertEqual(len(self.events.list(open_only=True, confirmed_only=False)), 2)

    def test_delete_event_and_update_of_missing_event(self):
        saved = self.events.add(self.new_event())
        self.events.delete(saved.id)
        self.assertEqual(self.events.list(confirmed_only=False), [])
        with self.assertRaises(LookupError):
            self.events.update(saved)

    def test_shifts_crud_and_duty_lookup(self):
        shift = self.shifts.add(Shift("Ana", "C10", datetime(2026, 6, 10, 7), datetime(2026, 6, 10, 15)))
        self.shifts.add(Shift("Caio", "C11", datetime(2026, 6, 10, 7), datetime(2026, 6, 10, 15)))
        self.assertEqual(self.shifts.employees_on_duty("C10", datetime(2026, 6, 10, 9)), ["Ana"])
        self.assertEqual(self.shifts.employees_on_duty("C10", datetime(2026, 6, 10, 15)), [])
        self.assertEqual(len(self.shifts.list()), 2)
        self.assertTrue(self.shifts.delete(shift.id))
        self.assertFalse(self.shifts.delete(shift.id))

    def test_tracker_works_end_to_end_on_sqlite(self):
        self.shifts.add(Shift("Ana", "C10", datetime(2026, 6, 10, 7), datetime(2026, 6, 10, 15)))
        tracker = TrackRupture(self.events, self.shifts, confirmations_required=2)
        profile = cafe_profile(capacity=10)

        def capture(detected, minutes):
            evaluation = profile.evaluate({"cafe": detected}, {"cafe"})
            return tracker.execute(profile, evaluation, T0 + timedelta(minutes=minutes), 1)

        capture(1, 0)
        capture(1, 0.5)
        capture(9, 12)
        [event] = self.events.list()
        self.assertEqual((event.duration_minutes, event.employee), (12.0, "Ana"))


if __name__ == "__main__":
    unittest.main()
