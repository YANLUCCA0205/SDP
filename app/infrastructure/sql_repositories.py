"""Persistência em SQLite via SQLAlchemy 2.0 (eventos de ruptura e escala de funcionários).

Datas são gravadas como horário local sem fuso (o fuso é configurável em SDP_TIMEZONE), pois a escala
de trabalho é informada em horário local.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.domain.rupture import RuptureEvent
from app.domain.shift import Shift

TEXT_LENGTH = 120


class Base(DeclarativeBase):
    pass


class RuptureEventRow(Base):
    __tablename__ = "rupture_events"
    __table_args__ = (Index("ix_rupture_open", "camera_id", "product", "ended_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    aisle: Mapped[str] = mapped_column(String(TEXT_LENGTH))
    product: Mapped[str] = mapped_column(String(TEXT_LENGTH))
    category: Mapped[str] = mapped_column(String(TEXT_LENGTH))
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    consecutive_captures: Mapped[int] = mapped_column(Integer)
    people_total: Mapped[int] = mapped_column(Integer, default=0)
    people_samples: Mapped[int] = mapped_column(Integer, default=0)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    employee: Mapped[str | None] = mapped_column(String(TEXT_LENGTH * 3), nullable=True)


class ShiftRow(Base):
    __tablename__ = "shifts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee: Mapped[str] = mapped_column(String(TEXT_LENGTH))
    aisle: Mapped[str] = mapped_column(String(TEXT_LENGTH), index=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime)
    ends_at: Mapped[datetime] = mapped_column(DateTime)


def create_sqlite_engine(url: str) -> Engine:
    # A API atende requisições em threads diferentes (threadpool); cada operação usa sua própria sessão.
    return create_engine(url, connect_args={"check_same_thread": False})


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _event_from_row(row: RuptureEventRow) -> RuptureEvent:
    return RuptureEvent(
        id=row.id,
        camera_id=row.camera_id,
        aisle=row.aisle,
        product=row.product,
        category=row.category,
        started_at=row.started_at,
        ended_at=row.ended_at,
        consecutive_captures=row.consecutive_captures,
        people_total=row.people_total,
        people_samples=row.people_samples,
        confirmed=row.confirmed,
        employee=row.employee,
    )


def _copy_event_to_row(event: RuptureEvent, row: RuptureEventRow) -> None:
    row.camera_id = event.camera_id
    row.aisle = event.aisle
    row.product = event.product
    row.category = event.category
    row.started_at = event.started_at
    row.ended_at = event.ended_at
    row.consecutive_captures = event.consecutive_captures
    row.people_total = event.people_total
    row.people_samples = event.people_samples
    row.confirmed = event.confirmed
    row.employee = event.employee


class SqlEventRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(self, event: RuptureEvent) -> RuptureEvent:
        row = RuptureEventRow()
        _copy_event_to_row(event, row)
        with self._session_factory.begin() as session:
            session.add(row)
            session.flush()
            event.id = row.id
        return event

    def update(self, event: RuptureEvent) -> None:
        with self._session_factory.begin() as session:
            row = session.get(RuptureEventRow, event.id)
            if row is None:
                raise LookupError(f"Evento {event.id} não encontrado")
            _copy_event_to_row(event, row)

    def delete(self, event_id: int) -> None:
        with self._session_factory.begin() as session:
            row = session.get(RuptureEventRow, event_id)
            if row is not None:
                session.delete(row)

    def find_open(self, camera_id: str, product: str) -> RuptureEvent | None:
        statement = (
            select(RuptureEventRow)
            .where(
                RuptureEventRow.camera_id == camera_id,
                RuptureEventRow.product == product,
                RuptureEventRow.ended_at.is_(None),
            )
            .order_by(RuptureEventRow.started_at.desc())
            .limit(1)
        )
        with self._session_factory() as session:
            row = session.scalars(statement).first()
            return _event_from_row(row) if row else None

    def list(
        self,
        *,
        camera_id: str | None = None,
        confirmed_only: bool = True,
        open_only: bool = False,
        limit: int = 100,
    ) -> list[RuptureEvent]:
        statement = select(RuptureEventRow).order_by(RuptureEventRow.started_at.desc()).limit(limit)
        if camera_id is not None:
            statement = statement.where(RuptureEventRow.camera_id == camera_id)
        if confirmed_only:
            statement = statement.where(RuptureEventRow.confirmed.is_(True))
        if open_only:
            statement = statement.where(RuptureEventRow.ended_at.is_(None))
        with self._session_factory() as session:
            return [_event_from_row(row) for row in session.scalars(statement)]


class SqlShiftRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(self, shift: Shift) -> Shift:
        row = ShiftRow(
            employee=shift.employee, aisle=shift.aisle, starts_at=shift.starts_at, ends_at=shift.ends_at
        )
        with self._session_factory.begin() as session:
            session.add(row)
            session.flush()
            shift.id = row.id
        return shift

    def list(self) -> list[Shift]:
        statement = select(ShiftRow).order_by(ShiftRow.starts_at.desc())
        with self._session_factory() as session:
            return [self._to_shift(row) for row in session.scalars(statement)]

    def delete(self, shift_id: int) -> bool:
        with self._session_factory.begin() as session:
            row = session.get(ShiftRow, shift_id)
            if row is None:
                return False
            session.delete(row)
            return True

    def employees_on_duty(self, aisle: str, at: datetime) -> list[str]:
        statement = (
            select(ShiftRow.employee)
            .where(ShiftRow.aisle == aisle, ShiftRow.starts_at <= at, ShiftRow.ends_at > at)
            .order_by(ShiftRow.starts_at)
        )
        with self._session_factory() as session:
            return list(session.scalars(statement))

    @staticmethod
    def _to_shift(row: ShiftRow) -> Shift:
        return Shift(
            id=row.id, employee=row.employee, aisle=row.aisle, starts_at=row.starts_at, ends_at=row.ends_at
        )
