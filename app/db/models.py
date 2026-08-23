from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class ScanStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class TrainingRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_scan_id: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    capture_session_id: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    sticker_id: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    dataset_source: Mapped[str | None] = mapped_column(String(50), index=True, nullable=True)
    dataset_split: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    image_uri: Mapped[str] = mapped_column(Text)
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    content_type: Mapped[str] = mapped_column(String(50))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    gps_accuracy_meters: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    device_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    app_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    native_qr_success: Mapped[bool] = mapped_column(Boolean, default=False)
    native_qr_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    native_failure_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)

    blur_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    brightness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    brightness_category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    preprocessing_strategy: Mapped[str | None] = mapped_column(String(100), nullable=True)
    qr_decoder_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    ml_prediction: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_prediction: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_charger_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prediction_source: Mapped[str | None] = mapped_column(String(50), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default=ScanStatus.QUEUED.value, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    label: Mapped[ScanLabel | None] = relationship(
        back_populates="scan", cascade="all, delete-orphan", uselist=False
    )


class ScanLabel(Base):
    __tablename__ = "scan_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(
        ForeignKey("scans.id", ondelete="CASCADE"), unique=True, index=True
    )
    correct_qr_payload: Mapped[str] = mapped_column(Text)
    charger_id: Mapped[str] = mapped_column(String(100), index=True)
    confirmation_source: Mapped[str] = mapped_column(String(50))
    confirmed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_status: Mapped[str] = mapped_column(
        String(30), default=ReviewStatus.PENDING.value, index=True
    )
    label_version: Mapped[int] = mapped_column(Integer, default=1)
    training_eligible: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    scan: Mapped[Scan] = relationship(back_populates="label")


class Charger(Base):
    __tablename__ = "chargers"

    charger_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    qr_payload: Mapped[str] = mapped_column(Text, unique=True, index=True)
    latitude: Mapped[float] = mapped_column(Float, index=True)
    longitude: Mapped[float] = mapped_column(Float, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(50), unique=True)
    manifest_path: Mapped[str] = mapped_column(Text)
    example_count: Mapped[int] = mapped_column(Integer)
    grouping_strategy: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(50), unique=True)
    dataset_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    exact_match_accuracy: Mapped[float] = mapped_column(Float)
    character_error_rate: Mapped[float] = mapped_column(Float)
    qr_recovery_rate: Mapped[float] = mapped_column(Float)
    degraded_scan_accuracy: Mapped[float] = mapped_column(Float)
    clean_scan_accuracy: Mapped[float] = mapped_column(Float)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    model_path: Mapped[str] = mapped_column(Text)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(
        String(30), default=TrainingRunStatus.QUEUED.value, index=True
    )
    dry_run: Mapped[bool] = mapped_column(Boolean, default=True)
    dataset_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


Index("ix_chargers_coordinates", Charger.latitude, Charger.longitude)
