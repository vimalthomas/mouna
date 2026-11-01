from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    attempts: Mapped[list["Attempt"]] = relationship(back_populates="user")
    sign_progress: Mapped[list["SignProgress"]] = relationship(back_populates="user")


class Sign(Base):
    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gloss: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    attempts: Mapped[list["Attempt"]] = relationship(back_populates="sign")
    progress: Mapped[list["SignProgress"]] = relationship(back_populates="sign")


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id"), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    user: Mapped["User"] = relationship(back_populates="attempts")
    sign: Mapped["Sign"] = relationship(back_populates="attempts")


class SignProgress(Base):
    __tablename__ = "sign_progress"
    __table_args__ = (UniqueConstraint("user_id", "sign_id", name="uq_sign_progress_user_sign"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id"), index=True, nullable=False)
    total_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user: Mapped["User"] = relationship(back_populates="sign_progress")
    sign: Mapped["Sign"] = relationship(back_populates="progress")
