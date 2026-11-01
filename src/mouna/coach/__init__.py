"""Teach/correct coaching package: scores a live attempt against a known
target sign (closed-set comparison), as opposed to the open-set
classification the rest of the app does."""

from mouna.coach.compare import score_attempt

__all__ = ["score_attempt"]
