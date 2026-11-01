from dataclasses import dataclass, field


@dataclass
class CoachingSession:
    session_id: str
    user_id: int
    target_sign: str = ""
    attempts: int = 0
    correct_count: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return round(self.correct_count / self.attempts, 4)

    def record_attempt(self, predicted: str, correct: bool) -> None:
        self.attempts += 1
        if correct:
            self.correct_count += 1
        self.history.append(
            {
                "attempt": self.attempts,
                "target": self.target_sign,
                "predicted": predicted,
                "correct": correct,
            }
        )
