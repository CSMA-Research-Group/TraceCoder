"""Literal implementation of paper Algorithm 2, including its equal-score branch."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str
    best_code: str
    best_score: int
    next_code: str
    stagnation: int


def rollback(attempted_code, attempted_score, best_code, best_score, previous_score, stagnation):
    if attempted_score > best_score:
        return Decision("accept", attempted_code, attempted_score, attempted_code, 0)
    if attempted_score == best_score:
        return Decision("continue", best_code, best_score, attempted_code, stagnation + 1)
    if attempted_score < previous_score:
        return Decision("rollback", best_code, best_score, best_code, stagnation + 1)
    return Decision("continue", best_code, best_score, attempted_code, stagnation + 1)
