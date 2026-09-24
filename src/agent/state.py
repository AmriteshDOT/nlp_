from typing import TypedDict


class EssayState(TypedDict):
    essay: str
    score: float
    context: str
    feedback: str
    eval_comments: str
    revision_count: int
