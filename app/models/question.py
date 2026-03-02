from typing import List


class Question:
    """
    Represents a generated Multiple-Choice Question.

    Attributes:
        answerText   – The correct answer string.
        questionText – The question string.
        distractors  – List of wrong answer options (up to 3).
        lang         – ISO-639-1 language code ('en' | 'vi').
                       Set automatically by MCQGenerator based on detected pipeline.
    """

    def __init__(
        self,
        answerText: str,
        questionText: str = '',
        distractors: List[str] = None,
        lang: str = 'en',
    ):
        self.answerText = answerText
        self.questionText = questionText
        self.distractors = distractors if distractors is not None else []
        self.lang = lang

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Question(lang={self.lang!r}, "
            f"answer={self.answerText!r}, "
            f"question={self.questionText[:60]!r})"
        )
