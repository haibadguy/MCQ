"""
Vietnamese MCQ Generator – Orchestrator (Leaf Two-Stage Architecture)
=====================================================================
Combines ViT5 QA generator + ViT5 Distractor generator — same two-stage
design as Leaf/Tangsang English pipeline, but fully in Vietnamese.

QA  format:  "[MASK] <sep> {context}"   → "{answer} <sep> {question}"
DG  format:  "{answer} <sep> {question} <sep> {context}" → "{d1} <sep> {d2} <sep> {d3}"

Model loading is LAZY — server starts even if models not yet trained.
After training (train_vit5_qa.py + train_vit5_distractor.py), models are
auto-detected from local paths on first call.
"""

import re
from typing import List, Optional
import toolz

from app.modules.duplicate_removal import remove_duplicates, remove_distractors_duplicate_with_correct_answer
from app.models.question import Question

REQUIRED_DISTRACTOR_COUNT = 3


class VietnameseMCQGenerator:
    """
    Native Vietnamese MCQ generator — two-stage ViT5 pipeline (Leaf/Tangsang compatible).
    """

    def __init__(self, is_verbose: bool = False):
        self.is_verbose = is_verbose
        self._qa_generator   = None
        self._dg_generator   = None
        self._load_attempted = False

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load both models on first call. Returns True if ready."""
        if self._load_attempted:
            return self._qa_generator is not None

        self._load_attempted = True
        try:
            from app.ml_models.vit5_vietnamese.vit5_qa_generator import ViT5QAGenerator
            self._qa_generator = ViT5QAGenerator(self.is_verbose)
        except Exception as e:
            print(f"[ViMCQGen] ⚠ QA model load failed: {e}")
            return False

        try:
            from app.ml_models.vit5_vietnamese.vit5_distractor_generator import ViT5DistractorGenerator
            self._dg_generator = ViT5DistractorGenerator(self.is_verbose)
        except Exception as e:
            print(f"[ViMCQGen] ⚠ Distractor model load failed: {e}")
            # Non-fatal: heuristic fallback will be used via the generator itself
            from app.ml_models.vit5_vietnamese.vit5_distractor_generator import ViT5DistractorGenerator
            self._dg_generator = ViT5DistractorGenerator(self.is_verbose)

        return True

    # ── Main entry ────────────────────────────────────────────────────────────

    def generate_mcq_questions(self, context_vi: str, desired_count: int) -> List[Question]:
        """
        Generate full MCQs natively in Vietnamese using two-stage ViT5 pipeline.

        Stage 1 (QA):  context → (answer, question)
        Stage 2 (DG):  (answer, question, context) → [d1, d2, d3]

        Returns list of Question objects (may be empty if models not available).
        """
        if not self._ensure_loaded():
            print("[ViMCQGen] Pipeline not available — returning empty.")
            return []

        context_splits = self._split_context(context_vi, desired_count)
        questions = []

        # ── Stage 1: QA generation ──
        for split in context_splits:
            try:
                answer, question_text = self._qa_generator.generate_qna(split)
                if answer and question_text:
                    q = Question(answer.capitalize(), question_text, lang='vi')
                    questions.append(q)
            except Exception as exc:
                if self.is_verbose:
                    print(f"[ViMCQGen] QA failed: {exc}")

        # Deduplicate by answer
        questions = list(toolz.unique(questions, key=lambda x: x.answerText.lower()))

        # ── Stage 2: Distractor generation ──
        all_answers = [q.answerText for q in questions]

        for question in questions:
            try:
                distractors = self._dg_generator.generate(
                    answer=question.answerText,
                    question=question.questionText,
                    context=context_vi,
                    num_distractors=REQUIRED_DISTRACTOR_COUNT + 2
                )
            except Exception as exc:
                if self.is_verbose:
                    print(f"[ViMCQGen] Distractor failed: {exc}")
                distractors = []

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(
                question.answerText, distractors
            )

            # Cross-question fallback (same as English pipeline)
            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                extra = [a for a in all_answers
                         if a.lower() != question.answerText.lower()
                         and a not in distractors]
                distractors = remove_duplicates(distractors + extra)
                distractors = remove_distractors_duplicate_with_correct_answer(
                    question.answerText, distractors
                )

            question.distractors = distractors[:REQUIRED_DISTRACTOR_COUNT]

            if self.is_verbose:
                print('-------------------')
                print(f'[ViMCQGen] Q: {question.questionText}')
                print(f'[ViMCQGen] A: {question.answerText}')
                print(f'[ViMCQGen] D: {question.distractors}')

        return questions

    # ── Context splitter ──────────────────────────────────────────────────────

    def _split_context(self, context: str, desired_count: int) -> List[str]:
        """Split context into sentence chunks using underthesea if available."""
        try:
            from underthesea import sent_tokenize as vi_sent
            sents = vi_sent(context)
        except ImportError:
            sents = re.split(r'(?<=[.!?])\s+', context.strip())
            sents = [s for s in sents if s.strip()]

        if not sents:
            return [context]

        sent_ratio = len(sents) / max(desired_count, 1)
        if sent_ratio < 1:
            return sents

        take_count = max(int(sent_ratio + 1), 2)
        splits, i = [], 0
        while i < len(sents):
            splits.append(' '.join(sents[i:i + take_count]))
            i += max(take_count - 1, 1)
        return splits
