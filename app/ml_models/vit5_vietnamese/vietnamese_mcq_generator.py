"""
Vietnamese MCQ Generator – Orchestrator (Option A)
===================================================
Combines ViT5 QA generation with PhoBERT distractor generation
to produce full MCQs natively in Vietnamese.

Pipeline:
  context (VI) → ViT5 QA → (answer, question)
                 PhoBERT → distractors (cosine similarity from context)
  → Question object with 1 correct + 3 distractors
"""

from typing import List
from nltk.tokenize import sent_tokenize

from app.ml_models.vit5_vietnamese.vit5_qa_generator import ViT5QAGenerator
from app.ml_models.vit5_vietnamese.phobert_distractor_generator import PhoBERTDistractorGenerator
from app.modules.duplicate_removal import remove_duplicates, remove_distractors_duplicate_with_correct_answer
from app.models.question import Question

import toolz

REQUIRED_DISTRACTOR_COUNT = 3


class VietnameseMCQGenerator:
    """
    Native Vietnamese MCQ generator.
    Does NOT use machine translation – all processing in Vietnamese.
    """

    def __init__(self, is_verbose: bool = False):
        self.is_verbose = is_verbose
        self.qa_generator = ViT5QAGenerator(is_verbose)
        self.distractor_generator = PhoBERTDistractorGenerator(is_verbose)

    def generate_mcq_questions(self, context_vi: str, desired_count: int) -> List[Question]:
        """
        Generate MCQs natively from Vietnamese text.
        
        Args:
            context_vi: Input text in Vietnamese
            desired_count: Number of MCQs to generate
        
        Returns:
            List of Question objects
        """
        context_splits = self._split_context(context_vi, desired_count)
        questions = []

        for split in context_splits:
            try:
                answer, question_text = self.qa_generator.generate_qna(split)
                q = Question(answer.capitalize(), question_text)
                questions.append(q)
            except Exception as e:
                if self.is_verbose:
                    print(f"[ViMCQGen] QA generation failed for split: {e}")
                continue

        # Deduplicate by answer
        questions = list(toolz.unique(questions, key=lambda x: x.answerText.lower()))

        # Generate distractors for each question
        for question in questions:
            distractors = self.distractor_generator.generate(
                answer=question.answerText,
                context=context_vi,
                num_distractors=REQUIRED_DISTRACTOR_COUNT + 2  # ask for extra, filter later
            )

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(
                question.answerText, distractors
            )

            # Cross-question fallback if PhoBERT didn't produce enough
            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                all_answers = [q.answerText for q in questions if q.answerText != question.answerText]
                extra = [a for a in all_answers if a not in distractors]
                distractors = distractors + extra
                distractors = remove_duplicates(distractors)
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

    def _split_context(self, context: str, desired_count: int) -> List[str]:
        """
        Split Vietnamese context into chunks.
        Uses simple sentence splitting (underthesea optional, fallback to punctuation).
        """
        try:
            # Try underthesea Vietnamese sentence splitting
            from underthesea import sent_tokenize as vi_sent_tokenize
            sents = vi_sent_tokenize(context)
        except ImportError:
            # Fallback: split on Vietnamese sentence-ending punctuation
            import re
            sents = re.split(r'(?<=[.!?])\s+', context.strip())
            if not sents:
                sents = [context]

        if not sents:
            return [context]

        sent_ratio = len(sents) / max(desired_count, 1)

        if sent_ratio < 1:
            return sents

        take_count = max(int(sent_ratio + 1), 2)
        splits = []
        i = 0
        while i < len(sents):
            chunk = ' '.join(sents[i:i + take_count])
            splits.append(chunk)
            i += max(take_count - 1, 1)

        return splits
