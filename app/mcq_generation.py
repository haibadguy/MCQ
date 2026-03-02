from typing import List
from nltk.tokenize import sent_tokenize
import toolz
import time

from app.modules.duplicate_removal import remove_distractors_duplicate_with_correct_answer, remove_duplicates
from app.modules.text_cleaning import clean_text
from app.modules.answer_type import get_answer_type
from app.modules.language_router import detect_pipeline

# English T5 pipeline
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.ml_models.sense2vec_distractor_generation.sense2vec_generation import Sense2VecDistractorGeneration

# Vietnamese ViT5 + PhoBERT pipeline (Option A)
from app.ml_models.vit5_vietnamese.vietnamese_mcq_generator import VietnameseMCQGenerator

from app.models.question import Question

REQUIRED_DISTRACTOR_COUNT = 3


class MCQGenerator():
    def __init__(self, is_verbose=False):
        start_time = time.perf_counter()
        print('Loading ML Models...')

        # ── English T5 pipeline ──────────────────────────────────────────────
        self.question_generator = QuestionGenerator()
        print('Loaded QuestionGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.distractor_generator = DistractorGenerator()
        print('Loaded DistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration()
        print('Loaded Sense2VecDistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        # ── Vietnamese ViT5 + PhoBERT pipeline ──────────────────────────────
        self.vietnamese_generator = VietnameseMCQGenerator(is_verbose)
        print('Loaded VietnameseMCQGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        print(f'All models loaded in {round(time.perf_counter() - start_time, 2)}s')

    # ── Main entry point ────────────────────────────────────────────────────

    def generate_mcq_questions(self, context: str, desired_count: int) -> List[Question]:
        """
        Route to the appropriate pipeline based on detected language.
          - Vietnamese → ViT5 native pipeline (Option A)
          - English    → T5 SQuAD/RACE + Sense2Vec pipeline
        """
        lang = detect_pipeline(context)
        print(f'[MCQGenerator] Detected pipeline: {lang}')

        if lang == 'vi':
            questions = self.vietnamese_generator.generate_mcq_questions(context, desired_count)
        else:
            questions = self._generate_for_english(context, desired_count)

        # Stamp language on every question so downstream (API / frontend) can use it
        for q in questions:
            q.lang = lang

        return questions

    # ── English T5 pipeline ─────────────────────────────────────────────────

    def _generate_for_english(self, context: str, desired_count: int) -> List[Question]:
        cleaned_text = clean_text(context)
        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)

        for question in questions:
            print('-------------------')
            print(question.answerText)
            print(question.questionText)
            print(question.distractors)

        return questions

    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        context_splits = self._split_context_according_to_desired_count(context, desired_count)
        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate_qna(split)
            # lang is stamped in generate_mcq_questions(); default 'en' is correct here
            questions.append(Question(answer.capitalize(), question))

        questions = list(toolz.unique(questions, key=lambda x: x.answerText))
        return questions

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        answer_types = {q.answerText: get_answer_type(q.questionText) for q in questions}
        all_answers = list(answer_types.keys())

        for question in questions:
            t5_distractors = self.distractor_generator.generate(5, question.answerText, question.questionText, context)
            distractors = t5_distractors

            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                needed = REQUIRED_DISTRACTOR_COUNT - len(distractors)
                s2v_distractors = self.sense2vec_distractor_generator.generate(question.answerText, needed + 2)
                distractors = distractors + s2v_distractors

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            # Type-aware cross-question fallback
            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                current_type = get_answer_type(question.questionText)
                type_matched = [
                    a for a in all_answers
                    if a.lower() != question.answerText.lower()
                    and a not in distractors
                    and answer_types.get(a) == current_type
                ]
                distractors = distractors + type_matched
                distractors = remove_duplicates(distractors)
                distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                any_cross = [
                    a for a in all_answers
                    if a.lower() != question.answerText.lower()
                    and a not in distractors
                ]
                distractors = distractors + any_cross
                distractors = remove_duplicates(distractors)
                distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            question.distractors = distractors[:REQUIRED_DISTRACTOR_COUNT]

        return questions

    def _split_context_according_to_desired_count(self, context: str, desired_count: int) -> List[str]:
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count

        if sent_ratio < 1:
            return sents

        take_sents_count = int(sent_ratio + 1)
        context_splits = []
        start_sent_index = 0

        while start_sent_index < len(sents):
            context_split = ' '.join(sents[start_sent_index: start_sent_index + take_sents_count])
            context_splits.append(context_split)
            start_sent_index += take_sents_count - 1

        return context_splits
