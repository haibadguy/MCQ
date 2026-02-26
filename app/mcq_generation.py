from typing import List
from nltk.tokenize import sent_tokenize
import toolz

from app.modules.duplicate_removal import remove_distractors_duplicate_with_correct_answer, remove_duplicates
from app.modules.text_cleaning import clean_text
from app.modules.translator import detect_language, translate_to_english, translate_to_vietnamese
from app.modules.answer_type import get_answer_type
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.ml_models.sense2vec_distractor_generation.sense2vec_generation import Sense2VecDistractorGeneration
from app.models.question import Question

import time

REQUIRED_DISTRACTOR_COUNT = 3


class MCQGenerator():
    def __init__(self, is_verbose=False):
        start_time = time.perf_counter()
        print('Loading ML Models...')

        # Currently not used
        # self.answer_generator = AnswerGenerator()
        # print('Loaded AnswerGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.question_generator = QuestionGenerator()
        print('Loaded QuestionGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.distractor_generator = DistractorGenerator()
        print('Loaded DistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration()
        print('Loaded Sense2VecDistractorGenerator in', round(time.perf_counter() - start_time, 2), 'seconds.') if is_verbose else ''

    # Main function
    def generate_mcq_questions(self, context: str, desired_count: int) -> List[Question]:
        lang = detect_language(context)
        print(f'[Language detected: {lang}]')

        if lang == 'vi':
            return self._generate_for_vietnamese(context, desired_count)
        else:
            return self._generate_for_english(context, desired_count)

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

    def _generate_for_vietnamese(self, context_vi: str, desired_count: int) -> List[Question]:
        print('[Vietnamese mode] Translating context to English...')
        context_en = translate_to_english(context_vi)

        # Generate MCQs in English
        questions_en = self._generate_for_english(context_en, desired_count)

        # Translate results back to Vietnamese
        print('[Vietnamese mode] Translating results back to Vietnamese...')
        for question in questions_en:
            question.questionText = translate_to_vietnamese(question.questionText)
            question.answerText = translate_to_vietnamese(question.answerText)
            question.distractors = [translate_to_vietnamese(d) for d in question.distractors]

        return questions_en

    def _generate_answers(self, context: str, desired_count: int) -> List[Question]:
        # answers = self.answer_generator.generate(context, desired_count)
        answers = self._generate_multiple_answers_according_to_desired_count(context, desired_count)

        print(answers)
        unique_answers = remove_duplicates(answers)

        questions = []
        for answer in unique_answers:
            questions.append(Question(answer))

        return questions

    def _generate_questions(self, context: str, questions: List[Question]) -> List[Question]:        
        for question in questions:
            question.questionText = self.question_generator.generate(question.answerText, context)

        return questions

    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        context_splits = self._split_context_according_to_desired_count(context, desired_count)

        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate_qna(split)
            questions.append(Question(answer.capitalize(), question))

        questions = list(toolz.unique(questions, key=lambda x: x.answerText))

        return questions

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        # Pre-compute answer types from question text for cross-question fallback
        # Keyed by answerText for easy lookup; type derived from the question wording
        answer_types = {q.answerText: get_answer_type(q.questionText) for q in questions}
        all_answers = list(answer_types.keys())

        for question in questions:
            t5_distractors = self.distractor_generator.generate(5, question.answerText, question.questionText, context)

            distractors = t5_distractors
            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                # Supplement with sense2vec
                needed = REQUIRED_DISTRACTOR_COUNT - len(distractors)
                s2v_distractors = self.sense2vec_distractor_generator.generate(question.answerText, needed + 2)
                distractors = distractors + s2v_distractors

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            # Type-aware cross-question fallback
            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                current_type = get_answer_type(question.questionText)
                # First try: only same-type answers from other questions
                type_matched = [
                    a for a in all_answers
                    if a.lower() != question.answerText.lower()
                    and a not in distractors
                    and answer_types.get(a) == current_type
                ]
                distractors = distractors + type_matched
                distractors = remove_duplicates(distractors)
                distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            # Last resort: any cross-question answer (only if still short after type-matching)
            if len(distractors) < REQUIRED_DISTRACTOR_COUNT:
                any_cross = [
                    a for a in all_answers
                    if a.lower() != question.answerText.lower()
                    and a not in distractors
                ]
                distractors = distractors + any_cross
                distractors = remove_duplicates(distractors)
                distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            # Always cap to exactly REQUIRED_DISTRACTOR_COUNT
            question.distractors = distractors[:REQUIRED_DISTRACTOR_COUNT]

        return questions

    # Helper functions 
    def _generate_answer_for_each_sentence(self, context: str) -> List[str]:
        sents = sent_tokenize(context)

        answers = []
        for sent in sents:
            answers.append(self.answer_generator.generate(sent, 1)[0])

        return answers

    #TODO: refactor to create better splits closer to the desired amount
    def _split_context_according_to_desired_count(self, context: str, desired_count: int) -> List[str]:
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count

        context_splits = []

        if sent_ratio < 1:
            return sents
        else:
            take_sents_count = int(sent_ratio + 1)

            start_sent_index = 0

            while start_sent_index < len(sents):
                context_split = ' '.join(sents[start_sent_index: start_sent_index + take_sents_count])
                context_splits.append(context_split)
                start_sent_index += take_sents_count - 1

        return context_splits
    
