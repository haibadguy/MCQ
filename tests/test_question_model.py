"""
tests/test_question_model.py
=============================
Unit tests for app.models.question.Question.
Verifies the new `lang` field, default values, and __dict__ serialisation
(critical for the Flask api_gateway json response).

Run:
    cd d:\\ChuyenDeHTTT\\Leaf-Question-Generation
    python -m pytest tests/test_question_model.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.models.question import Question


class TestQuestionModel(unittest.TestCase):
    """Tests for the Question data model."""

    # ── Construction ─────────────────────────────────────────────────────────

    def test_basic_construction(self):
        """Minimum required field (answerText) should suffice."""
        q = Question("Paris")
        self.assertEqual(q.answerText, "Paris")

    def test_default_question_text_is_empty(self):
        q = Question("Paris")
        self.assertEqual(q.questionText, '')

    def test_default_distractors_is_empty_list(self):
        """Distractors default to [] (and must NOT share the same list object)."""
        q1 = Question("A")
        q2 = Question("B")
        q1.distractors.append("X")
        # q2.distractors should be independent (mutable default bug fixed)
        self.assertEqual(q2.distractors, [])

    def test_default_lang_is_en(self):
        """lang should default to 'en' for backward compatibility."""
        q = Question("test")
        self.assertEqual(q.lang, 'en')

    # ── Setting lang ─────────────────────────────────────────────────────────

    def test_lang_set_to_vi(self):
        q = Question("Hà Nội", lang='vi')
        self.assertEqual(q.lang, 'vi')

    def test_lang_can_be_set_after_construction(self):
        q = Question("test")
        q.lang = 'vi'
        self.assertEqual(q.lang, 'vi')

    # ── __dict__ serialisation (api_gateway compatibility) ───────────────────

    def test_dict_contains_lang(self):
        """__dict__ must include 'lang' – used by api_gateway for JSON response."""
        q = Question("Tokyo", questionText="What is the capital of Japan?", lang='en')
        d = q.__dict__
        self.assertIn('lang', d)
        self.assertEqual(d['lang'], 'en')

    def test_dict_contains_all_expected_keys(self):
        expected_keys = {'answerText', 'questionText', 'distractors', 'lang'}
        q = Question("Tokyo", questionText="Capital of Japan?", lang='en')
        self.assertTrue(expected_keys.issubset(set(q.__dict__.keys())))

    def test_dict_serialisation_with_vi_lang(self):
        q = Question("Hà Nội", questionText="Thủ đô Việt Nam là gì?", lang='vi')
        q.distractors = ["TP. HCM", "Đà Nẵng", "Huế"]
        d = q.__dict__
        self.assertEqual(d['lang'], 'vi')
        self.assertEqual(d['distractors'], ["TP. HCM", "Đà Nẵng", "Huế"])

    # ── __repr__ ─────────────────────────────────────────────────────────────

    def test_repr_contains_lang(self):
        q = Question("Paris", lang='en')
        r = repr(q)
        self.assertIn("lang='en'", r)

    def test_repr_contains_answer(self):
        q = Question("Paris", lang='vi')
        r = repr(q)
        self.assertIn("Paris", r)
        self.assertIn("lang='vi'", r)

    # ── Full MCQ ──────────────────────────────────────────────────────────────

    def test_full_mcq_en(self):
        q = Question(
            answerText="Australia",
            questionText="Where do koalas live?",
            distractors=["Africa", "Europe", "Asia"],
            lang='en'
        )
        self.assertEqual(q.answerText, "Australia")
        self.assertEqual(len(q.distractors), 3)
        self.assertEqual(q.lang, 'en')

    def test_full_mcq_vi(self):
        q = Question(
            answerText="Hà Nội",
            questionText="Thủ đô của Việt Nam là gì?",
            distractors=["TP. HCM", "Đà Nẵng", "Huế"],
            lang='vi'
        )
        self.assertEqual(q.lang, 'vi')
        self.assertEqual(len(q.distractors), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
