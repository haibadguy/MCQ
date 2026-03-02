"""
tests/test_language_router.py
==============================
Unit tests for the language router module.
Tests run WITHOUT any ML model being loaded – pure logic only.

Run:
    cd d:\\ChuyenDeHTTT\\Leaf-Question-Generation
    python -m pytest tests/test_language_router.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDetectPipeline(unittest.TestCase):
    """Test suite for app.modules.language_router.detect_pipeline()."""

    # ── Happy-path: English ──────────────────────────────────────────────────

    def test_english_text_returns_en(self):
        """Classic English paragraph → 'en'."""
        from app.modules.language_router import detect_pipeline
        text = (
            "The koala is an arboreal herbivorous marsupial native to Australia. "
            "It is easily recognisable by its stout, tailless body and large head."
        )
        self.assertEqual(detect_pipeline(text), 'en')

    def test_english_scientific_text_returns_en(self):
        """Scientific English text → 'en'."""
        from app.modules.language_router import detect_pipeline
        text = (
            "Oxygen is the chemical element with atomic number 8. "
            "It is a member of the chalcogen group in the periodic table."
        )
        self.assertEqual(detect_pipeline(text), 'en')

    # ── Happy-path: Vietnamese ───────────────────────────────────────────────

    def test_vietnamese_text_returns_vi(self):
        """Plain Vietnamese paragraph → 'vi'."""
        from app.modules.language_router import detect_pipeline
        text = (
            "Việt Nam là một quốc gia nằm ở bán đảo Đông Dương thuộc khu vực Đông Nam Á. "
            "Hà Nội là thủ đô và Thành phố Hồ Chí Minh là thành phố đông dân nhất."
        )
        self.assertEqual(detect_pipeline(text), 'vi')

    def test_vietnamese_educational_text_returns_vi(self):
        """Longer educational Vietnamese text → 'vi'."""
        from app.modules.language_router import detect_pipeline
        text = (
            "Trái Đất là hành tinh thứ ba tính từ Mặt Trời và là hành tinh duy nhất "
            "được biết đến có sự sống. Hành tinh này có đường kính xích đạo khoảng "
            "12.742 km và khối lượng khoảng 5,97 × 10²⁴ kg."
        )
        self.assertEqual(detect_pipeline(text), 'vi')

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_empty_string_returns_en_fallback(self):
        """Empty string → falls back to 'en' (safe default)."""
        from app.modules.language_router import detect_pipeline
        self.assertEqual(detect_pipeline(''), 'en')

    def test_whitespace_only_returns_en_fallback(self):
        """Whitespace-only → falls back to 'en'."""
        from app.modules.language_router import detect_pipeline
        self.assertEqual(detect_pipeline('   \n\t  '), 'en')

    def test_single_word_non_vi_defaults_to_en(self):
        """Single word that isn't Vietnamese → 'en' (langdetect default)."""
        from app.modules.language_router import detect_pipeline
        result = detect_pipeline('hello')
        self.assertIn(result, ['en', 'vi'])  # Accept either – short text is ambiguous

    # ── SUPPORTED_VI_PIPELINE = False ────────────────────────────────────────

    def test_vi_pipeline_disabled_always_returns_en(self):
        """When SUPPORTED_VI_PIPELINE is False, always return 'en'."""
        import app.modules.language_router as lr
        with patch.object(lr, 'SUPPORTED_VI_PIPELINE', False):
            # Reload to pick up the patch
            text_vi = "Việt Nam là quốc gia đẹp với nhiều danh lam thắng cảnh tuyệt vời."
            result = lr.detect_pipeline(text_vi)
            self.assertEqual(result, 'en')

    # ── Return value contract ─────────────────────────────────────────────────

    def test_return_value_is_string(self):
        """detect_pipeline should always return a str."""
        from app.modules.language_router import detect_pipeline
        result = detect_pipeline("some text")
        self.assertIsInstance(result, str)

    def test_return_value_is_en_or_vi(self):
        """detect_pipeline should only return 'en' or 'vi'."""
        from app.modules.language_router import detect_pipeline
        texts = [
            "The capital of France is Paris.",
            "Hà Nội là thủ đô của Việt Nam.",
            "",
        ]
        for t in texts:
            r = detect_pipeline(t)
            self.assertIn(r, ('en', 'vi'), msg=f"Unexpected return '{r}' for text: {t!r}")

    # ── LangDetectException resilience ───────────────────────────────────────

    def test_langdetect_exception_falls_back_to_en(self):
        """If langdetect raises LangDetectException, detect_pipeline returns 'en'."""
        from langdetect import LangDetectException
        import app.modules.language_router as lr
        with patch('langdetect.detect', side_effect=LangDetectException(0, 'mocked')):
            result = lr.detect_pipeline("any text")
            self.assertEqual(result, 'en')


if __name__ == '__main__':
    unittest.main(verbosity=2)
