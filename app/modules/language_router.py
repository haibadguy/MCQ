"""
Language Router – Dual Pipeline Dispatcher
==========================================
Detects input language and routes to:
  - 'en'  → English T5 pipeline (SQuAD QG + RACE distractor + Sense2Vec)
  - 'vi'  → Vietnamese ViT5 pipeline (ViT5 QA + PhoBERT distractors)

Uses langdetect (same library already in use by translator.py).
"""

from langdetect import detect, LangDetectException


SUPPORTED_VI_PIPELINE = True  # Set False to disable ViT5 and fall back to translation


def detect_pipeline(text: str) -> str:
    """
    Detect language of text and return pipeline identifier.

    Returns:
        'vi'  → use Vietnamese ViT5 native pipeline
        'en'  → use English T5 pipeline (default/fallback)
    """
    if not SUPPORTED_VI_PIPELINE:
        return 'en'
    try:
        lang = detect(text[:500])
        return lang if lang == 'vi' else 'en'
    except LangDetectException:
        return 'en'
