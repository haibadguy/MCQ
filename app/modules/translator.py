from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Returns ISO 639-1 language code (e.g. 'vi', 'en').
    Returns 'en' as fallback if detection fails.
    """
    try:
        # Use first 500 chars for speed
        return detect(text[:500])
    except LangDetectException:
        return 'en'


def translate_to_english(text: str) -> str:
    """Translate text to English using Google Translate (free tier)."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # fallback: return original if translation fails


def translate_to_vietnamese(text: str) -> str:
    """Translate text to Vietnamese using Google Translate (free tier)."""
    try:
        return GoogleTranslator(source='auto', target='vi').translate(text)
    except Exception:
        return text  # fallback: return original if translation fails
