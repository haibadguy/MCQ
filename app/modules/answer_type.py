import re


def get_answer_type(question_text: str) -> str:
    """
    Infer the expected answer type from the *question text*.

    This is more general and reliable than classifying the answer itself,
    because the question always signals what kind of answer is expected.

    Returns one of: 'PERSON_ORG', 'YEAR', 'LOCATION', 'NUMBER', 'OTHER'

    Examples:
        "Who discovered penicillin?"      → PERSON_ORG
        "When did Fleming publish?"        → YEAR
        "Where was Fleming born?"          → LOCATION
        "How many troops were mobilized?"  → NUMBER
        "What is the chemical symbol ...?" → OTHER
    """
    q = question_text.strip().lower()

    # --- WHO → person or organization ---
    if re.match(r'^who\b', q):
        return 'PERSON_ORG'

    # --- WHEN / year-related → date/year ---
    if (re.match(r'^when\b', q)
            or re.search(r'\bwhat year\b', q)
            or re.search(r'\bin what year\b', q)
            or re.search(r'\bby what year\b', q)
            or re.search(r'\bwhat date\b', q)):
        return 'YEAR'

    # --- WHERE / place-related → location ---
    if (re.match(r'^where\b', q)
            or re.search(r'\bwhat (city|country|state|place|location|region|town|nation|continent)\b', q)
            or re.search(r'\bin what (city|country|state|place|region|town|university|hospital)\b', q)):
        return 'LOCATION'

    # --- HOW MANY / HOW MUCH → number ---
    if re.match(r'^how (many|much|long|far|often|frequently|old)\b', q):
        return 'NUMBER'

    return 'OTHER'

