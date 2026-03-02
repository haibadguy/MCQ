"""
Leaf – MCQ Generation API Gateway
==================================
Endpoints:
  GET  /           → health check
  GET  /health      → JSON health status (lang pipelines enabled)
  POST /generate    → generate MCQs from plain text
  POST /generate/pdf → generate MCQs from uploaded PDF (binary multipart)

Response schema (per question):
  {
    "answerText":   str,
    "questionText": str,
    "distractors":  [str, str, str],
    "lang":         "en" | "vi"     ← NEW: language of detected input
  }
"""

import json
import contextlib

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from app.models.question import Question
from app.mcq_generation import MCQGenerator
from app.modules.language_router import detect_pipeline, SUPPORTED_VI_PIPELINE

# ── App setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Shared generator (heavy ML models loaded once at startup)
MCQ_Generator = MCQGenerator()


# ── Helper ──────────────────────────────────────────────────────────────────

def _questions_to_json(questions: list[Question]) -> list[dict]:
    """Serialize Question objects to plain dicts (json-safe)."""
    return [q.__dict__ for q in questions]


def _parse_count(raw) -> int:
    """Parse count field; default 5 if empty/invalid."""
    try:
        n = int(raw)
        return max(1, min(n, 20))   # clamp to [1, 20]
    except (TypeError, ValueError):
        return 5


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
@cross_origin()
def hello():
    """Basic health-check – used by Angular to detect if backend is running."""
    return json.dumps("Leaf MCQ Generation API – operational ✓")


@app.route("/health")
@cross_origin()
def health():
    """Structured health endpoint – returns pipeline capabilities."""
    return jsonify({
        "status": "ok",
        "pipelines": {
            "english_t5": True,
            "vietnamese_vit5": SUPPORTED_VI_PIPELINE,
        }
    })


@app.route("/generate", methods=["POST"])
@cross_origin()
def generate():
    """
    Generate MCQs from plain text.

    Request body (JSON):
        { "text": "<input text>", "count": <int 1-20> }

    Response (JSON array):
        [{ "answerText": ..., "questionText": ..., "distractors": [...], "lang": "en"|"vi" }, ...]
    """
    try:
        body = json.loads(request.data)
        text = body.get("text", "").strip()
        count = _parse_count(body.get("count", 5))
    except (json.JSONDecodeError, AttributeError):
        return jsonify({"error": "Invalid JSON body"}), 400

    if not text:
        return jsonify({"error": "Field 'text' is required and must not be empty."}), 400

    try:
        questions = MCQ_Generator.generate_mcq_questions(text, count)
        result = _questions_to_json(questions)
        return jsonify(result)
    except Exception as exc:
        app.logger.error("MCQ generation failed: %s", exc, exc_info=True)
        return jsonify({"error": "Internal generation error. Check backend logs."}), 500


@app.route("/generate/pdf", methods=["POST"])
@cross_origin()
def generate_pdf():
    """
    Generate MCQs from an uploaded PDF file.

    Request (multipart/form-data):
        file  – PDF binary
        count – number of MCQs (default 5)

    Delegates text extraction to the existing PDF module already used by the Flask app.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    pdf_file = request.files["file"]
    count = _parse_count(request.form.get("count", 5))

    try:
        # Re-use existing PDF extraction utility
        from app.modules.text_cleaning import clean_text
        import pdfplumber

        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()

        if not text:
            return jsonify({"error": "Could not extract text from PDF."}), 422

        text = clean_text(text)
        questions = MCQ_Generator.generate_mcq_questions(text, count)
        result = _questions_to_json(questions)
        return jsonify(result)

    except ImportError:
        return jsonify({"error": "pdfplumber not installed. Run: pip install pdfplumber"}), 501
    except Exception as exc:
        app.logger.error("PDF MCQ generation failed: %s", exc, exc_info=True)
        return jsonify({"error": "Internal generation error. Check backend logs."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("localhost", 9002, app, use_reloader=False, use_debugger=True)