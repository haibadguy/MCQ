"""
ViT5 Question-Answer Generator for Vietnamese
=============================================
Kiến trúc giống Leaf/Tangsang English QA pipeline:
  Input:  "[MASK] <sep> {context}"
  Output: "{answer} <sep> {question}"

Model priority:
  1. Local trained model (app/ml_models/vit5_vietnamese/qa_generator/)
  2. HuggingFace public model (namngo/pipeline-vit5-viquad-qg)

Train local:
  python training/vn/prepare_dataset.py
  python training/vn/train_vit5_qa.py
"""

import re
from pathlib import Path
from typing import Tuple, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Constants (mirror Leaf/Tangsang English) ──────────────────────────────────
SEP_TOKEN       = "<sep>"
LOCAL_MODEL_DIR = "app/ml_models/vit5_vietnamese/qa_generator"
HF_FALLBACK     = "namngo/pipeline-vit5-viquad-qg"
SOURCE_MAX_LEN  = 512
TARGET_MAX_LEN  = 80


class ViT5QAGenerator:
    """
    Vietnamese QA generator using the same two-stage format as Leaf English:
      input  = "[MASK] <sep> {context}"
      output = "{answer} <sep> {question}"

    Falls back to HuggingFace model if local model not found.
    """

    def __init__(self, is_verbose: bool = False):
        self.is_verbose = is_verbose
        self.tokenizer = None
        self.model = None
        self._model_name = None
        self._load()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self):
        local = Path(LOCAL_MODEL_DIR)
        if local.exists() and any(local.iterdir()):
            self._load_from(str(local), label="local trained")
        else:
            if self.is_verbose:
                print(f"[ViT5QA] Local model not found at {local}")
                print(f"[ViT5QA] Falling back to HuggingFace: {HF_FALLBACK}")
                print("[ViT5QA] To train locally: python training/vn/train_vit5_qa.py")
            self._load_from(HF_FALLBACK, label="HuggingFace")

    def _load_from(self, model_path: str, label: str):
        if self.is_verbose:
            print(f"[ViT5QA] Loading {label} model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure <sep> token is registered (needed if loading local model)
        if SEP_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([SEP_TOKEN])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self._model_name = model_path
        if self.is_verbose:
            print(f"[ViT5QA] ✓ Loaded from: {model_path}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate_qna(self, context: str) -> Tuple[str, str]:
        """
        Generate (answer, question) from context.
        Returns (answer, question) — same order as English QuestionGenerator.
        """
        # Leaf/Tangsang format: "[MASK] <sep> {context}"
        prompt = f"[MASK] {SEP_TOKEN} {context}"
        inputs = self.tokenizer(
            prompt,
            max_length=SOURCE_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=TARGET_MAX_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if self.is_verbose:
            print(f"[ViT5QA] Raw output: {decoded}")

        # Parse "{answer} <sep> {question}" (Leaf format)
        if SEP_TOKEN in decoded:
            parts = decoded.split(SEP_TOKEN, 1)
            answer   = parts[0].strip()
            question = parts[1].strip() if len(parts) > 1 else ""
            return answer, question

        # HuggingFace fallback model may produce different format
        # Try "question: ... answer: ..."
        m = re.search(r'question:\s*(.+?)\s*answer:\s*(.+)', decoded, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # Last resort: full output as question, first phrase as answer
        return self._extract_answer(context), decoded.strip()

    def _extract_answer(self, context: str) -> str:
        """Fallback: return first 4 words as rough answer proxy."""
        words = context.split()
        return ' '.join(words[:4]) if len(words) >= 4 else context[:50]
