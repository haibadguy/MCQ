"""
ViT5 Distractor Generator for Vietnamese
=========================================
Kiến trúc giống Leaf/Tangsang English DistractorGenerator:
  Input:  "{answer} <sep> {question} <sep> {context}"
  Output: "{d1} <sep> {d2} <sep> {d3}"

Model priority:
  1. Local trained model (app/ml_models/vit5_vietnamese/distractor_generator/)
  2. Heuristic fallback (n-gram extraction) nếu chưa có model

Train local:
  python training/vn/prepare_dataset.py
  python training/vn/train_vit5_distractor.py
"""

import re
import string
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Constants (mirror Leaf/Tangsang English DistractorGenerator) ──────────────
SEP_TOKEN       = "<sep>"
LOCAL_MODEL_DIR = "app/ml_models/vit5_vietnamese/distractor_generator"
SOURCE_MAX_LEN  = 512
TARGET_MAX_LEN  = 64


class ViT5DistractorGenerator:
    """
    Vietnamese distractor generator using the same seq2seq two-stage architecture
    as the English T5 distractor pipeline in Leaf/Tangsang.

    Input:  "{answer} <sep> {question} <sep> {context}"
    Output: "{d1} <sep> {d2} <sep> {d3}"
    """

    def __init__(self, is_verbose: bool = False):
        self.is_verbose = is_verbose
        self.tokenizer = None
        self.model = None
        self._ready = False
        self._load()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self):
        local = Path(LOCAL_MODEL_DIR)
        if not (local.exists() and any(local.iterdir())):
            if self.is_verbose:
                print(f"[ViT5DG] Local model not found at {local}")
                print("[ViT5DG] Will use heuristic fallback until local model is trained.")
                print("[ViT5DG] Train: python training/vn/train_vit5_distractor.py")
            return  # heuristic fallback active

        if self.is_verbose:
            print(f"[ViT5DG] Loading local model: {local}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(local))
        if SEP_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([SEP_TOKEN])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(local))
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self._ready = True
        if self.is_verbose:
            print("[ViT5DG] ✓ Local distractor model loaded.")

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(self, answer: str, question: str, context: str, num_distractors: int = 3) -> List[str]:
        """
        Generate Vietnamese distractors.

        Args:
            answer:  Correct answer string
            question: Question string
            context:  Source paragraph (Vietnamese)
            num_distractors: How many distractors to return (max 3 per call)

        Returns:
            List of distractor strings
        """
        if self._ready:
            return self._model_generate(answer, question, context, num_distractors)
        else:
            return self._heuristic_fallback(answer, context, num_distractors)

    # ── Model-based generation (Leaf format) ─────────────────────────────────

    def _model_generate(self, answer: str, question: str, context: str, num_distractors: int) -> List[str]:
        """
        Use trained ViT5 model. Mirrors English DistractorGenerator exactly.
        """
        generate_count = int(num_distractors / 3) + 1  # same logic as English pipeline

        # Input format: "{answer} <sep> {question} <sep> {context}"
        source = f"{answer} {SEP_TOKEN} {question} {SEP_TOKEN} {context}"
        encoding = self.tokenizer(
            source,
            max_length=SOURCE_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        generated_ids = self.model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            num_beams=generate_count,
            num_return_sequences=generate_count,
            max_length=TARGET_MAX_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )

        # Decode and collect all outputs (same logic as English DistractorGenerator)
        all_preds = set()
        for gid in generated_ids:
            decoded = self.tokenizer.decode(gid, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            all_preds.add(decoded)

        # Parse "{d1} <sep> {d2} <sep> {d3}" format
        distractors = []
        for pred in all_preds:
            cleaned = pred.replace("<pad>", "").replace("</s>", f"{SEP_TOKEN}")
            cleaned = self._replace_extra_ids(cleaned)
            parts = [p.strip() for p in cleaned.split(SEP_TOKEN) if p.strip()]
            distractors.extend(parts)

        # Clean, deduplicate, remove answer
        distractors = self._clean(distractors, answer)
        return distractors[:num_distractors]

    # ── Heuristic fallback (no model needed) ──────────────────────────────────

    def _heuristic_fallback(self, answer: str, context: str, num_distractors: int) -> List[str]:
        """
        Simple n-gram extraction fallback when local model is not trained yet.
        Suitable for demo/testing; replaced by the trained model after training.
        """
        words = context.split()
        answer_lower = answer.lower().strip()
        candidates = set()

        # Slide window 1–5 words
        for window in range(1, 6):
            for i in range(len(words) - window + 1):
                phrase = " ".join(words[i:i + window])
                phrase_clean = phrase.lower().strip()
                if (phrase_clean != answer_lower
                        and len(phrase) >= 2
                        and answer_lower not in phrase_clean
                        and phrase_clean not in answer_lower
                        and not re.match(r'^[\d\s]+$', phrase)):
                    candidates.add(phrase)

        result = list(candidates)[:num_distractors * 4]
        result = self._clean(result, answer)
        return result[:num_distractors]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clean(self, distractors: List[str], answer: str) -> List[str]:
        """Remove punctuation-only, duplicates, and answer-overlapping items."""
        answer_lower = answer.lower().strip()
        seen = set()
        cleaned = []
        for d in distractors:
            d = d.strip()
            if not d or len(d) < 2:
                continue
            d_norm = d.lower()
            if d_norm == answer_lower:
                continue
            if d_norm in seen:
                continue
            seen.add(d_norm)
            cleaned.append(d)
        return cleaned

    def _replace_extra_ids(self, text: str) -> str:
        """Replace T5's <extra_id_N> tokens with <sep> (mirrors English pipeline)."""
        while "<extra_id_" in text:
            start = text.index("<extra_id_")
            end   = text.index(">", start)
            text  = text[:start] + SEP_TOKEN + text[end + 1:]
        return text
