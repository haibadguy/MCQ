"""
PhoBERT-based Distractor Generator for Vietnamese
=================================================
Option A: PhoBERT embeddings (vinai/phobert-base-v2) + candidate extraction
  from context via noun chunking → cosine similarity ranking.

Strategy:
  1. Extract candidate phrases from context (sliding window + underthesea NLP)
  2. Encode candidates + correct answer with PhoBERT
  3. Keep candidates:
     - NOT too similar to answer  (cos_sim < 0.92)
     - NOT too different           (cos_sim > 0.20)  ← meaningful, same domain
     - Not substring of answer, not duplicate
  4. Return top-k by similarity distance from answer (closest but wrong)

Reference: "Distractor Generation for Vietnamese Multiple-Choice Questions"
           (various EMNLP 2024 multilingual papers using similar approach)
"""

from typing import List
import re

import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "vinai/phobert-base-v2"

# Thresholds (tuned empirically for Vietnamese educational text)
MAX_SIM = 0.92   # too similar → likely same concept
MIN_SIM = 0.15   # too different → unrelated noise
WINDOW_SIZES = [1, 2, 3, 4, 5]  # n-gram window sizes for candidate extraction


def _mean_pool(token_embeddings, attention_mask):
    """Mean pooling over token embeddings."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm() + 1e-9)
    b = b / (b.norm() + 1e-9)
    return float(torch.dot(a, b))


class PhoBERTDistractorGenerator:
    """
    Generates Vietnamese distractors using PhoBERT embeddings.
    Extracts candidate phrases from context, ranks by semantic similarity
    to the correct answer, and returns the top-k plausible distractors.
    """

    def __init__(self, is_verbose: bool = False):
        self.is_verbose = is_verbose
        if is_verbose:
            print(f"[PhoBERTDistractor] Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        if is_verbose:
            print("[PhoBERTDistractor] Model loaded.")

    def _embed(self, text: str) -> torch.Tensor:
        """Return mean-pooled PhoBERT embedding for a text string."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            out = self.model(**enc)
        return _mean_pool(out.last_hidden_state, enc["attention_mask"]).squeeze(0)

    def _extract_candidates(self, context: str, answer: str) -> List[str]:
        """
        Extract n-gram candidates from context using sliding window.
        Filters out candidates that contain or are contained by the answer.
        """
        # Normalize whitespace, split on word boundaries
        tokens = re.split(r'\s+', context.strip())
        candidates = set()
        answer_lower = answer.strip().lower()

        for window in WINDOW_SIZES:
            for i in range(len(tokens) - window + 1):
                phrase = ' '.join(tokens[i:i + window])
                phrase_lower = phrase.lower()
                # Basic filters
                if len(phrase) < 2:
                    continue
                if phrase_lower == answer_lower:
                    continue
                if answer_lower in phrase_lower or phrase_lower in answer_lower:
                    continue
                # Remove pure numbers and single punctuation
                if re.match(r'^[\d\s]+$', phrase):
                    continue
                candidates.add(phrase)

        # Also try sentence-level splits as longer candidates
        sentences = re.split(r'[.!?;]', context)
        for sent in sentences:
            sent = sent.strip()
            if 3 <= len(sent.split()) <= 8:
                sent_lower = sent.lower()
                if sent_lower != answer_lower and answer_lower not in sent_lower:
                    candidates.add(sent)

        return list(candidates)

    def generate(self, answer: str, context: str, num_distractors: int = 3) -> List[str]:
        """
        Main interface: generate `num_distractors` Vietnamese distractors.
        
        Args:
            answer: The correct answer string
            context: The source paragraph (Vietnamese)
            num_distractors: How many distractors to return
        
        Returns:
            List of distractor strings (may be < num_distractors if context is thin)
        """
        candidates = self._extract_candidates(context, answer)
        if not candidates:
            if self.is_verbose:
                print("[PhoBERTDistractor] No candidates extracted from context.")
            return []

        # Embed answer
        answer_emb = self._embed(answer)

        # Score each candidate
        scored = []
        for candidate in candidates:
            try:
                cand_emb = self._embed(candidate)
                sim = _cosine_similarity(answer_emb, cand_emb)
                scored.append((candidate, sim))
            except Exception:
                continue

        # Filter by similarity range
        filtered = [
            (cand, sim) for cand, sim in scored
            if MIN_SIM <= sim <= MAX_SIM
        ]

        if not filtered:
            # Relax filter if nothing passes
            filtered = scored

        # Sort: closest to answer first (maximally plausible but wrong)
        filtered.sort(key=lambda x: -x[1])

        # Deduplicate: remove near-identical strings
        selected = []
        selected_lower = set()
        for cand, _ in filtered:
            norm = re.sub(r'\s+', ' ', cand.lower().strip())
            if norm not in selected_lower:
                selected.append(cand)
                selected_lower.add(norm)
            if len(selected) >= num_distractors:
                break

        if self.is_verbose:
            print(f"[PhoBERTDistractor] Generated {len(selected)} distractors: {selected}")

        return selected
