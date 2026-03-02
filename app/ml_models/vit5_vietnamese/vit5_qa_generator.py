"""
ViT5 Question-Answer Generator for Vietnamese
=============================================
Model: namngo/pipeline-vit5-viquad-qg
  - Fine-tuned on ViQuAD + MLQA-vi
  - Paper: "Towards Vietnamese Question and Answer Generation: An Empirical Study"
    ACM TALLIP 2024 (Shaun-le/ViQAG)

Input format:  "generate question: <context>"
               or end-to-end: the model generates "question: ... answer: ..."

Output: List of (question, answer) tuples
"""

from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re


MODEL_NAME = "namngo/pipeline-vit5-viquad-qg"

# Parsing patterns for model output
QA_SPLIT_PATTERN = re.compile(r'question:\s*(.+?)\s*answer:\s*(.+)', re.IGNORECASE | re.DOTALL)
QUESTION_PATTERN = re.compile(r'question:\s*(.+)', re.IGNORECASE)


class ViT5QAGenerator:
    """
    Wraps the namngo/pipeline-vit5-viquad-qg model to generate
    Vietnamese question-answer pairs from a given context.
    """

    def __init__(self, is_verbose: bool = False):
        self.is_verbose = is_verbose
        if is_verbose:
            print(f"[ViT5QAGenerator] Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        if is_verbose:
            print(f"[ViT5QAGenerator] Model loaded successfully.")

    def generate_qna(self, context: str) -> Tuple[str, str]:
        """
        Generate a single (answer, question) pair from context.
        Returns (answer, question) – same order as English pipeline.
        """
        prompt = f"generate question: {context}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self.is_verbose:
            print(f"[ViT5QAGenerator] Raw output: {decoded}")

        # Try parsing "question: ... answer: ..."
        m = QA_SPLIT_PATTERN.search(decoded)
        if m:
            question = m.group(1).strip()
            answer = m.group(2).strip()
            return answer, question

        # Fallback: treat full output as question, extract noun phrase as answer
        question = decoded.strip()
        answer = self._extract_answer_from_context(context)
        return answer, question

    def _extract_answer_from_context(self, context: str) -> str:
        """Simple fallback: return first noun phrase (longest word sequence up to 5 tokens)."""
        words = context.split()
        # Return first 3–5 words as a rough answer proxy
        return ' '.join(words[:4]) if len(words) >= 4 else context[:50]

    def generate_multiple(self, context: str, num_return: int = 3) -> List[Tuple[str, str]]:
        """
        Generate multiple QA pairs using beam search diversity.
        Returns list of (answer, question) tuples.
        """
        prompt = f"generate question: {context}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=max(num_return * 2, 8),
            num_return_sequences=num_return,
            early_stopping=True
        )
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            m = QA_SPLIT_PATTERN.search(decoded)
            if m:
                question = m.group(1).strip()
                answer = m.group(2).strip()
                results.append((answer, question))
            else:
                answer = self._extract_answer_from_context(context)
                results.append((answer, decoded.strip()))
        return results
