"""
training/vn/prepare_dataset.py
================================
Chuẩn bị dataset tiếng Việt cho fine-tune ViT5 (QA + Distractor generation).

Dataset nguồn:
  QA:
    - UIT-ViQuAD 2.0 (uitnlp/viquad2) — 23k+ QA pairs từ Wikipedia VN
  Distractor:
    - ViMMRC 2.0     — MCQ thi thật, có 4 options + correct label
    - VSMRC          — MCQ synthetic chất lượng cao từ Wikipedia VN

Format output (giống Leaf/Tangsang):
  QA:         input = "[MASK] <sep> {context}"
              target = "{answer} <sep> {question}"

  Distractor: input = "{answer} <sep> {question} <sep> {context}"
              target = "{d1} <sep> {d2} <sep> {d3}"

Output files:
  training/vn/data/qa_train.json
  training/vn/data/qa_val.json
  training/vn/data/qa_test.json
  training/vn/data/dg_train.json
  training/vn/data/dg_val.json
  training/vn/data/dg_test.json

Cách chạy:
  python training/vn/prepare_dataset.py
  python training/vn/prepare_dataset.py --vimmrc_path path/to/vimmrc.json --vsmrc_path path/to/vsmrc.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

SEP = "<sep>"

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("training/vn/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def split_dataset(data: list, train=0.8, val=0.1) -> Tuple[list, list, list]:
    """80/10/10 split."""
    random.shuffle(data)
    n = len(data)
    t_end = int(n * train)
    v_end = t_end + int(n * val)
    return data[:t_end], data[t_end:v_end], data[v_end:]


def save_split(data: list, prefix: str):
    train, val, test = split_dataset(data)
    splits = {"train": train, "val": val, "test": test}
    for split_name, records in splits.items():
        path = OUT_DIR / f"{prefix}_{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(records):,} records → {path}")
    return len(train), len(val), len(test)


# ── Dataset 1: ViQuAD 2.0 (QA generation) ────────────────────────────────────

def load_viquad(max_samples: int = None) -> list:
    """
    Load UIT-ViQuAD 2.0 từ HuggingFace Datasets.
    Mỗi row: {context, question, answer_text}
    Output format: {input, target}
    """
    print("[ViQuAD] Loading uitnlp/viquad2 via HuggingFace Datasets…")
    try:
        from datasets import load_dataset
        ds = load_dataset("uitnlp/viquad2", trust_remote_code=True)
        split = ds.get("train", ds[list(ds.keys())[0]])
    except Exception as e:
        print(f"  [ViQuAD] HF load failed: {e}")
        print("  [ViQuAD] Place viquad2 JSON file at training/vn/data/raw/viquad2.json")
        return _load_viquad_local()

    records = []
    count = 0
    for row in split:
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()

        # answers field structure varies by dataset version
        answers_field = row.get("answers") or {}
        if isinstance(answers_field, dict):
            answer_list = answers_field.get("text", [])
        elif isinstance(answers_field, list):
            answer_list = answers_field
        else:
            answer_list = []

        answer = answer_list[0].strip() if answer_list else ""

        if not (context and question and answer):
            continue

        records.append({
            "input": f"[MASK] {SEP} {context}",
            "target": f"{answer} {SEP} {question}"
        })
        count += 1
        if max_samples and count >= max_samples:
            break

    print(f"  [ViQuAD] Loaded {len(records):,} QA records.")
    return records


def _load_viquad_local() -> list:
    """Fallback: load from local JSON (SQuAD-style format)."""
    local_path = Path("training/vn/data/raw/viquad2.json")
    if not local_path.exists():
        print(f"  [ViQuAD] Local file not found: {local_path}")
        return []

    with open(local_path, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for article in raw.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "").strip()
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                answers = qa.get("answers", [])
                answer = answers[0]["text"].strip() if answers else ""
                if context and question and answer:
                    records.append({
                        "input": f"[MASK] {SEP} {context}",
                        "target": f"{answer} {SEP} {question}"
                    })

    print(f"  [ViQuAD] Local load: {len(records):,} QA records.")
    return records


# ── Dataset 2: ViMMRC 2.0 (Distractor generation) ─────────────────────────────

def load_vimmrc(vimmrc_path: str = None) -> list:
    """
    Load ViMMRC 2.0 MCQ dataset.
    Format: {question, correct_option, incorrect_options, context/passage}
    Output: Leaf distractor format
    """
    if vimmrc_path is None:
        vimmrc_path = "training/vn/data/raw/vimmrc.json"

    path = Path(vimmrc_path)
    if not path.exists():
        print(f"  [ViMMRC] File not found: {path}")
        print("  [ViMMRC] Download from https://nlp.uit.edu.vn/datasets")
        print("  [ViMMRC] or: pip install datasets; load_dataset('uitnlp/ViMMRC')")
        try:
            return _load_vimmrc_hf()
        except Exception as e:
            print(f"  [ViMMRC] HF fallback also failed: {e}")
            return []

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = _convert_mcq_to_dg(raw if isinstance(raw, list) else raw.get("data", []))
    print(f"  [ViMMRC] Loaded {len(records):,} DG records from local file.")
    return records


def _load_vimmrc_hf() -> list:
    """Try HuggingFace uitnlp/ViMMRC."""
    from datasets import load_dataset
    print("  [ViMMRC] Trying HuggingFace uitnlp/ViMMRC…")
    ds = load_dataset("uitnlp/ViMMRC", trust_remote_code=True)
    split = ds.get("train", ds[list(ds.keys())[0]])
    raw_list = []
    for row in split:
        raw_list.append(row)
    records = _convert_mcq_to_dg(raw_list)
    print(f"  [ViMMRC] HF load: {len(records):,} DG records.")
    return records


def _convert_mcq_to_dg(rows: list) -> list:
    """Convert MCQ rows → Leaf distractor format."""
    records = []
    for row in rows:
        # Support multiple key schemas
        question = (row.get("question") or row.get("Question") or "").strip()
        context   = (row.get("context")  or row.get("paragraph") or row.get("article") or "").strip()

        # Detect options
        options = []
        if "options" in row and isinstance(row["options"], list):
            options = [o.strip() for o in row["options"]]
        else:
            for key in ["A", "B", "C", "D", "option_a", "option_b", "option_c", "option_d"]:
                val = row.get(key, "")
                if val:
                    options.append(val.strip())

        correct_key = (row.get("answer") or row.get("correct") or row.get("label") or "A")
        # Map "A"/"B"/"C"/"D" → index
        idx_map = {"A": 0, "B": 1, "C": 2, "D": 3, "0": 0, "1": 1, "2": 2, "3": 3}
        if isinstance(correct_key, str) and correct_key.upper() in idx_map:
            correct_idx = idx_map[correct_key.upper()]
        elif isinstance(correct_key, int):
            correct_idx = correct_key
        else:
            correct_idx = 0

        if len(options) < 4:
            continue

        correct  = options[correct_idx]
        incorrect = [o for i, o in enumerate(options) if i != correct_idx][:3]

        if not (question and context and correct and len(incorrect) == 3):
            continue

        records.append({
            "input":  f"{correct} {SEP} {question} {SEP} {context}",
            "target": f"{incorrect[0]} {SEP} {incorrect[1]} {SEP} {incorrect[2]}"
        })
    return records


# ── Dataset 3: VSMRC (Distractor generation — additional) ─────────────────────

def load_vsmrc(vsmrc_path: str = None) -> list:
    """
    Load VSMRC dataset (arXiv:2506.15978).
    Similar MCQ format to ViMMRC.
    """
    if vsmrc_path is None:
        vsmrc_path = "training/vn/data/raw/vsmrc.json"

    path = Path(vsmrc_path)
    if not path.exists():
        print(f"  [VSMRC] File not found: {path}")
        print("  [VSMRC] Available at: https://github.com/vsmrc-paper/vsmrc")
        print("  [VSMRC] Skipping VSMRC — using ViMMRC only.")
        return []

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", [])
    records = _convert_mcq_to_dg(data)
    print(f"  [VSMRC] Loaded {len(records):,} DG records.")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    print("\n" + "=" * 60)
    print("  Leaf-ViT5 Dataset Preparation")
    print("=" * 60)

    # ── QA dataset (ViQuAD) ──
    print("\n[1/2] Preparing QA dataset (ViQuAD 2.0)…")
    qa_records = load_viquad(max_samples=args.max_qa)
    if qa_records:
        tr, va, te = save_split(qa_records, "qa")
        print(f"  QA total: {tr+va+te:,} (train={tr} val={va} test={te})")
    else:
        print("  ⚠ No QA records — training/vn/data/qa_*.json will be empty.")

    # ── Distractor dataset (ViMMRC + VSMRC) ──
    print("\n[2/2] Preparing Distractor dataset (ViMMRC 2.0 + VSMRC)…")
    dg_records = []
    dg_records += load_vimmrc(args.vimmrc_path)
    dg_records += load_vsmrc(args.vsmrc_path)

    if dg_records:
        random.shuffle(dg_records)
        tr, va, te = save_split(dg_records, "dg")
        print(f"  DG total: {tr+va+te:,} (train={tr} val={va} test={te})")
    else:
        print("  ⚠ No DG records — provide --vimmrc_path or place vimmrc.json/vsmrc.json in data/raw/.")

    print("\n✓ Dataset preparation complete.")
    print(f"  Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Vietnamese ViT5 training data")
    parser.add_argument("--vimmrc_path", default=None, help="Path to vimmrc.json")
    parser.add_argument("--vsmrc_path",  default=None, help="Path to vsmrc.json")
    parser.add_argument("--max_qa",      type=int, default=None, help="Max QA samples (default: all)")
    args = parser.parse_args()
    main(args)
