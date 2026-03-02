"""
training/vn/train_vit5_qa.py
==============================
Fine-tune VietAI/vit5-base cho QA generation (ViQuAD 2.0).

Kiến trúc giống Leaf/Tangsang English QA:
  Input:  "[MASK] <sep> {context}"
  Target: "{answer} <sep> {question}"

Sử dụng LoRA (rank=16) để train chỉ ~8GB VRAM.
Sau train: merge LoRA → save app/ml_models/vit5_vietnamese/qa_generator/

Cách chạy:
  python training/vn/train_vit5_qa.py
  python training/vn/train_vit5_qa.py --epochs 5 --batch_size 8 --output_dir app/ml_models/vit5_vietnamese/qa_generator
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "VietAI/vit5-base"
SEP_TOKEN       = "<sep>"
SOURCE_MAX_LEN  = 512
TARGET_MAX_LEN  = 80
DEFAULT_OUTPUT  = "app/ml_models/vit5_vietnamese/qa_generator"
DATA_DIR        = Path("training/vn/data")


# ── Dataset ───────────────────────────────────────────────────────────────────

class QADataset(Dataset):
    def __init__(self, records: list, tokenizer, source_max_len=512, target_max_len=80):
        self.records = records
        self.tokenizer = tokenizer
        self.src_len = source_max_len
        self.tgt_len = target_max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        src = self.tokenizer(
            rec["input"],
            max_length=self.src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt = self.tokenizer(
            rec["target"],
            max_length=self.tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tgt["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ── Load data ─────────────────────────────────────────────────────────────────

def load_split(name: str) -> list:
    path = DATA_DIR / f"qa_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python training/vn/prepare_dataset.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── LoRA setup ────────────────────────────────────────────────────────────────

def apply_lora(model, rank: int = 16, alpha: int = 32):
    """Apply LoRA via PEFT."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"],         # T5 attention projections
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model, True
    except ImportError:
        print("⚠ peft not installed — training full model (needs more VRAM).")
        print("  Install: pip install peft>=0.11.0")
        return model, False


def merge_and_save(model, tokenizer, output_dir: str, lora_used: bool):
    """Merge LoRA weights into base model and save."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if lora_used:
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir)
            print(f"✓ LoRA merged model saved → {output_dir}")
        except Exception as e:
            print(f"LoRA merge failed ({e}), saving adapter instead.")
            model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Tokenizer saved → {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("  ViT5 QA Generator – Fine-tune with LoRA")
    print("=" * 60)
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LoRA rank  : {args.lora_rank}\n")

    # Tokenizer
    print("[1/5] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_tokens([SEP_TOKEN])          # add <sep> like Leaf English

    # Model
    print("[2/5] Loading model…")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    print("[3/5] Applying LoRA…")
    model, lora_used = apply_lora(model, rank=args.lora_rank, alpha=args.lora_rank * 2)

    # Datasets
    print("[4/5] Loading datasets…")
    train_data = load_split("train")
    val_data   = load_split("val")
    train_ds = QADataset(train_data, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
    val_ds   = QADataset(val_data,   tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
    )

    # Trainer
    print("[5/5] Starting training…")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    # Save merged model
    merge_and_save(model, tokenizer, args.output_dir, lora_used)
    print("\n✓ QA training complete!")
    print(f"  Model saved → {Path(args.output_dir).resolve()}")
    print("  Next step: python training/vn/train_vit5_distractor.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ViT5 for Vietnamese QA generation")
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--lora_rank",   type=int,   default=16)
    parser.add_argument("--fp16",        action="store_true", default=True)
    parser.add_argument("--no_fp16",     dest="fp16", action="store_false")
    args = parser.parse_args()
    main(args)
