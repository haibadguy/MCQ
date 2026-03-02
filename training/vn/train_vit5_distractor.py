"""
training/vn/train_vit5_distractor.py
======================================
Fine-tune VietAI/vit5-base cho Distractor generation (ViMMRC 2.0 + VSMRC).

Kiến trúc giống Leaf/Tangsang English Distractor:
  Input:  "{answer} <sep> {question} <sep> {context}"
  Target: "{d1} <sep> {d2} <sep> {d3}"

Cách chạy:
  python training/vn/train_vit5_distractor.py
  python training/vn/train_vit5_distractor.py --epochs 6 --batch_size 4 --output_dir app/ml_models/vit5_vietnamese/distractor_generator
"""

import argparse
import json
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
TARGET_MAX_LEN  = 64
DEFAULT_OUTPUT  = "app/ml_models/vit5_vietnamese/distractor_generator"
DATA_DIR        = Path("training/vn/data")


# ── Dataset ───────────────────────────────────────────────────────────────────

class DGDataset(Dataset):
    def __init__(self, records: list, tokenizer, source_max_len=512, target_max_len=64):
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


def load_split(name: str) -> list:
    path = DATA_DIR / f"dg_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python training/vn/prepare_dataset.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_lora(model, rank: int = 16, alpha: int = 32):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"],
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model, True
    except ImportError:
        print("⚠ peft not installed — training full model.")
        return model, False


def merge_and_save(model, tokenizer, output_dir: str, lora_used: bool):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if lora_used:
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(output_dir)
        except Exception as e:
            print(f"LoRA merge failed ({e}), saving adapter.")
            model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved → {output_dir}")


def main(args):
    print("=" * 60)
    print("  ViT5 Distractor Generator – Fine-tune with LoRA")
    print("=" * 60)
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_tokens([SEP_TOKEN])

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    model, lora_used = apply_lora(model, rank=args.lora_rank, alpha=args.lora_rank * 2)

    train_data = load_split("train")
    val_data   = load_split("val")
    train_ds = DGDataset(train_data, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
    val_ds   = DGDataset(val_data,   tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=300,
        weight_decay=0.01,
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

    merge_and_save(model, tokenizer, args.output_dir, lora_used)
    print("\n✓ Distractor training complete!")
    print(f"  Model saved → {Path(args.output_dir).resolve()}")
    print("  Next step: python main.py  (to demo both EN + VI)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ViT5 for Vietnamese distractor generation")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",     type=int,   default=6)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--lora_rank",  type=int,   default=16)
    parser.add_argument("--fp16",       action="store_true", default=True)
    parser.add_argument("--no_fp16",    dest="fp16", action="store_false")
    args = parser.parse_args()
    main(args)
