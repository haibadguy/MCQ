# ViT5 Vietnamese Pipeline – Model Setup

## Mô hình cần tải

Cả 2 model đều tải **tự động từ HuggingFace** khi khởi động lần đầu (không cần tải thủ công):

| Vai trò | Model HuggingFace | Kích thước |
|---------|------------------|-----------|
| QA Generation | `namngo/pipeline-vit5-viquad-qg` | ~1.2 GB |
| Distractor Embeddings | `vinai/phobert-base-v2` | ~560 MB |

## Cài đặt

```bash
# Tạo môi trường ảo riêng (Python 3.9+)
python -m venv venv_vit5
.\venv_vit5\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements-vit5.txt
```

## Cách hoạt động (Option A)

```
Context (VI)
  ↓
ViT5 QA Generator (namngo/pipeline-vit5-viquad-qg)
  → (answer, question)
  ↓
PhoBERT Distractor Generator (vinai/phobert-base-v2)
  → Trích candidate phrases từ context
  → Rank bằng cosine similarity (0.15 < sim < 0.92)
  → Top 3 distractors
  ↓
Question object (1 đúng + 3 sai)
```

## Tham khảo

- **ViT5**: "ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation" (NAACL 2022, arXiv:2205.06457)
- **ViQAG**: "Towards Vietnamese Question and Answer Generation: An Empirical Study" (ACM TALLIP 2024)
- **PhoBERT**: "PhoBERT: Pre-trained language models for Vietnamese" (Findings of EMNLP 2020)
