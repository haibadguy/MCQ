# Training – Vietnamese ViT5 Pipeline

Hướng dẫn train ViT5 local cho Vietnamese MCQ generation.

## Kiến trúc (giống Leaf/Tangsang English)

```
Stage 1 – QA Generation:
  Input:  "[MASK] <sep> {context}"
  Output: "{answer} <sep> {question}"
  Model:  VietAI/vit5-base + LoRA → app/ml_models/vit5_vietnamese/qa_generator/

Stage 2 – Distractor Generation:
  Input:  "{answer} <sep> {question} <sep> {context}"
  Output: "{d1} <sep> {d2} <sep> {d3}"
  Model:  VietAI/vit5-base + LoRA → app/ml_models/vit5_vietnamese/distractor_generator/
```

## Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| GPU VRAM | 8 GB | 12 GB (RTX 3060/4060) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 40 GB |
| CUDA | 11.8+ | 12.1+ |

## Cài đặt

```bash
pip install -r requirements-vit5.txt
```

## Dataset

| Dataset | Task | Nguồn | Mẫu |
|---------|------|-------|-----|
| ViQuAD 2.0 | QA generation | `uitnlp/viquad2` (HuggingFace) | ~23k |
| ViMMRC 2.0 | Distractor | `nlp.uit.edu.vn/datasets` | ~2k |
| VSMRC | Distractor | `arXiv:2506.15978` | ~16k |

## Giai đoạn 3: Chuẩn bị dataset

```bash
# Auto-download ViQuAD từ HuggingFace + local ViMMRC/VSMRC
python training/vn/prepare_dataset.py

# Nếu có file ViMMRC/VSMRC sẵn
python training/vn/prepare_dataset.py \
  --vimmrc_path path/to/vimmrc.json \
  --vsmrc_path  path/to/vsmrc.json

# Output
training/vn/data/qa_train.json   (~18k mẫu)
training/vn/data/qa_val.json     (~2k)
training/vn/data/qa_test.json    (~2k)
training/vn/data/dg_train.json   (~13k mẫu)
training/vn/data/dg_val.json     (~1.5k)
training/vn/data/dg_test.json    (~1.5k)
```

## Giai đoạn 4: Train QA generator

```bash
python training/vn/train_vit5_qa.py

# Tùy chỉnh (8GB VRAM → batch_size=4)
python training/vn/train_vit5_qa.py \
  --epochs 5 \
  --batch_size 8 \
  --lora_rank 16 \
  --output_dir app/ml_models/vit5_vietnamese/qa_generator

# Thời gian ước tính: ~4-6h (RTX 3060, 5 epochs, ~18k mẫu)
```

## Giai đoạn 5: Train Distractor generator

```bash
python training/vn/train_vit5_distractor.py

# Tùy chỉnh (VRAM ít hơn → batch_size=4)
python training/vn/train_vit5_distractor.py \
  --epochs 6 \
  --batch_size 4 \
  --lora_rank 16 \
  --output_dir app/ml_models/vit5_vietnamese/distractor_generator

# Thời gian ước tính: ~3-5h (RTX 3060, 6 epochs, ~13k mẫu)
```

## Sau khi train xong

```bash
# Kiểm tra models đã có
ls app/ml_models/vit5_vietnamese/qa_generator/
ls app/ml_models/vit5_vietnamese/distractor_generator/

# Demo cả EN + VI
python main.py

# Khởi động server
.\venv\Scripts\python.exe api_gateway.py
```

## Fallback (trước khi train xong)

Nếu chưa có local model:
- **QA**: tự động dùng `namngo/pipeline-vit5-viquad-qg` từ HuggingFace
- **Distractor**: dùng heuristic n-gram extraction

Pipeline vẫn hoạt động, chỉ chất lượng thấp hơn.
