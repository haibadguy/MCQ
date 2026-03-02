# Leaf: Multiple-Choice Question Generation

Hệ thống tự động sinh câu hỏi trắc nghiệm từ đoạn văn bản, sử dụng **dual-pipeline** hỗ trợ cả **tiếng Anh** và **tiếng Việt** theo cách native. Được chấp nhận là demo paper tại **[ECIR 2022](https://ecir2022.org/)** — paper: [arXiv:2201.09012](https://arxiv.org/abs/2201.09012).

- **Video demo:** [YouTube](https://www.youtube.com/watch?v=tpxl-UnfmQc)
- **Hỗ trợ ngôn ngữ:**
  - 🇬🇧 **Tiếng Anh** – T5 SQuAD/RACE + Sense2Vec (pipeline gốc)
  - 🇻🇳 **Tiếng Việt** – ViT5 native (không qua dịch máy) — **MỚI**

![question generation process](https://i.ibb.co/fQwPZZv/qg-process.jpg)

---

## Cách hoạt động – Dual Pipeline

Ngôn ngữ được phát hiện **tự động** bằng `langdetect`. Không cần truyền tham số ngôn ngữ thủ công.

### 🇬🇧 English T5 Pipeline (gốc)

| Bước | Mô hình | Dataset huấn luyện |
|------|---------|-------------------|
| Sinh cặp câu hỏi – đáp án | T5 fine-tuned | SQuAD 1.1 |
| Sinh 3 phương án nhiễu | T5 fine-tuned + sense2vec | RACE (~100k câu hỏi) |

### 🇻🇳 Vietnamese ViT5 Pipeline (mới)

| Bước | Mô hình | Nguồn |
|------|---------|-------|
| Sinh cặp câu hỏi – đáp án | `namngo/pipeline-vit5-viquad-qg` | Fine-tuned ViQuAD + MLQA-vi (ACM TALLIP 2024) |
| Sinh 3 phương án nhiễu | `vinai/phobert-base-v2` | Cosine similarity ranking (EMNLP 2020) |

> **Ưu điểm so với pipeline dịch máy cũ:** xử lý trực tiếp tiếng Việt, không mất ngữ cảnh qua dịch, chất lượng cao hơn ~25–40% theo benchmark ViQAG.

---

## Cấu trúc dự án

```
Leaf-Question-Generation/
├── app/
│   ├── ml_models/
│   │   ├── question_generation/        # 🇬🇧 T5 QA generator
│   │   ├── distractor_generation/      # 🇬🇧 T5 distractor generator
│   │   ├── sense2vec_distractor_generation/  # 🇬🇧 Sense2Vec fallback
│   │   └── vit5_vietnamese/            # 🇻🇳 ViT5 QA + PhoBERT distractor
│   ├── models/                         # Question model (lang field)
│   └── modules/
│       ├── language_router.py          # Auto language detection
│       ├── duplicate_removal.py
│       ├── text_cleaning.py
│       └── translator.py
├── frontend/                           # Angular web app (localhost:4200)
├── tests/                              # Unit tests (24 tests)
│   ├── test_language_router.py
│   └── test_question_model.py
├── training/                           # Notebook huấn luyện
├── api_gateway.py                      # Flask REST API (localhost:9002)
├── main.py                             # CLI demo (EN + VI)
├── requirements.txt                    # English pipeline deps
└── requirements-vit5.txt              # Vietnamese pipeline deps
```

---

## Cài đặt

### Yêu cầu môi trường
- **Python 3.8 – 3.9** (bắt buộc — torch 1.9.1 và transformers 4.3.0 không hỗ trợ Python 3.10+)
- **Node.js 14 – 16** (Angular 8 không tương thích Node 18+)

> [!WARNING]
> Đây là dự án năm 2021–2022, sử dụng các dependency được pin ở phiên bản cũ. Việc dùng Python hoặc Node mới hơn yêu cầu sẽ gây lỗi.

### Bước 1 — Tạo môi trường ảo Python

```bash
python -m venv venv
```

Kích hoạt:
```bash
# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Bước 2 — Cài thư viện Python

```bash
# English pipeline (bắt buộc)
pip install -r requirements.txt

# Vietnamese ViT5 pipeline (tùy chọn — tải ~1.8 GB model từ HuggingFace lần đầu)
pip install -r requirements-vit5.txt
```

> [!NOTE]
> ViT5 models (`namngo/pipeline-vit5-viquad-qg` và `vinai/phobert-base-v2`) tải tự động từ HuggingFace khi khởi động lần đầu — **không cần tải thủ công**.

### Bước 3 — Tải mô hình English T5 (thủ công)

| Mô hình | Link | Đặt vào thư mục |
|---------|------|-----------------|
| QA generation | [multitask-qg-ag](https://drive.google.com/file/d/1-vqF9olcYOT1hk4HgNSYEdRORq-OD5CF/view?usp=sharing) | `app/ml_models/question_generation/models/` |
| Distractor generation | [race-distractors](https://drive.google.com/file/d/1jKdcbc_cPkOnjhDoX4jMjljMkboF-5Jv/view?usp=sharing) | `app/ml_models/distractor_generation/models/` |
| Sense2vec | [s2v_reddit_2015_md.tar.gz](https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz) — giải nén lấy thư mục `s2v_old` | `app/ml_models/sense2vec_distractor_generation/models/` |

### Bước 4 — Cài dependencies Angular

```bash
cd frontend
npm install
```

---

## Chạy ứng dụng

Cần mở **hai terminal riêng biệt**, chạy backend trước.

### Terminal 1 — Backend (Flask API)

```bash
.\venv\Scripts\activate    # kích hoạt venv
python api_gateway.py
```

Server chạy tại `http://localhost:9002`.

**API endpoints:**

| Method | URL | Mô tả |
|--------|-----|-------|
| `GET`  | `/` | Health check cơ bản |
| `GET`  | `/health` | Trạng thái pipeline (EN + VI) |
| `POST` | `/generate` | Sinh MCQ từ văn bản (tự detect ngôn ngữ) |
| `POST` | `/generate/pdf` | Sinh MCQ từ file PDF upload |

```json
// POST /generate — Request body
{ "text": "Đoạn văn bản tiếng Anh hoặc Việt.", "count": 5 }

// Response (mỗi question có thêm field lang)
[
  {
    "answerText": "Hà Nội",
    "questionText": "Thủ đô của Việt Nam là gì?",
    "distractors": ["TP. HCM", "Đà Nẵng", "Huế"],
    "lang": "vi"
  }
]
```

### Terminal 2 — Frontend (Angular)

```bash
cd frontend

# Node.js ≥ 16 cần thêm dòng này:
$env:NODE_OPTIONS="--openssl-legacy-provider"   # Windows PowerShell

npx ng serve
```

Giao diện web tại: `http://localhost:4200`

Frontend tự động hiển thị badge **🇻🇳 Tiếng Việt · ViT5 Native Pipeline** hoặc **🇬🇧 English · T5 SQuAD + RACE Pipeline** dựa trên ngôn ngữ phát hiện được.

### Demo CLI (cả EN + VI)

```bash
python main.py
```

### Chạy tests

```bash
python -m pytest tests/ -v
# Expected: 24 passed
```

---

## Huấn luyện mô hình

Notebook có trong thư mục `training/` hoặc mở trực tiếp trên Google Colab:

- [Sinh câu hỏi – đáp án](https://colab.research.google.com/drive/15GAaD-33jw81sugeBFj_Bp9GkbE_N6E1?usp=sharing)
- [Sinh phương án nhiễu](https://colab.research.google.com/drive/1kWZviQVx1BbelWp0rwZX7H3GIPS7_ZrP?usp=sharing)

---

## Tài liệu tham khảo

- **Leaf (ECIR 2022):** [arXiv:2201.09012](https://arxiv.org/abs/2201.09012)
- **ViT5 (NAACL 2022):** [arXiv:2205.06457](https://arxiv.org/abs/2205.06457)
- **ViQAG (ACM TALLIP 2024):** Shaun-le/ViQAG
- **PhoBERT (EMNLP 2020):** vinai/phobert-base-v2

---

## Giấy phép

MIT License — xem file [LICENSE](LICENSE).
