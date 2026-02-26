# Leaf: Multiple-Choice Question Generation

Hệ thống tự động sinh câu hỏi trắc nghiệm từ đoạn văn bản, sử dụng hai mô hình **T5 Transformer** fine-tuned kết hợp sense2vec. Được chấp nhận là demo paper tại **[ECIR 2022](https://ecir2022.org/)** — paper: [arXiv:2201.09012](https://arxiv.org/abs/2201.09012).

- **Video demo:** [YouTube](https://www.youtube.com/watch?v=tpxl-UnfmQc)
- **Hỗ trợ ngôn ngữ:** Tiếng Anh (T5 trực tiếp) · Tiếng Việt (tự động phát hiện, dịch → sinh → dịch ngược)

![question generation process](https://i.ibb.co/fQwPZZv/qg-process.jpg)

---

## Cách hoạt động

| Bước | Mô hình | Dataset huấn luyện |
|------|---------|-------------------|
| Sinh cặp câu hỏi – đáp án | T5 fine-tuned | SQuAD 1.1 |
| Sinh 3 phương án nhiễu | T5 fine-tuned + sense2vec | RACE (~100k câu hỏi) |

Mỗi câu hỏi đầu ra luôn có **đúng 4 đáp án** (1 đúng + 3 sai). Nếu input là tiếng Việt, hệ thống tự động dịch sang tiếng Anh trước khi xử lý, sau đó dịch kết quả ngược lại.

---

## Cấu trúc dự án

```
Leaf-Question-Generation/
├── app/
│   ├── ml_models/          # Các mô hình AI (QG, DG, sense2vec)
│   ├── models/             # Data models (Question)
│   └── modules/            # Utilities (translator, text_cleaning, ...)
├── frontend/               # Angular web app (chạy tại localhost:4200)
├── training/               # Notebook huấn luyện mô hình
├── api_gateway.py          # Flask REST API (chạy tại localhost:9002)
├── main.py                 # Chạy thử nhanh từ terminal
└── requirements.txt
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
pip install -r requirements.txt
```

### Bước 3 — Tải mô hình AI

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
# Kích hoạt venv (nếu chưa)
.\venv\Scripts\activate

python api_gateway.py
```

Server chạy tại `http://localhost:9002`. **Giữ terminal này mở.**

**API endpoint:**

| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/` | Kiểm tra server |
| `POST` | `/generate` | Sinh câu hỏi từ văn bản |

```json
// POST /generate — Request body
{ "text": "Đoạn văn bản đầu vào.", "count": 5 }
```

### Terminal 2 — Frontend (Angular)

```bash
cd frontend

# Node.js ≥ 16 cần thêm dòng này:
$env:NODE_OPTIONS="--openssl-legacy-provider"   # Windows PowerShell

npx ng serve
```

Giao diện web tại: `http://localhost:4200`

### Chạy thử nhanh từ terminal (không cần frontend)

```bash
python main.py
```

---

## Huấn luyện mô hình

Notebook có trong thư mục `training/` hoặc mở trực tiếp trên Google Colab:

- [Sinh câu hỏi – đáp án](https://colab.research.google.com/drive/15GAaD-33jw81sugeBFj_Bp9GkbE_N6E1?usp=sharing)
- [Sinh phương án nhiễu](https://colab.research.google.com/drive/1kWZviQVx1BbelWp0rwZX7H3GIPS7_ZrP?usp=sharing)

---

## Giấy phép

MIT License — xem file [LICENSE](LICENSE).
