# Demo Guide — Day 7 RAG Lab

**Nhóm:** Nguyen Hoang Minh, Duy, Nam, Dũng  
**Corpus:** 6 files trong `data/` (Internal Knowledge Assistant)

---

## Chuẩn bị

```bash
pip install -r requirements.txt
```

Tạo `.env` (copy từ `.env.example`):

```env
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
```

> Demo đầy đủ cần OpenAI API key cho embedding (`text-embedding-3-small`) và LLM (`gpt-4o-mini`).

---

## 1. Kiểm tra code (pytest)

```bash
pytest tests/ -v
```

Kỳ vọng: **42/42 passed**.

---

## 2. CLI demo (`main.py`)

```bash
python main.py
```

- Hỏi câu tùy ý trong terminal.
- Mặc định dùng strategy **Minh** (recursive, chunk_size=400).
- In top-k chunks + agent answer.

**Câu hỏi mẫu cho demo:**

| # | Câu hỏi | Kỳ vọng top-1 source |
|---|---------|----------------------|
| 1 | What is Python used for? | `python_intro.txt` |
| 2 | How does a vector store work? | `vector_store_notes.md` |
| 3 | What is the RAG system architecture? | `rag_system_design.md` |
| 4 | What are common customer support issues? | `customer_support_playbook.txt` |
| 5 | What are common retrieval failure cases in Vietnamese? | `vi_retrieval_notes.md` |

---

## 3. Gradio UI (`app.py`) — demo nhóm

```bash
python app.py
```

Mở trình duyệt: **http://127.0.0.1:7860**

### Cách trình bày (5–7 phút)

1. **Giới thiệu pipeline:** load docs → chunk (strategy) → embed → search → GPT-4o-mini.
2. **Chọn strategy** trong dropdown — mỗi thành viên giải thích method của mình:
   - **Minh** — Recursive character splitting (`chunk_size=400`, 43 chunks)
   - **Duy** — Parent/child chunking (parent 700 / child 220, 99 chunks)
   - **Nam** — Document-structure (markdown headings, 36 chunks)
   - **Dũng** — Semantic chunking (embedding similarity ≥ 0.55, 81 chunks)
3. **Chạy cùng câu hỏi** trên 2–3 strategy khác nhau → so sánh top-k scores và sources.
4. **Highlight Q3** (RAG architecture): Duy/Nam có top-3 sạch hơn (3/3 đúng doc) nhờ filter metadata.
5. **Highlight Q5** (Vietnamese): Nam đạt top-1 score cao nhất (0.614) với `language=vi` filter.

### Ví dụ so sánh trực tiếp trên UI

| Bước | Strategy | Câu hỏi | Điểm cần chỉ |
|------|----------|---------|--------------|
| A | Minh | What is the RAG system architecture? | Top-3 có 1 chunk `python_intro.txt` nhiễu |
| B | Duy | (cùng câu) | Top-3 toàn `rag_system_design.md`, score cao hơn |
| C | Nam | What are common retrieval failure cases in Vietnamese? | Top-1 score cao nhất trong nhóm |

### Failure case (tùy chọn)

- Đổi LLM Provider → `mock`, embedding mock → hỏi "What is Python used for?" → retrieval sai, agent từ chối trả lời → chứng minh lỗi ở retrieval layer.

---

## 4. So sánh 6 strategies (`compare_strategies.py`)

```bash
python scripts/compare_strategies.py
```

- Chạy 4 team strategies + 2 baselines trên cùng 5 benchmark queries.
- Dùng OpenAI embeddings; in bảng rubric và lưu `report/compare_results.json`.

---

## 5. Benchmark cá nhân (`run_benchmarks.py`)

```bash
python scripts/run_benchmarks.py
```

Chạy 5 queries với strategy Minh + OpenAI, in retrieval + agent answer.

---

## Kết quả so sánh (tóm tắt)

| Strategy | Chunks | Rubric /10 | Ghi chú |
|----------|--------|------------|---------|
| Minh — Recursive | 43 | 10 | Cân bằng, ít chunk nhất trong team |
| Duy — Parent/child | 99 | 10 | Q2/Q3 top-3 sạch với `department` filter |
| Nam — Doc-structure | 36 | 10 | Ít chunk nhất; Q5 top-1 score cao nhất |
| Dũng — Semantic | 81 | 10 | Chunk theo nghĩa; Q3 có nhiễu như Minh |
| Baseline fixed-size | 35 | 10 | Cắt cứng theo ký tự |
| Baseline sentence | 32 | 10 | Gộp theo câu |

*Tất cả 4 team strategies đạt 10/10 retrieval rubric trên 5 benchmark queries.*

---

## Files liên quan

| File | Mô tả |
|------|-------|
| `src/team_strategies.py` | Parent/child, doc-structure, semantic chunkers |
| `src/bootstrap.py` | Pipeline build store + agent theo `strategy_key` |
| `app.py` | Gradio UI |
| `report/REPORT.md` | Báo cáo đầy đủ |
| `report/compare_results.json` | Số liệu benchmark có thể tái tạo |
