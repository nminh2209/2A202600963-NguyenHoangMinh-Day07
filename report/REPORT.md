# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyen Hoang Minh  
**Nhóm:** Day07-RAG — Nguyen Hoang Minh & Duy  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector embedding có cosine similarity cao khi chúng hướng gần nhau trong không gian vector — tức là hai đoạn text mang ý nghĩa tương tự hoặc liên quan chặt chẽ, dù có thể khác từ ngữ.

**Ví dụ HIGH similarity:**
- Sentence A: *Vector databases store embeddings for similarity search.*
- Sentence B: *A vector store keeps numerical representations of text for semantic retrieval.*
- Tại sao tương đồng: Cả hai đều mô tả cùng khái niệm vector store / embedding storage với từ vựng gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: *The cat sat on the mat.*
- Sentence B: *Stock prices rose sharply today.*
- Tại sao khác: Hai câu thuộc chủ đề hoàn toàn khác nhau, không chia sẻ ngữ cảnh hay ý nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Embeddings thường được chuẩn hóa và quan trọng hơn ở *hướng* (semantic direction) hơn độ dài vector. Cosine similarity đo góc giữa hai vector nên ổn định hơn khi độ dài biến thiên; Euclidean distance dễ bị ảnh hưởng bởi magnitude mà không phản ánh tốt mức độ liên quan ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> Thay số: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`  
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25 chunks` — tăng từ 23 lên 25. Overlap lớn hơn giúp các chunk liền kề chia sẻ ngữ cảnh, giảm nguy cơ cắt đứt một ý quan trọng nằm ở ranh giới chunk (ví dụ một bước trong quy trình troubleshooting).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Internal Knowledge Assistant — tài liệu kỹ thuật, RAG design, vector store, customer support playbook (song ngữ Anh–Việt)

**Tại sao nhóm chọn domain này?**
> Domain này sát với nội dung lab (embedding, chunking, retrieval, RAG agent) và mô phỏng bài toán thực tế: nhân viên cần tra cứu runbook, thiết kế hệ thống, và hướng dẫn support. Bộ dữ liệu có cả tiếng Anh lẫn tiếng Việt nên phù hợp để thử metadata filter và failure cases đa ngôn ngữ.

> **Lưu ý:** Nhóm dùng 6 tài liệu mẫu có sẵn trong `data/` của lab làm corpus thống nhất, gán metadata schema nhóm thiết kế, thay vì thu thập tài liệu ngoài mới. Điều này đủ để chạy benchmark nhưng hạn chế điểm *Document Set Quality* vì nguồn chưa phải tài liệu nhóm tự thu thập.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | Sample data (lab) | 1,944 | `source`, `extension`, `category=reference`, `language=en`, `department=support` |
| 2 | vector_store_notes.md | Sample data (lab) | 2,123 | `source`, `extension`, `category=technical`, `language=en`, `department=engineering` |
| 3 | rag_system_design.md | Sample data (lab) | 2,391 | `source`, `extension`, `category=technical`, `language=en`, `department=engineering` |
| 4 | customer_support_playbook.txt | Sample data (lab) | 1,692 | `source`, `extension`, `category=reference`, `language=en`, `department=support` |
| 5 | chunking_experiment_report.md | Sample data (lab) | 1,987 | `source`, `extension`, `category=technical`, `language=en`, `department=support` |
| 6 | vi_retrieval_notes.md | Sample data (lab) | 1,667 | `source`, `extension`, `category=technical`, `language=vi`, `department=support` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `language` | string | `en`, `vi` | Lọc tài liệu đúng ngôn ngữ khi câu hỏi tiếng Việt hoặc tiếng Anh |
| `department` | string | `engineering`, `support` | Thu hẹp phạm vi tìm kiếm theo phòng ban (kỹ thuật vs support) |
| `category` | string | `technical`, `reference` | Phân biệt tài liệu thiết kế hệ thống với tài liệu hướng dẫn ngắn |
| `source` | string | `data/rag_system_design.md` | Truy vết nguồn chunk khi review kết quả retrieval |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=200)` trên 3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | FixedSizeChunker (`fixed_size`) | 10 | 194.4 | Trung bình — dễ cắt giữa câu |
| python_intro.txt | SentenceChunker (`by_sentences`) | 5 | 387.0 | Tốt — giữ ranh giới câu |
| python_intro.txt | RecursiveChunker (`recursive`) | 15 | 127.8 | Tốt nhất — ưu tiên đoạn, kích thước đều hơn |
| vector_store_notes.md | FixedSizeChunker (`fixed_size`) | 11 | 193.0 | Trung bình |
| vector_store_notes.md | SentenceChunker (`by_sentences`) | 8 | 263.6 | Khá tốt |
| vector_store_notes.md | RecursiveChunker (`recursive`) | 19 | 109.9 | Tốt — tách theo heading/đoạn |
| rag_system_design.md | FixedSizeChunker (`fixed_size`) | 12 | 199.2 | Trung bình |
| rag_system_design.md | SentenceChunker (`by_sentences`) | 5 | 476.0 | Chunk dài, có thể quá dense |
| rag_system_design.md | RecursiveChunker (`recursive`) | 21 | 112.0 | Tốt nhất — giữ section markdown |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> RecursiveChunker thử lần lượt các separator theo thứ tự ưu tiên (`\n\n` → `\n` → `. ` → ` ` → ký tự). Nếu một mảnh vẫn dài hơn `chunk_size`, nó đệ quy xuống separator nhỏ hơn; nếu hết separator thì cắt theo ký tự. Các mảnh nhỏ được gộp lại miễn là tổng độ dài không vượt `chunk_size`.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu nhóm gồm markdown có heading, đoạn văn, và playbook nhiều câu. Recursive chunking tôn trọng cấu trúc đoạn trước khi cắt nhỏ, giúp mỗi chunk mang đủ ngữ cảnh cho retrieval — đặc biệt với runbook và thiết kế RAG có nhiều section.

**Code snippet (nếu custom):**
```python
# Dùng RecursiveChunker có sẵn — benchmark & demo dùng chunk_size=400
chunker = RecursiveChunker(chunk_size=400)
chunks = chunker.chunk(document_text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| rag_system_design.md | best baseline (SentenceChunker) | 5 | 476.0 | Khá tốt nhưng chunk dài |
| rag_system_design.md | **RecursiveChunker (của tôi)** | 21 | 112.0 | Tốt hơn — chunk vừa phải, giữ section |

### So Sánh Với Thành Viên Khác

*Bảng dưới là kết quả chạy bằng `python scripts/compare_strategies.py` trên cùng corpus (6 file trong `data/`) và cùng 5 benchmark queries; dùng `text-embedding-3-small`.*

| Thành viên | Strategy | Chunks indexed | Retrieval rubric | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------|------------------|-----------|----------|
| Minh | `RecursiveChunker` (`chunk_size=400`) | 43 | **10 / 10** | Top-3 relevant 5/5; chunk giữ section markdown | Q3 top-3 có 1 chunk `python_intro.txt` nhiễu (2/3 đúng doc) |
| Duy | `SentenceChunker` (`max_sentences=2`) + `search_with_filter` | 49 | **10 / 10** | Q3 top-3 **3/3** từ `rag_system_design.md`; filter `engineering` tăng score Q2 (0.726 vs 0.692) | 49 chunks → index lớn hơn; chunk theo câu khó giữ ngữ cảnh đoạn dài |

**Chi tiết từng query (Minh vs Duy):**

| # | Query | Minh top-1 (score) | Duy top-1 (score) | Duy filter | Ai lợi thế? |
|---|-------|-------------------|-------------------|------------|-------------|
| 1 | Python used for? | `python_intro.txt` (0.701) | `python_intro.txt` (0.701) | none | Hòa |
| 2 | Vector store workflow? | `vector_store_notes.md` (0.692) | `vector_store_notes.md` (**0.726**) | `department=engineering` | **Duy** — filter loại support docs |
| 3 | RAG architecture? | `rag_system_design.md` (0.544), top-3: 2/3 đúng doc | `rag_system_design.md` (0.544), top-3: **3/3** đúng doc | `department=engineering` | **Duy** — không lẫn `python_intro.txt` |
| 4 | Support issues? | `customer_support_playbook.txt` (0.473) | `customer_support_playbook.txt` (0.473) | `department=support` | Hòa |
| 5 | VN failure cases? | `vi_retrieval_notes.md` (0.587) | `vi_retrieval_notes.md` (0.521) | `language=vi` | **Minh** — score cao hơn khi cùng filter |

**Cách Duy tiếp cận:**
> Duy dùng `SentenceChunker(max_sentences=2)` và áp `metadata_filter` trước khi search (ví dụ `language=vi`, `department=engineering`, `department=support`) để giảm nhiễu.

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Cả Minh và Duy đều đạt **10/10 retrieval rubric** trên cùng 5 queries. Khác nhau chủ yếu ở cách quản lý nhiễu: Duy lọc metadata giúp Q2/Q3 top-3 “sạch” hơn, còn Minh ít chunk hơn và đạt top-1 score cao ở Q5.

Trong thảo luận nhóm còn có Dũng và Nam; trong báo cáo này, mình tập trung số liệu chạy có thể kiểm tra lại theo rubric (Minh vs Duy).

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])(?:\s+|\n)` để tách sau dấu kết thúc câu (`. `, `! `, `? `, `.\n`). Các câu được strip whitespace, rồi gộp theo nhóm `max_sentences_per_chunk`. Edge case: text rỗng trả về `[]`; text không có dấu câu vẫn được coi là một câu duy nhất.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Base case: text rỗng → `[]`; text ngắn hơn `chunk_size` → `[text]`. Nếu separator hiện tại không có trong text, đệ quy sang separator tiếp theo. Nếu separator là `""`, cắt cứng theo ký tự. Khi split được, các phần nhỏ được gộp; phần quá dài được đệ quy tiếp.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi `Document` được embed qua `embedding_fn`, lưu record `{id, content, embedding, metadata}` vào in-memory list (hoặc ChromaDB nếu có). Search embed câu query, tính dot product với từng vector đã lưu (mock embedder đã normalize nên dot product tương đương cosine similarity), sort giảm dần, trả top-k kèm `content`, `metadata`, `score`.

**`search_with_filter` + `delete_document`** — approach:
> Filter **trước** khi search: chỉ giữ records có metadata khớp toàn bộ điều kiện filter, rồi chạy `_search_records` trên tập đã lọc. Delete quét store, xóa mọi record có `id == doc_id` hoặc `metadata['doc_id'] == doc_id`, trả `True` nếu có ít nhất một bản ghi bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)`, ghép nội dung các chunk bằng `\n\n` làm context, build prompt dạng *Context + Question + Answer:*, rồi gọi `llm_fn(prompt)`. Với OpenAI, `llm_fn` dùng `gpt-4o-mini` và system message yêu cầu chỉ trả lời dựa trên context.

**Demo / benchmark pipeline** (`src/bootstrap.py`):
> Trước khi index: `RecursiveChunker(chunk_size=400)` chia mỗi file thành nhiều chunk, gán `doc_id` + `chunk_index` vào metadata. Embedder: `text-embedding-3-small` khi có `OPENAI_API_KEY`. Kết quả benchmark: **43 chunks** indexed từ 6 file gốc.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-8.4.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED

============================= 42 passed in 0.17s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*(Dùng `MockEmbedder` — deterministic, không phản ánh ngữ nghĩa thật nhưng hữu ích để test pipeline.)*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a programming language. | Python is widely used for software development. | high | 0.109 | Một phần — score dương nhưng thấp |
| 2 | The cat sat on the mat. | Stock prices rose sharply today. | low | -0.228 | Có — score âm, rất thấp |
| 3 | Vector databases store embeddings. | A vector store keeps numerical representations of text. | high | 0.190 | Có — score dương cao nhất trong bảng |
| 4 | How do I reset my password? | Password recovery requires email verification. | high | -0.206 | Không — mock embed cho score âm |
| 5 | Machine learning uses neural networks. | Deep learning models have many layers. | high | -0.159 | Không — mock embed cho score âm |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Trong mock embed, cặp 4 và 5 có score âm dù hai câu có chủ đề gần nhau. Mock embed là deterministic fallback nên không đảm bảo phản ánh semantic similarity như embedding thật. Khi dùng OpenAI embeddings, retrieval khớp với tài liệu hơn.

---

## 6. Results — Cá nhân (10 điểm)

**Cấu hình benchmark (setup hiện tại):**
- Chunking: `RecursiveChunker(chunk_size=400)` → **43 chunks** từ 6 file
- Embeddings: `text-embedding-3-small` (`EMBEDDING_PROVIDER=openai`)
- LLM: `gpt-4o-mini` (`LLM_PROVIDER=openai`)
- Chạy qua: `python scripts/run_benchmarks.py` / `python main.py`

### Benchmark Queries & Gold Answers (nhóm thống nhất — Minh & Duy)

| # | Query | Gold Answer | Retrieval method |
|---|-------|-------------|------------------|
| 1 | What is Python used for? | Python dùng cho automation, backend/API, data analysis, scientific computing, và ML; phổ biến với FastAPI, Django, pandas, PyTorch. | `search()` |
| 2 | How does a vector store work? | Pipeline: chunk → embed → store vector + metadata → embed query → rank by similarity. | `search()` |
| 3 | What is the RAG system architecture? | Ingestion pipeline (chunk + metadata) → retrieval layer (embed + search + filter) → application layer (prompt + LLM, grounded answers). | `search()` |
| 4 | What are common customer support issues? | Account setup, billing errors, password recovery, service limitations; cần nội dung cụ thể và metadata phân loại audience. | `search()` |
| 5 | What are common retrieval failure cases in Vietnamese? | Tài liệu cũ xếp hạng cao, keyword mismatch, embedding xử lý kém nội dung song ngữ; cần test truy vấn thực tế và ghi failure cases. | `search_with_filter(metadata_filter={"language": "vi"})` |

### Kết Quả Của Tôi (OpenAI embeddings + chunked index)

| # | Query | Top-3 Sources (score) | Relevant in top-3? | Agent answer grounded? |
|---|-------|----------------------|--------------------|------------------------|
| 1 | What is Python used for? | `python_intro.txt` (0.701, 0.623, 0.559) | **Có** — cả 3 chunk từ đúng tài liệu | **Có** — automation, backend, data analysis, ML, APIs |
| 2 | How does a vector store work? | `vector_store_notes.md` (0.692, 0.513, 0.459) | **Có** — cả 3 chunk từ đúng tài liệu | **Có** — 4 bước: chunk, embed, store, query+rank |
| 3 | What is the RAG system architecture? | `rag_system_design.md` (0.544, 0.360), `python_intro.txt` (0.324) | **Có** — 2/3 từ rag_system_design | **Có** — ingestion, retrieval, application layer |
| 4 | What are common customer support issues? | `customer_support_playbook.txt` (0.473, 0.405, 0.402) | **Có** — cả 3 chunk từ playbook | **Có** — account setup, billing, password, service limits |
| 5 | Vietnamese failure cases *(filtered)* | `vi_retrieval_notes.md` (0.587, 0.461, 0.458) | **Có** — cả 3 chunk tiếng Việt đúng nguồn | **Có** — tài liệu cũ, keyword mismatch, song ngữ |

**Chi tiết Top-1 từng query:**

| # | Top-1 preview | Score |
|---|---------------|-------|
| 1 | *Python is a high-level programming language widely used for automation, backend services, data analysis...* | 0.701 |
| 2 | *A vector store is a database or storage layer designed to keep embeddings and retrieve the most similar items...* | 0.692 |
| 3 | *RAG System Design for an Internal Knowledge Assistant — Background: product team wants an assistant...* | 0.544 |
| 4 | *The support team uses the knowledge assistant to answer repetitive questions about account setup, billing errors...* | 0.473 |
| 5 | *Trong thực tế, retrieval không phải lúc nào cũng đúng. Một số lỗi thường gặp là tài liệu cũ vẫn xếp hạng cao...* | 0.587 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5 / 5**

**Điểm retrieval theo rubric (`SCORING.md`):** ước tính **10 / 10** (mỗi query có relevant chunk trong top-3 và agent answer khớp gold answer).

### So sánh với baseline lỗi (mock embed — failure case tham chiếu)

Trước khi sửa pipeline, cùng query #1 với `MockEmbedder` + index nguyên file (6 documents):

| Query | Top-1 (mock) | Score | Kết quả |
|-------|--------------|-------|---------|
| What is Python used for? | `chunking_experiment_report.md` | 0.117 | **Không relevant** — agent GPT-4o-mini từ chối trả lời vì context sai |

Điều này chứng minh retrieval layer quan trọng hơn LLM: GPT-4o-mini hành xử đúng (không hallucinate) nhưng bị fail vì retrieve sai chunk.

---

## 7. What I Learned (5 điểm — Demo)

### Exercise 3.5 — Failure Analysis

**Failure case quan sát được (thực tế từ CLI/UI):**

| Mục | Chi tiết |
|-----|----------|
| **Query** | `What is Python used for?` |
| **Triệu chứng** | Top-1 = `chunking_experiment_report.md` (score 0.117), **không** phải `python_intro.txt`. Agent GPT-4o-mini trả lời: *"The provided context does not contain information about the uses of Python."* |
| **Cấu hình lỗi** | `MockEmbedder` + index **nguyên file** (6 documents, không chunk) |
| **Nguyên nhân** | (1) Mock embed hash-based không hiểu ngữ nghĩa "Python"; (2) whole-file index làm chunk quá lớn, signal bị loãng; (3) LLM **đúng** khi từ chối — lỗi nằm ở retrieval, không phải generation |
| **Cách sửa đã áp dụng** | `RecursiveChunker(400)` → 43 chunks + `text-embedding-3-small` → top-3 đều từ `python_intro.txt`, score 0.55–0.70, answer grounded |
| **Bài học** | Cần kiểm tra retrieval trước khi đổ lỗi cho LLM. Chất lượng chunk/embed ảnh hưởng trực tiếp đến kết quả RAG. |

**Failure case thứ hai (query ngoài corpus):**

| Mục | Chi tiết |
|-----|----------|
| **Query** | `What is Javascripts used for?` |
| **Triệu chứng** | Top-1 = `python_intro.txt` (mock, score 0.077) — false positive gần nhau về điểm số |
| **Kết quả agent** | Từ chối trả lời đúng — không có tài liệu JavaScript trong corpus |
| **Đề xuất** | Thêm metadata `topic` + kiểm tra score threshold; nếu top score < ngưỡng thì trả "không đủ dữ liệu" thay vì đưa context nhiễu |

---

**Điều Duy nêu trong nhóm:**
> Trong benchmark `compare_strategies.py`, khi lọc `department=engineering`, Q3 top-3 không còn chunk `python_intro.txt` nhiễu và tập trung vào `rag_system_design.md`.

**Điều nhóm khác trao đổi (qua demo):**
> Một nhóm khác chia sẻ rằng khi dùng embedding thật (không dùng mock), retrieval thường khớp nội dung tốt hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Thu thập 2–3 tài liệu ngoài lab để tăng tính "nhóm tự chuẩn bị". (2) Thêm `last_updated` và `topic` vào metadata. (3) Tích hợp filter vào agent khi query có ngôn ngữ rõ (theo hướng Duy). (4) Luôn log top-k + score khi demo để failure analysis có hệ thống.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá | Lý do |
|----------|------|-------------------|-------|
| Warm-up | Cá nhân | 5 / 5 | Đủ cosine explanation + chunking math |
| Document selection | Nhóm | 6 / 10 | Corpus = lab samples, chưa tự thu thập ngoài |
| Chunking strategy | Nhóm | 14 / 15 | So sánh benchmark thật Minh vs Duy (`compare_strategies.py`); chưa custom chunker |
| My approach | Cá nhân | 9 / 10 | Core `src/` đầy đủ; demo pipeline ở `bootstrap.py` |
| Similarity predictions | Cá nhân | 4 / 5 | 3/5 dự đoán đúng với mock; ghi chú kết quả đủ |
| Results | Cá nhân | 9 / 10 | 5/5 relevant top-3 với OpenAI setup; có failure baseline |
| Core implementation (tests) | Cá nhân | 30 / 30 | 42/42 pytest pass |
| Demo | Nhóm | 4 / 5 | Có Gradio UI + CLI + bảng so sánh strategy |
| **Tổng** | | **86 / 100** | |
