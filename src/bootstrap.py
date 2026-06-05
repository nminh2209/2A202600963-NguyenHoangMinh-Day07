from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import KnowledgeBaseAgent
from .embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from .chunking import RecursiveChunker, SentenceChunker
from .llm import LLM_PROVIDER_ENV, create_llm_fn
from .models import Document
from .store import EmbeddingStore

DEFAULT_SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            continue
        if not path.exists() or not path.is_file():
            continue

        content = path.read_text(encoding="utf-8")
        extension = path.suffix.lower()
        category = "technical" if extension == ".md" else "reference"
        language = "vi" if "vi_" in path.name else "en"
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={
                    "source": str(path),
                    "extension": extension,
                    "category": category,
                    "language": language,
                    "department": "engineering" if "rag" in path.name or "vector" in path.name else "support",
                },
            )
        )

    return documents


def chunk_documents(
    documents: list[Document],
    strategy: str = "recursive",
    chunk_size: int = 400,
    max_sentences: int = 2,
) -> list[Document]:
    """Split source files into smaller chunks before indexing."""
    if strategy == "sentence":
        chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
    else:
        chunker = RecursiveChunker(chunk_size=chunk_size)

    chunked: list[Document] = []

    for document in documents:
        pieces = chunker.chunk(document.content)
        for index, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            metadata["doc_id"] = document.id
            metadata["chunk_index"] = index
            chunked.append(
                Document(
                    id=f"{document.id}_chunk_{index}",
                    content=piece,
                    metadata=metadata,
                )
            )

    return chunked


def get_embedder(provider: str | None = None, api_key: str | None = None):
    load_dotenv(override=False)
    resolved_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    selected = (provider or os.getenv(EMBEDDING_PROVIDER_ENV, "mock")).strip().lower()

    # Use real embeddings when an OpenAI key is available for the live demo.
    if selected == "mock" and resolved_key:
        llm_provider = os.getenv(LLM_PROVIDER_ENV, "mock").strip().lower()
        if llm_provider == "openai":
            selected = "openai"

    if selected == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if selected == "openai":
        if not resolved_key:
            return _mock_embed
        try:
            return OpenAIEmbedder(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL),
                api_key=resolved_key,
            )
        except Exception:
            return _mock_embed
    return _mock_embed


def build_rag_system(
    sample_files: list[str] | None = None,
    llm_provider: str | None = None,
    api_key: str | None = None,
    embedding_provider: str | None = None,
    collection_name: str = "rag_demo_store",
    chunk_strategy: str = "recursive",
    chunk_size: int = 400,
    max_sentences: int = 2,
) -> tuple[EmbeddingStore, KnowledgeBaseAgent]:
    files = sample_files or DEFAULT_SAMPLE_FILES
    docs = load_documents_from_files(files)
    if not docs:
        raise ValueError("No documents could be loaded from the sample file list.")

    embedder = get_embedder(embedding_provider, api_key=api_key)
    chunks = chunk_documents(
        docs,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        max_sentences=max_sentences,
    )
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    store.add_documents(chunks)

    llm_fn = create_llm_fn(provider=llm_provider, api_key=api_key)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    return store, agent


def format_search_results(results: list[dict], limit: int = 220) -> str:
    if not results:
        return "_No chunks retrieved._"

    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        source = item.get("metadata", {}).get("source", "unknown")
        preview = item["content"][:limit].replace("\n", " ")
        lines.append(
            f"**{index}.** score={item['score']:.3f} | source={source}\n"
            f"> {preview}{'...' if len(item['content']) > limit else ''}"
        )
    return "\n\n".join(lines)
