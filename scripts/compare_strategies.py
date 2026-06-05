"""
Compare Minh (RecursiveChunker) vs Duy (SentenceChunker + metadata filters).
Run: python scripts/compare_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.bootstrap import build_rag_system

BENCHMARKS = [
    {
        "id": 1,
        "query": "What is Python used for?",
        "gold_source": "python_intro.txt",
        "minh_filter": None,
        "duy_filter": None,
    },
    {
        "id": 2,
        "query": "How does a vector store work?",
        "gold_source": "vector_store_notes.md",
        "minh_filter": None,
        "duy_filter": {"department": "engineering"},
    },
    {
        "id": 3,
        "query": "What is the RAG system architecture?",
        "gold_source": "rag_system_design.md",
        "minh_filter": None,
        "duy_filter": {"department": "engineering"},
    },
    {
        "id": 4,
        "query": "What are common customer support issues?",
        "gold_source": "customer_support_playbook.txt",
        "minh_filter": None,
        "duy_filter": {"department": "support"},
    },
    {
        "id": 5,
        "query": "What are common retrieval failure cases in Vietnamese?",
        "gold_source": "vi_retrieval_notes.md",
        "minh_filter": {"language": "vi"},
        "duy_filter": {"language": "vi"},
    },
]


def source_name(metadata: dict) -> str:
    return Path(metadata.get("source", "unknown")).name


def is_relevant(result: dict, gold_source: str) -> bool:
    return gold_source in source_name(result.get("metadata", {}))


def search(store, query: str, metadata_filter: dict | None, top_k: int = 3) -> list[dict]:
    if metadata_filter:
        return store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
    return store.search(query, top_k=top_k)


def score_run(store, spec: dict, member: str) -> dict:
    metadata_filter = spec["duy_filter"] if member == "duy" else spec["minh_filter"]
    results = search(store, spec["query"], metadata_filter)
    relevant_count = sum(1 for item in results if is_relevant(item, spec["gold_source"]))
    top1 = results[0] if results else None
    return {
        "query_id": spec["id"],
        "query": spec["query"],
        "filter": metadata_filter,
        "top1_source": source_name(top1["metadata"]) if top1 else "none",
        "top1_score": top1["score"] if top1 else 0.0,
        "top1_relevant": is_relevant(top1, spec["gold_source"]) if top1 else False,
        "relevant_in_top3": relevant_count,
        "rubric_pts": 2 if relevant_count >= 1 and (top1 and is_relevant(top1, spec["gold_source"])) else (
            1 if relevant_count >= 1 else 0
        ),
        "top3": [
            f"{source_name(item['metadata'])} ({item['score']:.3f})" for item in results
        ],
    }


def main() -> None:
    minh_store, _ = build_rag_system(
        collection_name="compare_minh",
        llm_provider="mock",
        chunk_strategy="recursive",
        chunk_size=400,
    )
    duy_store, _ = build_rag_system(
        collection_name="compare_duy",
        llm_provider="mock",
        chunk_strategy="sentence",
        max_sentences=2,
    )

    print(f"MINH_CHUNKS={minh_store.get_collection_size()}")
    print(f"DUY_CHUNKS={duy_store.get_collection_size()}")

    minh_total = 0
    duy_total = 0

    for spec in BENCHMARKS:
        minh = score_run(minh_store, spec, "minh")
        duy = score_run(duy_store, spec, "duy")
        minh_total += minh["rubric_pts"]
        duy_total += duy["rubric_pts"]

        print(f"\nQ{spec['id']}: {spec['query']}")
        print(
            f"  MINH filter={minh['filter']} top1={minh['top1_source']} "
            f"score={minh['top1_score']:.3f} rel_top1={minh['top1_relevant']} "
            f"rel_top3={minh['relevant_in_top3']}/3 rubric={minh['rubric_pts']}/2"
        )
        print(f"       top3={minh['top3']}")
        print(
            f"  DUY  filter={duy['filter']} top1={duy['top1_source']} "
            f"score={duy['top1_score']:.3f} rel_top1={duy['top1_relevant']} "
            f"rel_top3={duy['relevant_in_top3']}/3 rubric={duy['rubric_pts']}/2"
        )
        print(f"       top3={duy['top3']}")

    print(f"\nMINH_RETRIEVAL_RUBRIC={minh_total}/10")
    print(f"DUY_RETRIEVAL_RUBRIC={duy_total}/10")


if __name__ == "__main__":
    main()
