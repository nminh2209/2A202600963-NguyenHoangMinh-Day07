"""
Simple Gradio UI for testing the Day 7 RAG agent.

Run:
    pip install gradio openai
    python app.py

Set OPENAI_API_KEY in .env or paste a key in the UI to use GPT-4o-mini.
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from src.bootstrap import build_rag_system, format_search_results
from src.llm import LLM_PROVIDER_ENV, OPENAI_LLM_MODEL

load_dotenv(override=False)

_store = None
_agent = None
_backend_label = "mock"


def _ensure_agent(llm_provider: str, api_key: str):
    global _store, _agent, _backend_label

    provider = llm_provider.strip().lower()
    key = api_key.strip() or None
    _store, _agent = build_rag_system(
        llm_provider=provider,
        api_key=key,
        collection_name="ui_rag_store",
    )
    llm_name = getattr(_agent._llm_fn, "_backend_name", "mock demo llm")
    embed_name = getattr(_store._embedding_fn, "_backend_name", _store._embedding_fn.__class__.__name__)
    _backend_label = (
        f"LLM: {llm_name} | Embeddings: {embed_name} | Chunks indexed: {_store.get_collection_size()}"
    )


def ask(question: str, top_k: int, llm_provider: str, api_key: str) -> tuple[str, str, str]:
    if not question.strip():
        return "Please enter a question.", "", _backend_label

    try:
        _ensure_agent(llm_provider, api_key)
        results = _store.search(question.strip(), top_k=int(top_k))
        answer = _agent.answer(question.strip(), top_k=int(top_k))
        return answer, format_search_results(results), _backend_label
    except ValueError as error:
        return str(error), "", _backend_label
    except Exception as error:
        return f"Error: {error}", "", _backend_label


def build_ui() -> gr.Blocks:
    default_provider = os.getenv(LLM_PROVIDER_ENV, "mock").strip().lower()
    if default_provider not in {"mock", "openai"}:
        default_provider = "mock"

    with gr.Blocks(title="Day 7 RAG Demo") as demo:
        gr.Markdown(
            "# Knowledge Base Agent\n"
            "Ask questions against the sample documents in `data/`. "
            f"Use **OpenAI ({OPENAI_LLM_MODEL})** for real answers, or **Mock** for offline testing.\n\n"
            "When an OpenAI API key is set, the app automatically uses **OpenAI embeddings** "
            "and indexes **chunked** documents for better retrieval."
        )

        with gr.Row():
            llm_provider = gr.Dropdown(
                choices=["mock", "openai"],
                value=default_provider,
                label="LLM Provider",
            )
            api_key = gr.Textbox(
                label="OpenAI API Key (optional if set in .env)",
                placeholder="sk-...",
                type="password",
            )

        question = gr.Textbox(
            label="Question",
            placeholder="What is Python used for in production environments?",
            lines=2,
        )
        top_k = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Top-k chunks")
        submit = gr.Button("Ask", variant="primary")

        backend = gr.Textbox(label="Active backends", interactive=False)
        answer = gr.Textbox(label="Agent answer", lines=8)
        sources = gr.Markdown(label="Retrieved chunks")

        examples = gr.Examples(
            examples=[
                ["What is Python used for?", 3, default_provider, ""],
                ["How does a vector store workflow work?", 3, default_provider, ""],
                ["What is the proposed RAG architecture?", 3, default_provider, ""],
                ["What customer support topics should be indexed?", 3, default_provider, ""],
                ["What are common retrieval failure cases?", 3, default_provider, ""],
            ],
            inputs=[question, top_k, llm_provider, api_key],
        )

        submit.click(ask, inputs=[question, top_k, llm_provider, api_key], outputs=[answer, sources, backend])
        demo.load(
            lambda: _backend_label,
            outputs=backend,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)
