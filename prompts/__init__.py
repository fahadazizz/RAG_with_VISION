"""Prompts package for RAG pipeline."""

from prompts.rag_prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_PROMPT,
    get_rag_prompt,
)

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RAG_PROMPT",
    "get_rag_prompt",
]
