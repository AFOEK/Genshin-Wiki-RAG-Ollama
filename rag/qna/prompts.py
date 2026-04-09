from __future__ import annotations
from .generators import generate
import textwrap

def build_context(chunks: list[dict]) -> str:
    parts = []
    for row in chunks:
        text = row["text"] or ""
        parts.append(
            f"[chunk_id={row['chunk_id']}] "
            f"title={row['title']} | source={row['source']} | url={row['url']}\n"
            f"{text}\n"
        )
    return "\n---\n".join(parts)


def summarize_chunk_group(cfg: dict, question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    prompt = textwrap.dedent(f"""
        You are helping build a faithful lore answer from retrieved Genshin Impact knowledge chunks.

        Question:
        {question}

        Context chunks:
        {context}

        Task:
        - Extract only information relevant to the question.
        - Preserve chronology when possible.
        - Keep proper nouns, factions, gods, nations, events, and relationships accurate.
        - Include chunk_id citations inline like [chunk_id=123].
        - If context is fragmentary, say so briefly.
        - Return concise bullet notes, not a full polished answer.
    """).strip()
    return generate(cfg, prompt)


def synthesize_final_answer(cfg: dict, question: str, notes: list[str], timeout: str = "15m",) -> str:
    if not notes:
        return "I couldn't summarize any retrieved context for this question."
    
    notes_block = "\n\n".join(
        f"### Notes from batch {i+1}\n{n}" for i, n in enumerate(notes)
    )
    prompt = textwrap.dedent(f"""
        You are answering a question using retrieved and pre-summarized Genshin Impact knowledge.

        Question:
        {question}

        Retrieved notes:
        {notes_block}

        Task:
        - Write a coherent, faithful answer using only the retrieved notes.
        - For broad lore questions, organize the answer chronologically.
        - Merge overlapping points and avoid repetition.
        - Keep important names and events accurate.
        - Include inline citations using the provided chunk_id citations.
        - If the retrieved notes are incomplete, say what is missing.
        - Do not invent facts outside the notes.
    """).strip()
    return generate(cfg, prompt)