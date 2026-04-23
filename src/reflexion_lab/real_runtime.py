from __future__ import annotations
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_prompt,
    build_evaluator_prompt,
    build_reflector_prompt,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-your"):
            raise ValueError("OPENAI_API_KEY is not set. Please edit your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


def _chat(system: str, user: str, model: str | None = None) -> tuple[str, int, int]:
    """Returns (content, prompt_tokens, completion_tokens)."""
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return content, prompt_tokens, completion_tokens


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, float]:
    """Returns (answer, total_tokens, latency_ms)."""
    context_texts = [f"{c.title}: {c.text}" for c in example.context]
    user_prompt = build_actor_prompt(example.question, context_texts, reflection_memory)

    t0 = time.perf_counter()
    content, prompt_tok, completion_tok = _chat(ACTOR_SYSTEM, user_prompt)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract final answer from "Final Answer: ..." line
    answer = content.strip()
    for line in reversed(content.splitlines()):
        if line.lower().startswith("final answer:"):
            answer = line.split(":", 1)[1].strip()
            break

    return answer, prompt_tok + completion_tok, latency_ms


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int]:
    """Returns (JudgeResult, total_tokens)."""
    user_prompt = build_evaluator_prompt(example.question, example.gold_answer, answer)

    content, prompt_tok, completion_tok = _chat(EVALUATOR_SYSTEM, user_prompt)

    try:
        # Strip markdown code fences if present
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
        result = JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", ""),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: score 0
        result = JudgeResult(score=0, reason=f"Parse error from evaluator. Raw: {content[:200]}")

    return result, prompt_tok + completion_tok


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    answer: str,
) -> tuple[ReflectionEntry, int]:
    """Returns (ReflectionEntry, total_tokens)."""
    user_prompt = build_reflector_prompt(
        example.question, example.gold_answer, answer, judge.reason, attempt_id
    )

    content, prompt_tok, completion_tok = _chat(REFLECTOR_SYSTEM, user_prompt)

    try:
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", ""),
            next_strategy=data.get("next_strategy", ""),
        )
    except (json.JSONDecodeError, KeyError):
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Failed to parse reflection.",
            next_strategy="Try again more carefully.",
        )

    return entry, prompt_tok + completion_tok
