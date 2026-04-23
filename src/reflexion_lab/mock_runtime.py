from __future__ import annotations
import json
import os
import time
from dotenv import load_dotenv
from llama_cpp import Llama

from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_prompt,
    build_evaluator_prompt,
    build_reflector_prompt,
)
from .schemas import QAExample, JudgeResult, ReflectionEntry

load_dotenv()
_llm: Llama | None = None


def _get_llm() -> Llama:
    global _llm
    if _llm is None:
        model_path = os.getenv("LOCAL_MODEL_PATH", "").strip()
        if not model_path:
            raise ValueError("LOCAL_MODEL_PATH chưa được set trong .env")
        n_ctx = int(os.getenv("LOCAL_MODEL_N_CTX", "4096"))
        _llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            chat_format="chatml",
            verbose=False,
        )
    return _llm


def _chat(system: str, user: str) -> tuple[str, int, int]:
    llm = _get_llm()
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    content = resp["choices"][0]["message"]["content"] or ""
    usage = resp.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, int, float]:
    context_texts = [f"{c.title}: {c.text}" for c in example.context]
    user_prompt = build_actor_prompt(example.question, context_texts, reflection_memory)

    t0 = time.perf_counter()
    content, p_tok, c_tok = _chat(ACTOR_SYSTEM, user_prompt)
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = content.strip()
    for line in reversed(content.splitlines()):
        if line.lower().startswith("final answer:"):
            answer = line.split(":", 1)[1].strip()
            break

    return answer, p_tok + c_tok, latency_ms


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int]:
    user_prompt = build_evaluator_prompt(example.question, example.gold_answer, answer)
    content, p_tok, c_tok = _chat(EVALUATOR_SYSTEM, user_prompt)

    try:
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
        result = JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", ""),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except Exception:
        result = JudgeResult(score=0, reason=f"Parse error from evaluator. Raw: {content[:200]}")

    return result, p_tok + c_tok


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult, answer: str) -> tuple[ReflectionEntry, int]:
    user_prompt = build_reflector_prompt(example.question, example.gold_answer, answer, judge.reason, attempt_id)
    content, p_tok, c_tok = _chat(REFLECTOR_SYSTEM, user_prompt)

    try:
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", ""),
            next_strategy=data.get("next_strategy", ""),
        )
    except Exception:
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Failed to parse reflection.",
            next_strategy="Re-check second-hop reasoning and entities.",
        )

    return entry, p_tok + c_tok