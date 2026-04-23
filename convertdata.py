import json
from pathlib import Path

src = Path("data/hotpotqa_100.json")
dst = Path("data/hotpotqa_100_lab.json")

raw = json.loads(src.read_text(encoding="utf-8"))
out = []
for item in raw:
    ctx = []
    for title, sents in item.get("context", []):
        ctx.append({
            "title": title,
            "text": " ".join(s.strip() for s in sents if isinstance(s, str)).strip()
        })

    out.append({
        "qid": item.get("_id", ""),
        "difficulty": item.get("level", "medium"),
        "question": item.get("question", ""),
        "gold_answer": item.get("answer", ""),
        "context": ctx
    })

dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved: {dst}")