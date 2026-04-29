"""LLM-as-judge runner. Scores one or two completion files against a rubric or
pairwise comparison using a judge model.

Judge backends:
  - ``local:<hf_id>``  (default: ``local:Qwen/Qwen2.5-7B-Instruct``): load the
    judge via HF transformers on a single GPU. Fits in bf16 on H100/H200.
  - ``openai:<model>`` (e.g. ``openai:gpt-4o-mini``). Requires ``OPENAI_API_KEY``.
  - ``anthropic:<model>`` (e.g. ``anthropic:claude-opus-4-6``). Requires
    ``ANTHROPIC_API_KEY``.

Rubric mode (default): scores one completion file per prompt on helpfulness /
correctness / conciseness (1-5 each). Used to compare variants quantitatively.

Pairwise mode (--pairwise): scores two completion files against each other,
doing both A/B orderings to mitigate position bias.

Usage:
    # Rubric scoring:
    shared/launch.sh python 07_eval/llm_judge.py \
        --prompts 07_eval/prompts/squad_100.jsonl \
        --responses $RESULTS_ROOT/outputs/qwen3_squad.jsonl \
        --judge local:Qwen/Qwen2.5-7B-Instruct \
        --out $RESULTS_ROOT/judge/rubric.jsonl

    # Pairwise:
    shared/launch.sh python 07_eval/llm_judge.py \
        --prompts 07_eval/prompts/squad_100.jsonl \
        --responses-a $RESULTS_ROOT/outputs/base.jsonl \
        --responses-b $RESULTS_ROOT/outputs/lora.jsonl \
        --pairwise --judge local:Qwen/Qwen2.5-7B-Instruct \
        --out $RESULTS_ROOT/judge/pairwise.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of the first top-level JSON object.

    Bracket-balanced so nested ``{...}`` in the judge output parses correctly;
    a non-greedy regex matched the first inner `}` and broke on nested rubrics.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None


# ------------------ judge backends -----------------------------------------
class _LocalJudge:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
        self.model.eval()

    def __call__(self, system: str, user: str) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=256, temperature=0.2, top_p=0.95,
                do_sample=True, pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


class _OpenAIJudge:
    def __init__(self, model: str) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def __call__(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2, max_tokens=256,
        )
        return r.choices[0].message.content or ""


class _AnthropicJudge:
    def __init__(self, model: str) -> None:
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, system: str, user: str) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=256, temperature=0.2,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text if r.content else ""


def _build_judge(spec: str):
    backend, _, model = spec.partition(":")
    if backend == "local":
        return _LocalJudge(model or "Qwen/Qwen2.5-7B-Instruct")
    if backend == "openai":
        return _OpenAIJudge(model or "gpt-4o-mini")
    if backend == "anthropic":
        return _AnthropicJudge(model or "claude-opus-4-7")
    raise SystemExit(f"unknown judge backend: {spec!r} (use local:<id>, openai:<model>, anthropic:<model>)")


_VALID_WINNERS = {"A", "B", "tie"}
_FLIP = {"A": "B", "B": "A", "tie": "tie"}


def _normalize_winner(raw) -> str | None:
    """Normalize a judge's `winner` field; returns None for unrecognized values."""
    if not isinstance(raw, str):
        return None
    val = raw.strip()
    # Accept 'a', 'A', 'tie', 'TIE', ' a ', etc. but reject 'neither', 'both', ''.
    if val.lower() == "tie":
        return "tie"
    upper = val.upper()
    if upper in {"A", "B"}:
        return upper
    return None


# ------------------ scoring modes ------------------------------------------
def rubric_mode(args, judge, tmpl) -> list[dict]:
    prompts = {p["id"]: p for p in _load_jsonl(args.prompts)}
    responses = _load_jsonl(args.responses)
    sys_t = tmpl["rubric"]["system"].strip()
    results = []
    for r in responses:
        p = prompts.get(r["id"])
        if p is None:
            continue
        user_t = tmpl["rubric"]["user"].format(
            prompt=p["prompt"], reference=p.get("reference", ""), response=r["response"],
        )
        raw = judge(sys_t, user_t)
        parsed = _extract_json(raw) or {}
        results.append({
            "id": r["id"],
            "helpfulness": parsed.get("helpfulness"),
            "correctness": parsed.get("correctness"),
            "conciseness": parsed.get("conciseness"),
            "rationale": parsed.get("rationale", ""),
            "raw": raw,
        })
    return results


def pairwise_mode(args, judge, tmpl) -> list[dict]:
    prompts = {p["id"]: p for p in _load_jsonl(args.prompts)}
    a_map = {r["id"]: r["response"] for r in _load_jsonl(args.responses_a)}
    b_map = {r["id"]: r["response"] for r in _load_jsonl(args.responses_b)}
    sys_t = tmpl["pairwise"]["system"].strip()
    results = []
    for pid, p in prompts.items():
        ra = a_map.get(pid)
        rb = b_map.get(pid)
        if ra is None or rb is None:
            continue
        # Both orderings → average out position bias.
        ab = judge(sys_t, tmpl["pairwise"]["user"].format(
            prompt=p["prompt"], response_a=ra, response_b=rb,
        ))
        ba = judge(sys_t, tmpl["pairwise"]["user"].format(
            prompt=p["prompt"], response_a=rb, response_b=ra,
        ))
        pab = _extract_json(ab) or {}
        pba = _extract_json(ba) or {}
        ab_winner = _normalize_winner(pab.get("winner"))
        ba_winner_raw = _normalize_winner(pba.get("winner"))
        # Translate the B-A ordering back to the A-B frame; unrecognized → None.
        ba_winner = _FLIP[ba_winner_raw] if ba_winner_raw else None
        results.append({
            "id": pid,
            "ab_winner": ab_winner,
            "ba_winner": ba_winner,
            "ab_rationale": pab.get("rationale", ""),
            "ba_rationale": pba.get("rationale", ""),
        })
    return results


def _summarize(results: list[dict], pairwise: bool) -> None:
    if pairwise:
        # Count a decisive A/B win only when both orderings agree. Disagreement
        # and any unparsed judge output collapse into "tie" for reporting.
        tallies = {"A": 0, "B": 0, "tie": 0, "unparsed": 0}
        for r in results:
            ab, ba = r.get("ab_winner"), r.get("ba_winner")
            if ab is None or ba is None:
                tallies["unparsed"] += 1
            elif ab == ba and ab in _VALID_WINNERS:
                tallies[ab] += 1
            else:
                tallies["tie"] += 1
        print(f"[judge] pairwise tallies (consensus of both orderings): {tallies}")
    else:
        def _avg(k: str):
            vals = [r[k] for r in results if isinstance(r.get(k), int)]
            return sum(vals) / len(vals) if vals else None
        print(f"[judge] avg helpfulness={_avg('helpfulness')}  "
              f"correctness={_avg('correctness')}  conciseness={_avg('conciseness')}  "
              f"(n={len(results)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--responses", type=Path)
    ap.add_argument("--responses-a", type=Path)
    ap.add_argument("--responses-b", type=Path)
    ap.add_argument("--pairwise", action="store_true")
    ap.add_argument("--judge", default="local:Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", type=Path,
                    default=Path(__file__).parent / "judge_prompts.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tmpl = yaml.safe_load(args.template.read_text())
    judge = _build_judge(args.judge)
    print(f"[judge] backend: {args.judge}")

    if args.pairwise:
        if not (args.responses_a and args.responses_b):
            ap.error("--pairwise requires --responses-a and --responses-b")
        results = pairwise_mode(args, judge, tmpl)
    else:
        if not args.responses:
            ap.error("rubric mode needs --responses")
        results = rubric_mode(args, judge, tmpl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[judge] wrote {len(results)} rows → {args.out}")
    _summarize(results, args.pairwise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
