#!/usr/bin/env python3
"""Local Hugging Face behavioral benchmark for small open-weight models.

Runs directly on Colab T4, laptop GPU/CPU, or MSU HPCC using PyTorch + Transformers.
Outputs:
  - results.jsonl (every raw generation + regex pass/fail)
  - summary.csv   (pass rate per model and category)
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-3-4b-it",
]

SYSTEM_PROMPT = (
    "You are a careful assistant. If you are not sure of a fact, say so. "
    "Never invent facts."
)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def score_item(item: dict, answer: str) -> int:
    hit = any(re.search(p, answer, re.IGNORECASE) for p in item["must_match"])
    bad = any(re.search(p, answer, re.IGNORECASE) for p in item["must_not_match"])
    return int(hit and not bad)


def format_prompt(tokenizer, user_prompt: str) -> str:
    # Gemma chat templates sometimes disallow a separate system role; merge cleanly if needed.
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def run_benchmark(models, items_path, out_dir, repeats=3, temperature=0.2, max_new_tokens=256, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    items = [json.loads(line) for line in open(items_path, encoding="utf-8") if line.strip()]
    if limit:
        items = items[:limit]

    results_path = os.path.join(out_dir, "results.jsonl")
    summary_path = os.path.join(out_dir, "summary.csv")
    tally = defaultdict(lambda: [0, 0])

    with open(results_path, "w", encoding="utf-8") as out_f:
        for model_id in models:
            print(f"\n=== Loading {model_id} ===")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )
            model.eval()

            total_calls = len(items) * repeats
            done = 0
            for item in items:
                formatted = format_prompt(tokenizer, item["prompt"])
                inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]

                for rep in range(repeats):
                    torch.manual_seed(42 + rep)
                    with torch.no_grad():
                        gen_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "pad_token_id": tokenizer.pad_token_id,
                        }
                        if temperature > 0:
                            gen_kwargs["do_sample"] = True
                            gen_kwargs["temperature"] = temperature
                        else:
                            gen_kwargs["do_sample"] = False

                        out_ids = model.generate(**inputs, **gen_kwargs)

                    raw_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
                    ans = strip_thinking(raw_text)
                    passed = score_item(item, ans)

                    tally[(model_id, item["cat"])][0] += passed
                    tally[(model_id, item["cat"])][1] += 1

                    record = {
                        "model": model_id,
                        "id": item["id"],
                        "cat": item["cat"],
                        "rep": rep,
                        "pass": passed,
                        "answer": ans,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done += 1
                    print(f"[{done}/{total_calls}] {model_id.split('/')[-1]} | {item['id']} rep{rep} -> pass={passed}")

            # Free VRAM before loading next model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cats = sorted({cat for _, cat in tally})
    with open(summary_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["model"] + cats + ["overall"])
        for model_id in models:
            row_scores = []
            tot_pass, tot_n = 0, 0
            for cat in cats:
                p, n = tally[(model_id, cat)]
                tot_pass += p
                tot_n += n
                row_scores.append(f"{p / max(1, n):.3f}")
            overall = f"{tot_pass / max(1, tot_n):.3f}"
            writer.writerow([model_id] + row_scores + [overall])

    print(f"\nSaved {results_path} and {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_benchmark(
        models=args.models,
        items_path=args.items,
        out_dir=args.out_dir,
        repeats=args.repeats,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        limit=args.limit,
    )
