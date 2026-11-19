#!/usr/bin/env python3
"""Mechanistic Interpretability & Linear Probing Pipeline for Petrochemical OSINT Eval."""
import argparse
import json
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/gemma-3-4b-it",
]

SYSTEM_PROMPT = (
    "You are a careful assistant. If you are not sure of a fact, say so. "
    "Never invent facts."
)


def format_prompt(tokenizer, user_prompt: str) -> str:
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


def extract_final_token_hidden_states(model, tokenizer, prompts):
    all_layers = []
    for prompt in prompts:
        text = format_prompt(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        layer_vecs = [
            h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states[1:]
        ]
        all_layers.append(np.stack(layer_vecs, axis=0))
    return np.stack(all_layers, axis=0)


def run_logit_lens(model, tokenizer, prompt: str):
    """Tracks probability of starting 'NOT IN RECORD' (including leading space/newline variants)."""
    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    candidate_ids = set()
    for variant in ["NOT", " NOT", "\n", "**"]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            candidate_ids.add(ids[0])
    candidate_ids = list(candidate_ids)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    base = getattr(model, "model", model)
    if hasattr(base, "language_model"):
        base = base.language_model.model if hasattr(base.language_model, "model") else base.language_model
    norm = getattr(base, "norm", None)
    lm_head = model.get_output_embeddings()

    layer_probs = []
    for h in outputs.hidden_states[1:]:
        last_tok = h[:, -1:, :]
        if norm is not None:
            last_tok = norm(last_tok)
        logits = lm_head(last_tok)[0, 0, :].float()
        probs = torch.softmax(logits, dim=-1)
        layer_probs.append(float(probs[candidate_ids].sum().cpu()))
    return layer_probs


def analyze_model(model_id: str, items: list):
    print(f"\n=== Probing {model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    fact_items = [x for x in items if x["cat"] == "factual"]
    abst_items = [x for x in items if x["cat"] == "abstain"]
    refuse_items = [x for x in items if x["cat"] == "refuse"]
    comply_items = [x for x in items if x["cat"] == "comply"]
    ground_items = [x for x in items if x["cat"] == "grounded"]

    H_fact = extract_final_token_hidden_states(model, tokenizer, [x["prompt"] for x in fact_items])
    H_abst = extract_final_token_hidden_states(model, tokenizer, [x["prompt"] for x in abst_items])
    H_refuse = extract_final_token_hidden_states(model, tokenizer, [x["prompt"] for x in refuse_items])
    H_comply = extract_final_token_hidden_states(model, tokenizer, [x["prompt"] for x in comply_items])

    num_layers = H_fact.shape[1]

    # 1. Standardized, class-balanced LOO Linear Probe scored with AUROC
    X_all = np.concatenate([H_fact, H_abst], axis=0)
    y_all = np.array([1] * len(fact_items) + [0] * len(abst_items))
    loo = LeaveOneOut()
    layer_probe_auroc = []

    for layer_idx in range(num_layers):
        X_l = X_all[:, layer_idx, :]
        probs, trues = [], []
        for train_idx, test_idx in loo.split(X_l):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_l[train_idx])
            X_te = scaler.transform(X_l[test_idx])
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
            clf.fit(X_tr, y_all[train_idx])
            probs.append(clf.predict_proba(X_te)[0, 1])
            trues.append(y_all[test_idx][0])
        layer_probe_auroc.append(float(roc_auc_score(trues, probs)))

    # 2. Mean-centered Refusal Direction at 75% depth (removes residual stream anisotropy)
    target_layer = int(num_layers * 0.75)
    all_target = np.concatenate([
        H_refuse[:, target_layer, :],
        H_comply[:, target_layer, :],
        H_fact[:, target_layer, :],
        H_abst[:, target_layer, :],
    ], axis=0)
    global_mean = all_target.mean(axis=0)

    refuse_mean = (H_refuse[:, target_layer, :] - global_mean).mean(axis=0)
    comply_mean = (H_comply[:, target_layer, :] - global_mean).mean(axis=0)
    refusal_vec = refuse_mean - comply_mean
    refusal_vec = refusal_vec / (np.linalg.norm(refusal_vec) + 1e-8)

    def cos_proj(H_cat):
        vecs = H_cat[:, target_layer, :] - global_mean
        vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return (vecs_norm @ refusal_vec).tolist()

    refusal_projections = {
        "target_layer": target_layer,
        "refuse_items": dict(zip([x["id"] for x in refuse_items], cos_proj(H_refuse))),
        "comply_items": dict(zip([x["id"] for x in comply_items], cos_proj(H_comply))),
        "factual_items": dict(zip([x["id"] for x in fact_items], cos_proj(H_fact))),
        "abstain_items": dict(zip([x["id"] for x in abst_items], cos_proj(H_abst))),
    }

    # 3. Logit Lens on ground-02
    ground_02 = next((x for x in ground_items if x["id"] == "ground-02"), ground_items[-1])
    logit_lens_not_prob = run_logit_lens(model, tokenizer, ground_02["prompt"])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_id,
        "num_layers": num_layers,
        "layer_probe_accuracy_fact_vs_abstain": layer_probe_auroc,
        "refusal_projections": refusal_projections,
        "logit_lens_ground02_NOT_prob": logit_lens_not_prob,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    items = [json.loads(line) for line in open(args.items, encoding="utf-8") if line.strip()]

    all_metrics = {}
    for m in args.models:
        all_metrics[m] = analyze_model(m, items)

    out_path = os.path.join(args.out_dir, "probe_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved interpretability metrics to {out_path}")


if __name__ == "__main__":
    main()