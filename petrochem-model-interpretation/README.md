# Petrochemical OSINT AI Safety Benchmark & Interpretability Suite

A self-contained evaluation and mechanistic interpretability toolkit for testing small open-weight LLMs (`Llama-3.2-3B-Instruct`, `Qwen2.5-3B-Instruct`, `Gemma-3-4B-it`) on petrochemical OSINT accuracy, hallucination abstention, context grounding, and chemical hazard refusal.

## Folder Layout
```text
osint-petrochemical-tracker/
├── items.jsonl               # 17 test prompts (factual, abstain, grounded, refuse, comply)
├── local_bench.py            # Local PyTorch/Transformers behavioral benchmark
├── probe_interpret.py        # Hidden-state extraction, linear probe, refusal vector & logit lens
├── plot_results.py           # Generates 3 publication-ready PNG figures
└── colab_pipeline.ipynb      # End-to-end Google Colab runner
```

## Quickstart Tutorial (Google Drive + Colab -> GitHub)

1. **Upload to Google Drive:**
   Upload all files from this bundle (`items.jsonl`, `local_bench.py`, `probe_interpret.py`, `plot_results.py`, `colab_pipeline.ipynb`) into your `My Drive/osint-petrochemical-tracker/` folder.
2. **Open in Google Colab with GPU:**
   Double-click `colab_pipeline.ipynb` in Google Drive -> **Open with Google Colaboratory**. Switch runtime to **T4 GPU** (*Runtime -> Change runtime type -> T4 GPU*).
3. **Accept Model Licenses & Log In:**
   On Hugging Face, accept the license agreements for `meta-llama/Llama-3.2-3B-Instruct` and `google/gemma-3-4b-it` (`Qwen/Qwen2.5-3B-Instruct` is ungated). Run Cell 2 in Colab and paste your Hugging Face Read token.
4. **Run Cells 3–5:**
   - `local_bench.py` generates `outputs/results.jsonl` and `outputs/summary.csv`.
   - `probe_interpret.py` extracts residual-stream hidden states across every layer and writes `outputs/probe_metrics.json`.
   - `plot_results.py` renders three PNG charts in `outputs/`.
5. **Push to GitHub:**
   In Colab (or locally once Drive syncs), initialize git in `osint-petrochemical-tracker/` and push `items.jsonl`, the `.py` scripts, `outputs/summary.csv`, `outputs/probe_metrics.json`, and the `outputs/*.png` figures to your repository.
