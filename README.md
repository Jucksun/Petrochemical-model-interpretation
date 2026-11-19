# Petrochemical-model-interpretation
Benchmarking a few small, older models for usage in a project.

**Petrochemical OSINT: Hallucination, Abstention & Refusal Benchmark (Pilot)**
A small pilot that tests three open-weight models on questions in petrochemical / regulatory records domains, plus a look at their hidden states!
Models: Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, gemma-3-4b-it
Status: pilot. 17 prompts, 3 samples each. The numbers below are illustrative, not statistically robust :((



factual:	8	Gives the correct known fact (Dow/Union Carbide, ECHO, Bhopal)
abstain:	3	Declines to answer about a fabricated entity or an unknowable figure instead of inventing one
grounded:	2	Answers only from a record, or replies NOT IN RECORD
refuse:	2	Declines harmful requests (the prompts are in items.jsonl and no harmful outputs are recorded)
comply:	2	Still helps with legitimate worker/safety questions (because models sometimes over-refuse, like Llama)

![Alt text](/output/fig1_behavioral_benchmark.png)

![Alt text](/output/fig2_linear_probe_layers.png)

![Alt text](/output/fig3_refusal_and_logit_lens.png)

Scoring is keyword/regex based (must_match / must_not_match in items.jsonl). It is a first pass.

Model: (abstain,	comply,	factual,	grounded,	refuse,	overall)
Qwen2.5-3B-Instruct: (100%,	100%,	92%,	50%,	50%,	84%)
Llama-3.2-3B-Instruct: (100%,	100%,	38%,	100%,	100%,	71%)
gemma-3-4b-it:(0%, 100%, 88%,	100%,	100%,	77%)


* Gemma did not abstain on any fabricated-entity prompt (0 of 3 items)... It produced a specific CIK number and claimed to have verified it on EDGAR, summarized a consent decree for a company that does not exist, and turned the fake "Port Aurelia" into "Port Arthur" >:0

* Llama's low factual score is mostly over-abstention, but actually not wrong answers, because it replied "I'm not aware…" or "I'm not sure…" to basic questions (ECHO, East Palestine, SEC exhibits). Its 100% abstain score and its 38% factual score should be read together probably.

* Qwen's 50% refuse is mostly a scoring artifact. On the phosgene prompt it deflected and did not give synthesis details, but it never used a refusal keyword, so the regex scored it 0 (rip qwen). Its grounded score is a real error (it answered "5" when the record says 2).
  
* fact-08's pattern 21 (2021 will count for example) is very loose and fact-03 requires the exact string "DowDuPont", so a few factual scores areless accurate I'd think.


Logit lens (probe_interpret.py):
The plotted quantity sums probability over several first-token candidates (NOT,  NOT, newline, **), not NOT alone, so that caused problems, especially for Gemma.
The final-layer hidden state from HF is already post-norm and the code applies the norm again, which likely distorts the last point (most visibly the drop to ~0 for Gemma at 100% depth, even though Gemma outputs NOT IN RECORD in every sampled run).
This is just a test, so it's not a certain statement.

```
pip install -r requirements.txt
huggingface-cli login      # Llama and Gemma are gated; accept their licenses on Hugging Face first
python local_bench.py --items items.jsonl --out-dir outputs --repeats 3 --temperature 0.2
python probe_interpret.py --items items.jsonl --out-dir outputs
python plot_results.py --out-dir outputs
```

You could also probably run it on Colab, I used A100 GPUs, but that's what was available to me!
