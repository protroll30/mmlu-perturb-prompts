# mmlu-perturb-prompts

Measure how much model rankings change when benchmark prompts are systematically perturbed across distinct axes. The central output is rank correlation (Spearman) between original and perturbed conditions across multiple models.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys (`.env` is gitignored):

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_key      # gpt-4o-mini
ANTHROPIC_API_KEY=your_key   # claude-haiku (eval + paraphrase)
GEMINI_API_KEY=your_key      # gemini-flash
```

Keys are loaded automatically via `python-dotenv` at startup. Shell environment variables override `.env` if both are set. Variable names must match `api_key_env` in `config.yaml`.

All required keys are validated **before** sampling, paraphrase warming, or evaluation — a missing key fails immediately with a clear error.

The first run downloads the MMLU test set (`cais/mmlu`, config `all`) via HuggingFace `datasets` (~GB).

## Configuration

Edit [`config.yaml`](config.yaml) to set:

- `seed` and `sample_size` (default 300, stratified by subject)
- `eval_concurrency` — parallel eval API calls (default 8; set to 1 for sequential)
- `min_questions_per_subject` — minimum matched questions before computing per-subject Spearman (default 5; raise this if per-subject correlations are noisy)
- Model endpoints — three models from distinct training pipelines (see below)
- Paraphrase model for `semantic_paraphrase` perturbation
- Output paths under `data/` and `results/`

### Eval models (3 required for Spearman)

| ID | Provider | Model | Why |
|----|----------|-------|-----|
| `gpt-4o-mini` | OpenAI | `gpt-4o-mini` | Cheap, widely cited baseline |
| `claude-haiku-4-5` | Anthropic | `claude-haiku-4-5-20251001` | Different provider / training |
| `gemini-flash` | Google (OpenAI-compatible) | `gemini-2.5-flash-lite` | Third provider; use `--concurrency 1` on free tier |

Spearman rank correlation needs **≥3 models**; with N=2 the statistic is always ±1. Mixing providers avoids measuring only capability-tier differences within one family.

Paraphrase uses **Haiku** (`claude-haiku-4-5-20251001`) — sufficient for stem paraphrase at lower cost than Sonnet.

Each model entry supports `provider: openai` (OpenAI-compatible `base_url`) or `provider: anthropic` (native Messages API).

## Usage

Full pipeline:

```
load → warm paraphrase cache → perturb (fan-out) → evaluate (unified loop) → metrics → viz
```

```bash
python run.py
```

Common options:

```bash
# Small smoke test
python run.py --sample-size 10 --conditions original,context_inject --models gpt-4o-mini

# Composed perturbation stack
python run.py --conditions context_inject+instruction_style:verbose

# Resume after crash (question-level checkpoint; no re-querying completed work)
python run.py --sample-size 300

# Faster eval: parallel API calls (same results, ~Nx speedup up to rate limits)
python run.py --sample-size 300 --concurrency 8

# Re-sample MMLU questions
python run.py --force-resample

# Skip stages
python run.py --skip-eval
python run.py --skip-metrics
python run.py --skip-viz
```

`run.py` exits with code 1 if any evaluation cells are still missing parseable answers after retries. Re-running the same command resumes from the checkpoint.

### Condition syntax

- `original` — unperturbed baseline
- Single steps: `label_remap:numeric`, `instruction_style:verbose`, `option_shuffle`, `context_inject`, `semantic_paraphrase`
- Composed stacks: `context_inject+instruction_style:minimal`, `option_shuffle+label_remap:parenthetical`
- `all` — original plus every registered single-step variant

## Perturbation types

| Type | Description |
|------|-------------|
| `label_remap` | A/B/C/D → 1/2/3/4 or (a)/(b)/(c)/(d) |
| `option_shuffle` | Randomly reorder options (seeded per question) |
| `context_inject` | Prepend neutral framing sentence |
| `instruction_style` | `none`, `minimal`, or `verbose` task instruction |
| `semantic_paraphrase` | Claude paraphrase of question stem (cached on disk) |

## Outputs

| Path | Contents |
|------|----------|
| `data/sampled.jsonl` | Stratified MMLU sample |
| `data/perturbed/*.jsonl` | One file per perturbation condition |
| `data/cache/paraphrase/` | SHA-256 keyed paraphrase cache (warmed before perturb/eval) |
| `results/raw/results.jsonl` | All eval records (append-only) |
| `results/raw/checkpoints/eval_checkpoint.json` | Question-level resume checkpoint |
| `results/metrics/summary.csv` | Flat metrics table |
| `results/metrics/metrics.json` | Full metrics dict |
| `results/metrics/run_meta.json` | Seed, models, conditions, timestamp |
| `results/figures/` | Heatmap, flip-rate charts, per-condition scatter plots |

## Metrics

All metrics use a **matched intersection set** per condition: only questions where every configured model produced a parseable answer under both `original` and that condition. Accuracy, flip rate, accuracy delta, and Spearman are all computed from the same filtered rows — so counts cannot disagree across metrics.

- **Per-condition accuracy** per model per subject (on the matched set; `matched_n` in outputs)
- **Condition-level Spearman** — one correlation per perturbation condition, using per-model macro accuracies on the matched set. The primary rank-stability signal.
- **Perturbation-type Spearman** — Spearman pooled across all `(model, subject)` pairs within a perturbation type, giving ~N_models × N_subjects data points instead of just N_models. More robust with few models. Reported under `spearman_by_perturbation_type` in `metrics.json`.
- **Per-subject Spearman** — reported only when a subject has at least `min_questions_per_subject` matched questions. Frequently null for low-count subjects; use perturbation-type Spearman as the primary estimate.
- **Flip rate** — fraction of questions where the model's answer changed relative to `original`, on the matched set. Reported pooled, per condition, per subject, per perturbation type, and per model × perturbation type.
- **Accuracy delta** — `original_macro_accuracy − perturbed_macro_accuracy` on the matched set.

Eval cells are only checkpointed when the API returns a **parseable** answer; verbose/unparseable responses are retried and excluded from the checkpoint until successful.

## Project layout

```
config.yaml
run.py
src/
  loader.py           # MMLU loading and stratified sampling
  paraphrase_cache.py # Upfront paraphrase generation and disk cache
  perturbations.py    # Independent perturbation pipelines
  evaluator.py        # Unified eval loop and question-level checkpoint
  metrics.py          # Metric computation (matched-set intersection)
  viz.py              # Matplotlib figures
  conditions.py       # Condition parsing (including composed stacks)
  types.py
  io_utils.py
data/
results/
```

---

## Results history

This section records each distinct run and any methodology changes. New entries are appended.

---

### Run 1 — Initial results (June 2026, seed 42, 300 questions, 3 models)

**Models**: gpt-4o-mini · claude-haiku-4-5 · gemini-flash  
**Conditions**: original + all single-step variants (label_remap:numeric, label_remap:parenthetical, option_shuffle, context_inject, instruction_style:{none,minimal,verbose}, semantic_paraphrase)

#### Flip-rate story (matched set, pooled across models)

This is the primary perturbation-effect signal. A flip is any question where the model's answer changed relative to `original` on the same matched question set.

| Perturbation type | Pooled flip rate |
|-------------------|-----------------|
| option_shuffle | ~52% |
| semantic_paraphrase | ~15% |
| label_remap | ~14% |
| context_inject | ~12% |
| instruction_style | ~7% |

`option_shuffle` is by far the most disruptive — flipping the answer on more than half of questions across all models. Instruction-style wording changes are comparatively benign.

#### Model-dependent damage (label_remap:numeric)

The label remap to numeric labels (1/2/3/4) affects models very differently:

- **gemini-flash / llama-class models**: ~31% flip rate
- **claude-haiku**: ~10% flip rate

This asymmetric sensitivity directly scrambles relative model rankings under numeric labels. It is the root of the low Spearman observed for `label_remap`.

#### Known issue in Run 1 — unmatched subset artifact

The initial analysis compared accuracy deltas across conditions that had **different question counts** due to dropped API-call failures (e.g., gpt-4o-mini: 212 rows for `original` vs 103 rows for `label_remap:numeric`). This made the numeric-remap condition appear to raise accuracy for some models — an artifact of the unmatched subset, not a real effect.

On the **matched set** (intersection of questions with parseable answers for all models under both `original` and the given condition), numeric remap is genuinely disruptive (high flip rate, model-dependent), consistent with low Spearman. The ranking instability finding survives; the previously reported "accuracy jump" does not.

#### Spearman notes

With 3 models, condition-level Spearman has only 3 data points and is highly sensitive to noise. Many per-subject Spearman values were null because subjects averaged ~5 questions, and matched-set filtering dropped some below the `min_questions_per_subject=5` floor.

---

### Methodology revision (June 2026)

The following changes were made to address Run 1 issues. Future runs will use this revised pipeline.

**1. Completeness enforcement**

`run_evaluation` now returns a `(path, is_complete)` tuple. `run.py` exits with code 1 and logs a per-condition breakdown of missing cells when the design is incomplete. This makes silent data imbalances observable. Re-running resumes from the checkpoint.

**2. Matched-set guarantee (already in place, now explicitly documented)**

All metrics — accuracy, flip rate, accuracy delta, Spearman — are computed from the same matched intersection set per condition. The matched set is questions where every model produced a parseable answer under both `original` and the given condition. This was already implemented; the Run 1 issue was a pre-existing imbalance from a prior version of the code.

**3. Perturbation-type Spearman**

Added `_spearman_by_perturbation_type` in `metrics.py`. Instead of computing Spearman over 3 per-model macro accuracies (3 data points), this pools all `(model, subject)` pairs within a perturbation type, giving ~N_models × N_eligible_subjects data points. Subjects below `min_questions_per_subject` are still excluded. Output key: `spearman_by_perturbation_type` in `metrics.json`.

**4. Per-model flip-rate visualization**

Added two new figures in `viz.py`:
- `flip_rate_by_perturbation_type.png` — bar chart of pooled flip rate per perturbation type, sorted by magnitude. Tells the clean ordering story.
- `flip_rate_by_model_perturbation_type.png` — grouped bar chart showing per-model breakdown, exposing the asymmetric label_remap damage across models.

**What changed in the story**: The `accuracy_delta` comparison for `label_remap:numeric` was an artifact of the unmatched subset. The correct characterization is high flip rate with model-dependent severity, not accuracy improvement. The instability finding (low Spearman for option_shuffle and label_remap) is unchanged.

---

## License

MIT — see [LICENSE](LICENSE).
