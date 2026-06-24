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
GROQ_API_KEY=your_key        # llama-3-8b
```

Keys are loaded automatically via `python-dotenv` at startup. Shell environment variables override `.env` if both are set. Variable names must match `api_key_env` in `config.yaml`.

All required keys are validated **before** sampling, paraphrase warming, or evaluation — a missing key fails immediately with a clear error.

The first run downloads the MMLU test set (`cais/mmlu`, config `all`) via HuggingFace `datasets` (~GB).

## Configuration

Edit [`config.yaml`](config.yaml) to set:

- `seed` and `sample_size` (default 300, stratified by subject)
- `eval_concurrency` — parallel eval API calls (default 8; set to 1 for sequential)
- Model endpoints — three models from distinct training pipelines (see below)
- Paraphrase model for `semantic_paraphrase` perturbation
- Output paths under `data/` and `results/`

### Eval models (3 required for Spearman)

| ID | Provider | Model | Why |
|----|----------|-------|-----|
| `gpt-4o-mini` | OpenAI | `gpt-4o-mini` | Cheap, widely cited baseline |
| `claude-haiku-4-5` | Anthropic | `claude-haiku-4-5-20251001` | Different provider / training |
| `llama-3-8b` | Groq (OpenAI-compatible) | `llama-3.1-8b-instant` | Open-weight Llama 3.1 8B |

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
| `results/figures/` | Heatmap, flip-rate bar chart, scatter plots |

## Metrics

All metrics use a **matched intersection set** per condition: only questions where every configured model produced a parseable answer under both `original` and that condition. Accuracy, flip rate, and accuracy delta are computed from the same filtered rows, so counts cannot disagree.

- Per-condition accuracy per model per subject (on the matched set; `matched_n` in outputs)
- Spearman rank correlation at the **condition level** (one correlation per perturbation, using per-model macro accuracies on the matched set). Per-subject Spearman is reported only when a subject has at least `min_questions_per_subject` matched questions (default 5).
- Per-question flip rate (answer change vs. original, matched set only)
- Accuracy delta per model per condition (`original_macro_accuracy − perturbed_macro_accuracy` on the same matched questions)

Eval cells are only checkpointed when the API returns a **parseable** answer; verbose/unparseable responses are retried and excluded from the checkpoint until successful.

## Project layout

```
config.yaml
run.py
src/
  loader.py          # MMLU loading and stratified sampling
  paraphrase_cache.py # Upfront paraphrase generation and disk cache
  perturbations.py   # Independent perturbation pipelines
  evaluator.py       # Unified eval loop and question-level checkpoint
  metrics.py       # Metric computation
  viz.py           # Matplotlib figures
  conditions.py    # Condition parsing (including composed stacks)
  types.py
  io_utils.py
data/
results/
```

## License

MIT — see [LICENSE](LICENSE).
