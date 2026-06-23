# mmlu-perturb-prompts

Measure how much model rankings change when benchmark prompts are systematically perturbed across distinct axes. The central output is rank correlation (Spearman) between original and perturbed conditions across multiple models.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Set API keys via environment variables (referenced in `config.yaml`):

```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key   # required for semantic_paraphrase only
```

The first run downloads the MMLU test set (`cais/mmlu`, config `all`) via HuggingFace `datasets` (~GB).

## Configuration

Edit [`config.yaml`](config.yaml) to set:

- `seed` and `sample_size` (default 300, stratified by subject)
- Model endpoints (`id`, `base_url`, `model`, `api_key_env`)
- Paraphrase model for `semantic_paraphrase` perturbation
- Output paths under `data/` and `results/`

## Usage

Full pipeline (sample → perturb → evaluate → metrics → figures):

```bash
python run.py
```

Common options:

```bash
# Small smoke test
python run.py --sample-size 10 --conditions original,context_inject --models model_a

# Composed perturbation stack
python run.py --conditions context_inject+instruction_style:verbose

# Resume after crash (evaluation checkpoints per model × condition)
python run.py --sample-size 300

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
| `data/cache/paraphrase/` | SHA-256 keyed paraphrase cache |
| `results/raw/*.jsonl` | Per-question eval records |
| `results/raw/checkpoints/` | Resume checkpoints |
| `results/metrics/summary.csv` | Flat metrics table |
| `results/metrics/metrics.json` | Full metrics dict |
| `results/metrics/run_meta.json` | Seed, models, conditions, timestamp |
| `results/figures/` | Heatmap, flip-rate bar chart, scatter plots |

## Metrics

- Per-condition accuracy per model per subject
- Spearman rank correlation of model rankings (by per-subject accuracy) between original and each perturbation
- Per-question flip rate (answer change vs. original)
- Accuracy delta per perturbation type (original − perturbed, per model)

## Project layout

```
config.yaml
run.py
src/
  loader.py        # MMLU loading and stratified sampling
  perturbations.py # Perturbation transforms and cache
  evaluator.py     # Model querying and response parsing
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
