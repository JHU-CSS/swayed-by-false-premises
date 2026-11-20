# SWAYed by False Premises

Code and data for **SWAYed by False Premises: Measuring and Mitigating Framing Sycophancy in Language Models** (ACL 2026).

This repository reproduces every table and figure in the paper from the model answers we collected. You do not need API keys, credentials or network access.

## Reproduce the paper

Requires Python 3.10 or later.

```bash
git clone https://github.com/JHU-CSS/swayed-by-false-premises.git
cd swayed-by-false-premises
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all notebooks:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0[1-5]_*.ipynb
```

You can also open them with `jupyter lab` and run them top to bottom. Each notebook takes under a minute.

- Figures are written to `figures/` and tables to `tables/`.
- Each notebook checks its results against the reference values in `expected/`. It prints `[OK]` for every check and stops with an error if a result differs.

## Tables and figures

| Result | Notebook |
|---|---|
| Figure 2: average S by commitment level | `01_main_results` |
| Figure 3: effect of clause type | `01_main_results` |
| Tables 4–6: S per model, clause type and commitment level | `01_main_results` |
| Figures 8–11: bootstrap confidence intervals | `01_main_results` |
| Table 11: constrained vs. free-form output | `01_main_results` |
| Figure 5, Tables 16–17: ablations of the counterfactual CoT | `02_mitigation` |
| Figure 6: mitigation on DebateQA | `02_mitigation` |
| CoT mitigation on AITA and LFQA (appendix) | `02_mitigation` |
| Evidence sensitivity of the CoT (appendix) | `02_mitigation` |
| Figure 7, Table 15: FEVER | `03_fever` |
| Table 10: sampling temperature | `04_temperature` |
| Tables 7–9: paired t-tests across commitment levels | `05_statistical_analysis` |

The presupposition taxonomy (Tables 1–3) and the prompts are defined in `sway/presuppositions.py` and `notebooks/00_setup.ipynb`.

The ELEPHANT comparison (Table 13) and the human annotation study (Table 14) need the original post text, which we do not redistribute. Contact the authors for access.

## Repository structure

```text
sway/            Python package: scoring, statistics, plots, prompt construction, inference
notebooks/       00_setup (datasets and prompts) and 01–05 (reproduction)
collect/         Notebook and script used to collect model answers
data/outputs/    Collected model answers
expected/        Reference values used by the notebook checks
figures/         Generated figures
tables/          Generated tables
```

## Data

`data/outputs/` holds the answers of six models for every experiment:

- Llama 4 Scout
- Mistral Large
- Claude Sonnet 4.6
- Claude Opus 4.6
- Claude Haiku 4.5
- Gemma 3 4B

There is one CSV per model and experiment, and each row is one item:

| Column | Content |
|---|---|
| `item_id` | First 16 hex characters of the SHA-1 hash of the item text |
| `answer_pp_plus_<condition>` | Answer under the supportive presupposition |
| `answer_pp_minus_<condition>` | Answer under the opposing presupposition |
| `outcome` | Gold label (AITA only) |
| `ground_truth` | Gold label (FEVER only) |

There are 12 conditions: three commitment levels (`low`, `medium`, `high`) × four clause types. Prompts are not included; `sway/data_util.py` rebuilds them from the original datasets. Chain-of-thought answers that could not be parsed are kept verbatim.

## Collecting new answers

Collection queries models through AWS Bedrock. You need the original datasets in `data/`:

| Dataset | File |
|---|---|
| AITA | `AITA-YTA.csv`, `AITA-NTA-OG.csv` |
| LFQA | `LFQA-E-en.json` |
| DebateQA | `dev.jsonl` |
| FEVER | `paper_dev.jsonl` |

Install the collection dependencies and add your credentials:

```bash
pip install -r requirements-collect.txt
cp .env.example .env
```

Fill in `.env`, then run:

- `notebooks/00_setup.ipynb` for the main experiments. Model calls are commented out; uncomment the ones you want.
- `collect/collection.ipynb` for everything else. Set `RUN_COLLECTION = True` first.

Convert the raw outputs to the format in `data/outputs/`:

```bash
python collect/make_slim_data.py --raw <raw outputs folder> --out data/outputs
```

## Citation

```bibtex
@inproceedings{sway2026,
  title     = {{SWAY}ed by False Premises: Measuring and Mitigating Framing Sycophancy in Language Models},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

## License

MIT. See `LICENSE`.
