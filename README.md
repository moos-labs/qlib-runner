# qlib-runner

Runs [Qlib](https://github.com/microsoft/qlib) (Microsoft) analyses on
GitHub Actions, where the commands to execute live in a repository
**Actions variable** — not in code.

## The pattern (one repo per tool)

1. **Edit the commands**: repo *Settings → Secrets and variables → Actions →
   Variables* → set the `COMMANDS` variable to whatever shell commands this
   run should execute (multi-line works). The same can be done through the
   GitHub API (`PUT /repos/{owner}/{repo}/actions/variables/COMMANDS`).
2. **Trigger manually**: *Actions* tab → “Run analysis” → *Run workflow*
   (or `POST /repos/{owner}/{repo}/actions/workflows/run.yml/dispatches`).
   The workflow has **no other triggers** — it runs only when dispatched.
3. **Collect results**: the workflow uploads everything under `results/` as
   the `results` build artifact (`if: always()`, so partial results survive
   a failing step). Download it from the run page or via the API.

The workflow itself is generic: checkout → Python 3.11 →
`pip install -r requirements.txt` → run `${{ vars.COMMANDS }}` → upload
`results/`. To run a different analysis, change the variable — not the
workflow.

## Multi-market by design

Nothing in the workflow or the runner pattern is tied to one exchange.
Scripts take the **ticker symbol and analysis parameters as inputs** (CLI
args and/or the `COMMANDS` variable), so the same repo pattern can analyse
an EGX stock today and a US stock tomorrow — only the inputs change.
This pilot uses Qlib's own China (csi300) sample data because that is the
dataset Qlib's bundled benchmark config is written against; Qlib itself
also supports US-market data and custom datasets.

## Pilot: LightGBM + Alpha158 benchmark (Qlib's own workflow)

Pilot `COMMANDS` value:

```sh
mkdir -p results
set -o pipefail; python scripts/qlib_pilot.py 2>&1 | tee results/run.log
```

`scripts/qlib_pilot.py` runs three stages:

1. **Data** — download Qlib's hosted sample data to
   `~/.qlib/qlib_data/cn_data` (region `cn`). Qlib's own route is tried
   first, per its current docs:
   `python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn`
   (older docs spell the same entry point `qlib.run.get_data` /
   `scripts/get_data.py`; both spellings are attempted).
2. **Benchmark** — run the bundled LightGBM + Alpha158 workflow with
   `qrun`, using the vendored copy of Qlib's
   `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
   in `scripts/` (the pip package does not ship `examples/`).
   Config facts: market **csi300**, benchmark **SH000300**,
   handler **Alpha158**, model **LGBModel**; train 2008–2014,
   valid 2015–2016, test/backtest 2017-01-01 → 2020-08-01,
   TopkDropoutStrategy (topk 50, n_drop 5).
3. **Report** — walk the mlflow recorder output (`mlruns/`) and write
   `results/qlib_result.md` quoting the metrics Qlib itself logged
   (IC / Rank IC from SigAnaRecord, and the PortAnaRecord risk table:
   annualized return, information ratio, max drawdown), naming the
   recorder file each figure came from. Key analysis outputs
   (`report_*.csv`, `port_analysis_*`, predictions) are copied into
   `results/` when the workflow produces them, along with the full
   `qrun` output (`results/qrun_output.txt`).

### Honest notes

- **Official data host status**: Qlib's README currently says the official
  dataset is *"disabled temporarily"* due to a stricter data-security
  policy, and points users to a community mirror
  (`chenditc/investment_data`, `qlib_bin.tar.gz`). The pilot tries the
  official route first and only falls back to that documented mirror if
  no valid dataset results; `qlib_result.md` records exactly which route
  supplied the data.
- **Sample data quality**: Qlib's own warning applies — the sample data
  was collected from Yahoo Finance and *"might not be perfect"*; Qlib
  recommends preparing your own data for serious work. Treat the pilot's
  IC/backtest figures as a pipeline verification, not investment research.
- **No invented numbers**: this repo never computes replacement metrics.
  If a stage blocks, `qlib_result.md` says which stage and why, and the
  run log keeps the exact error.
- **OpenMP**: LightGBM needs `libgomp1` on Linux; the workflow's install
  step installs it via apt before pip.
- **Dependency pins**: `requirements.txt` pins numpy 1.26 / pandas 2.2 /
  mlflow <3 around `pyqlib==0.9.6`, the combination verified for
  Python 3.11; unpinned resolution currently drifts to majors that
  break Qlib. `joblib` is pinned to 1.2.0 specifically: Qlib's
  `ParallelExt` uses joblib internals removed in joblib ≥ 1.3, and with
  a newer joblib `qrun` fails during dataset loading
  (`AttributeError: 'ParallelExt' object has no attribute '_backend_args'`).
- **Runtime**: the full 2008–2020 Alpha158 benchmark is the real
  benchmark, not a trimmed demo — expect the qrun stage to take tens of
  minutes on a standard GitHub runner.
