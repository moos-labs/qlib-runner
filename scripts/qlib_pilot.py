#!/usr/bin/env python3
"""Qlib pilot: LightGBM + Alpha158 benchmark on Qlib's hosted sample data.

Stages
------
1. DATA   Download Qlib's sample data into ~/.qlib/qlib_data/cn_data.
          Qlib's own route is tried first, per its current docs:
          ``python -m qlib.cli.data qlib_data --target_dir ... --region cn``
          (older docs name the same entry point ``qlib.run.get_data`` /
          ``scripts/get_data.py``; both module spellings are attempted).
          Qlib's README states the official dataset host is "disabled
          temporarily" and points to a community mirror instead; if the
          official routes do not produce a valid dataset, that documented
          mirror (chenditc/investment_data qlib_bin.tar.gz) is used and
          the report says so explicitly.
2. QRUN   Run the bundled LightGBM + Alpha158 benchmark workflow with
          ``qrun`` on the vendored copy of
          examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
          (market csi300, benchmark SH000300, train 2008-2014,
          valid 2015-2016, test/backtest 2017-01-01..2020-08-01).
3. REPORT Walk the mlflow recorder output (mlruns/) that qrun produced,
          quote the metrics qlib itself logged (IC / Rank IC and the
          portfolio risk analysis: annualized return, information ratio,
          max drawdown), copy the key analysis outputs into results/,
          and write results/qlib_result.md.

No number in the report is computed by this script: every figure is
copied verbatim from a recorder file, whose path is named alongside it.
"""
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RUN_DIR = REPO_ROOT / "qlib_run"
DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"
CONFIG_SRC = Path(__file__).resolve().parent / "workflow_config_lightgbm_Alpha158.yaml"
COMMUNITY_URL = (
    "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
)

REPORT_LINES = []


def log(msg=""):
    print(msg, flush=True)
    REPORT_LINES.append(str(msg))


def run_streamed(cmd, cwd=None, timeout=None):
    """Run a command, streaming its output to our stdout (and the report)."""
    log(f"$ {' '.join(str(c) for c in cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    start = time.time()
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        log(f"  -> TIMEOUT after {timeout}s")
        return 124, time.time() - start
    except FileNotFoundError as e:
        log(f"  -> NOT FOUND: {e}")
        return 127, time.time() - start
    dur = time.time() - start
    log(f"  -> exit {rc} in {dur:.1f}s")
    return rc, dur


# ---------------------------------------------------------------- data stage
def data_status():
    """Inspect DATA_DIR; return dict of facts (empty values when absent)."""
    st = {"ok": False, "calendar_first": None, "calendar_last": None,
          "calendar_days": 0, "csi300_instruments": None,
          "all_instruments": None, "feature_dirs": 0}
    cal = DATA_DIR / "calendars" / "day.txt"
    if cal.exists():
        lines = [ln.strip() for ln in cal.read_text().splitlines() if ln.strip()]
        if lines:
            st["calendar_first"], st["calendar_last"] = lines[0], lines[-1]
            st["calendar_days"] = len(lines)
    for name, key in (("csi300.txt", "csi300_instruments"), ("all.txt", "all_instruments")):
        f = DATA_DIR / "instruments" / name
        if f.exists():
            st[key] = len([ln for ln in f.read_text().splitlines() if ln.strip()])
    feat = DATA_DIR / "features"
    if feat.exists():
        st["feature_dirs"] = len([p for p in feat.iterdir() if p.is_dir()])
    st["ok"] = bool(
        st["calendar_days"] > 1000
        and st["csi300_instruments"]
        and st["feature_dirs"] > 100
        and st["calendar_first"] and st["calendar_first"] <= "2008-01-02"
        and st["calendar_last"] and st["calendar_last"] >= "2020-08-01"
    )
    return st


def try_official_get_data():
    """Qlib's own download route (current docs: qlib.cli.data)."""
    for mod in ("qlib.cli.data", "qlib.run.get_data"):
        rc, _ = run_streamed(
            [sys.executable, "-m", mod, "qlib_data",
             "--target_dir", str(DATA_DIR), "--region", "cn"],
            timeout=1200,
        )
        st = data_status()
        log(f"  data check after {mod}: ok={st['ok']} "
            f"calendar={st['calendar_first']}..{st['calendar_last']} "
            f"({st['calendar_days']} days), csi300={st['csi300_instruments']}, "
            f"feature_dirs={st['feature_dirs']}")
        if st["ok"]:
            return mod
    return None


def try_community_mirror():
    """Documented fallback while the official dataset host is disabled."""
    tar_path = REPO_ROOT / "qlib_bin.tar.gz"
    log(f"Downloading community mirror: {COMMUNITY_URL}")
    try:
        urllib.request.urlretrieve(COMMUNITY_URL, tar_path)
    except Exception:
        log("Download failed:\n" + traceback.format_exc())
        return False
    size_mb = tar_path.stat().st_size / 1e6
    log(f"Downloaded {size_mb:.1f} MB; extracting to {DATA_DIR} (strip 1 component)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path) as tf:
            members = []
            for m in tf.getmembers():
                parts = Path(m.name).parts
                if len(parts) > 1:
                    m.name = str(Path(*parts[1:]))
                    members.append(m)
            try:
                tf.extractall(DATA_DIR, members=members, filter="data")
            except TypeError:
                tf.extractall(DATA_DIR, members=members)
    except Exception:
        log("Extraction failed:\n" + traceback.format_exc())
        return False
    finally:
        tar_path.unlink(missing_ok=True)
    return data_status()["ok"]


# ------------------------------------------------------------- recorder stage
def find_mlruns():
    roots = [RUN_DIR / "mlruns", REPO_ROOT / "mlruns", Path.home() / "mlruns"]
    found = [p for p in roots if p.is_dir()]
    if not found:  # last resort: search the run dir tree
        for p in RUN_DIR.rglob("mlruns"):
            if p.is_dir():
                found.append(p)
    return found


def parse_recorder(mlruns_list):
    """Collect every metric qlib logged, with its source file."""
    metrics = []          # (exp_name, run_id, metric_name, value, source_path)
    artifacts = []        # (run_id, path)
    for mlruns in mlruns_list:
        for exp_dir in sorted(p for p in mlruns.iterdir() if p.is_dir()):
            exp_name = exp_dir.name
            meta = exp_dir / "meta.yaml"
            if meta.exists():
                for ln in meta.read_text(errors="replace").splitlines():
                    if ln.startswith("name:"):
                        exp_name = ln.split(":", 1)[1].strip().strip("'\"") or exp_name
            for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                mdir = run_dir / "metrics"
                if mdir.is_dir():
                    for mf in sorted(mdir.iterdir()):
                        if not mf.is_file():
                            continue
                        last = None
                        for ln in mf.read_text(errors="replace").splitlines():
                            parts = ln.split()
                            if len(parts) >= 2:
                                try:
                                    last = float(parts[1])
                                except ValueError:
                                    pass
                        if last is not None:
                            metrics.append((exp_name, run_dir.name, mf.name,
                                            last, str(mf)))
                adir = run_dir / "artifacts"
                if adir.is_dir():
                    for f in sorted(adir.rglob("*")):
                        if f.is_file():
                            artifacts.append((run_dir.name, f))
    return metrics, artifacts


def summarise_artifacts(artifacts):
    """Copy key analysis outputs into results/; describe every artifact."""
    copied, notes = [], []
    for run_id, f in artifacts:
        rel = f.name
        notes.append(f"- `{f}` ({f.stat().st_size} bytes)")
        if f.suffix == ".csv" and f.name.startswith("report"):
            dst = RESULTS / f"report_{run_id[:8]}_{f.name}" if False else RESULTS / f.name
            if dst.exists():
                dst = RESULTS / f"{run_id[:8]}_{f.name}"
            shutil.copy2(f, dst)
            copied.append(dst.name)
        elif f.suffix == ".pkl" and "port_analysis" in f.name:
            try:
                import pandas as pd
                obj = pd.read_pickle(f)
                dst = RESULTS / f.name.replace(".pkl", ".csv")
                if hasattr(obj, "to_csv"):
                    obj.to_csv(dst)
                    copied.append(dst.name)
                shutil.copy2(f, RESULTS / f.name)
                copied.append(f.name)
            except Exception:
                notes.append(f"  (could not convert {f.name}: "
                             f"{traceback.format_exc(limit=1)})")
                shutil.copy2(f, RESULTS / f.name)
                copied.append(f.name)
        elif f.name == "pred.pkl":
            try:
                import pandas as pd
                obj = pd.read_pickle(f)
                dst = RESULTS / "pred.csv"
                obj.to_csv(dst)
                copied.append(dst.name)
                notes.append(f"  (pred.pkl converted to pred.csv: "
                             f"{getattr(obj, 'shape', '?')} rows/cols)")
            except Exception:
                shutil.copy2(f, RESULTS / "pred.pkl")
                copied.append("pred.pkl")
    return copied, notes


def port_analysis_tables(artifacts):
    """Render port_analysis risk DataFrames (qlib's own numbers) as markdown."""
    tables = []
    for _run_id, f in artifacts:
        if f.suffix == ".pkl" and "port_analysis" in f.name:
            try:
                import pandas as pd
                obj = pd.read_pickle(f)
                if hasattr(obj, "to_markdown"):
                    tables.append((str(f), obj.to_markdown()))
                else:
                    tables.append((str(f), f"```\n{obj}\n```"))
            except Exception:
                tables.append((str(f), f"(unreadable: {traceback.format_exc(limit=1)})"))
    return tables


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stage_notes = []
    exit_code = 0

    log("# Qlib pilot run")
    log(f"Started (UTC): {datetime.now(timezone.utc).isoformat()}")
    log(f"Python: {sys.version.split()[0]}")
    try:
        from importlib.metadata import version
        for pkg in ("pyqlib", "lightgbm", "pandas", "numpy", "mlflow"):
            try:
                log(f"{pkg}: {version(pkg)}")
            except Exception:
                log(f"{pkg}: (not found)")
    except Exception:
        log("version probe failed:\n" + traceback.format_exc())

    # ---- Stage 1: data -------------------------------------------------
    log("\n## Stage 1: data download")
    route = None
    if data_status()["ok"]:
        route = "already present"
        log("Valid dataset already present; skipping download.")
    else:
        mod = try_official_get_data()
        if mod:
            route = f"official: python -m {mod} qlib_data --region cn"
        elif try_community_mirror():
            route = ("community mirror (Qlib README fallback while the "
                     "official dataset is disabled): " + COMMUNITY_URL)
        else:
            route = "FAILED"
    st = data_status()
    log(f"Data route used: {route}")
    log(f"Data dir: {DATA_DIR}")
    log(f"Calendar: {st['calendar_first']} .. {st['calendar_last']} "
        f"({st['calendar_days']} trading days)")
    log(f"Instruments: csi300.txt={st['csi300_instruments']}, "
        f"all.txt={st['all_instruments']}, feature dirs={st['feature_dirs']}")
    stage_notes.append(("data", route, st))

    if not st["ok"]:
        exit_code = 2
        log("\nBLOCKED at data stage: no valid cn_data dataset after all routes.")
    else:
        # ---- Stage 2: qrun ---------------------------------------------
        log("\n## Stage 2: qrun LightGBM + Alpha158")
        cfg_dst = RUN_DIR / CONFIG_SRC.name
        shutil.copy2(CONFIG_SRC, cfg_dst)
        qrun_bin = shutil.which("qrun")
        if qrun_bin:
            cmd = [qrun_bin, cfg_dst.name]
        else:
            cmd = [sys.executable, "-m", "qlib.cli.run", cfg_dst.name]
            log("qrun entry point not on PATH; falling back to qlib.cli.run module")
        qrun_out = RUN_DIR / "qrun_stdout.txt"
        log(f"qrun output is also tee'd to {qrun_out}")
        start = time.time()
        with open(qrun_out, "w") as fh:
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(RUN_DIR),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
            rc = proc.wait()
        dur = time.time() - start
        log(f"qrun exit {rc} in {dur:.1f}s")
        shutil.copy2(qrun_out, RESULTS / "qrun_output.txt")
        stage_notes.append(("qrun", rc, dur))
        if rc != 0:
            exit_code = rc if rc else 1

    # ---- Stage 3: recorder extraction ----------------------------------
    log("\n## Stage 3: recorder / mlflow extraction")
    mlruns_list = find_mlruns()
    log(f"mlruns dirs found: {[str(p) for p in mlruns_list]}")
    metrics, artifacts = parse_recorder(mlruns_list)
    log(f"metrics found: {len(metrics)}; artifact files found: {len(artifacts)}")
    copied, artifact_notes = summarise_artifacts(artifacts)
    risk_tables = port_analysis_tables(artifacts)

    # ---- Write qlib_result.md ------------------------------------------
    out = []
    out.append("# Qlib pilot result — LightGBM + Alpha158 (qlib benchmark)")
    out.append("")
    out.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    out.append(f"- Total wall time: {time.time() - t0:.1f}s")
    out.append(f"- Overall exit code: {exit_code}")
    out.append("")
    out.append("## Setup")
    out.append("")
    for ln in REPORT_LINES:
        if ln.startswith(("Python:", "pyqlib:", "lightgbm:", "pandas:",
                          "numpy:", "mlflow:")):
            out.append(f"- {ln}")
    out.append("- Workflow config: `scripts/workflow_config_lightgbm_Alpha158.yaml` "
               "(vendored from microsoft/qlib `examples/benchmarks/LightGBM/`)")
    out.append("- Market / instruments: **csi300** (config `market: csi300`), "
               "benchmark **SH000300**, region **cn**")
    out.append("- Segments: train 2008-01-01..2014-12-31, valid 2015-01-01..2016-12-31, "
               "test 2017-01-01..2020-08-01; backtest 2017-01-01..2020-08-01, "
               "TopkDropoutStrategy topk=50 n_drop=5")
    out.append("")
    out.append("## Data")
    out.append("")
    out.append(f"- Route used: {route}")
    out.append(f"- Data dir: `{DATA_DIR}`")
    out.append(f"- Calendar: {st['calendar_first']} .. {st['calendar_last']} "
               f"({st['calendar_days']} trading days, calendars/day.txt)")
    out.append(f"- Instruments: csi300.txt rows={st['csi300_instruments']}, "
               f"all.txt rows={st['all_instruments']}, feature dirs={st['feature_dirs']}")
    out.append("")
    out.append("## Recorder metrics (quoted verbatim from mlflow metric files)")
    out.append("")
    if metrics:
        out.append("| experiment | run | metric | value | source file |")
        out.append("|---|---|---|---|---|")
        for exp, run_id, name, val, src in metrics:
            out.append(f"| {exp} | {run_id[:8]} | {name} | {val!r} | `{src}` |")
    else:
        out.append("No mlflow metric files were found — the workflow did not "
                   "produce recorder metrics (see run.log / qrun_output.txt).")
    out.append("")
    out.append("## Portfolio risk analysis (port_analysis, qlib's own tables)")
    out.append("")
    if risk_tables:
        for src, table in risk_tables:
            out.append(f"Source: `{src}`")
            out.append("")
            out.append(table)
            out.append("")
    else:
        out.append("No port_analysis artifact was produced.")
    out.append("")
    out.append("## Recorder artifacts")
    out.append("")
    out.extend(artifact_notes if artifact_notes else ["(none found)"])
    out.append("")
    out.append("## Files copied into results/")
    out.append("")
    out.extend([f"- {c}" for c in copied] or ["(none)"])
    out.append("")
    out.append("## Full pilot log")
    out.append("")
    out.append("```")
    out.extend(REPORT_LINES)
    out.append("```")
    (RESULTS / "qlib_result.md").write_text("\n".join(out) + "\n")
    log(f"\nWrote {RESULTS / 'qlib_result.md'}")
    log(f"Finished in {time.time() - t0:.1f}s; exit code {exit_code}")
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "qlib_result.md").write_text(
            "# Qlib pilot result\n\nBLOCKED by an unhandled exception in "
            "scripts/qlib_pilot.py:\n\n```\n" + traceback.format_exc() + "\n```\n"
            "\nPartial log:\n\n```\n" + "\n".join(REPORT_LINES) + "\n```\n")
        sys.exit(1)
