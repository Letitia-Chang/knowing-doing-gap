"""End-to-end run of every CPU stage on tiny synthetic activations (no GPU, no model)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROBE = Path(__file__).resolve().parent.parent / "src" / "probe"


def run(script, *args):
    r = subprocess.run([sys.executable, str(PROBE / script), *args], capture_output=True, text=True)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_cpu_stages_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n, layers, hidden = 80, 4, 12
    cats = np.array(["arithmetic", "factual"] * (n // 2))
    needed = (rng.random(n) < 0.5).astype(int)
    questions = [{"id": f"Q{i:04d}", "category": cats[i], "question": f"What is {i} + {i}?" if cats[i] == "arithmetic" else f"Who is person {i}?",
                  "tool_needed": bool(needed[i]), "tool_needed_acc80": bool(needed[i]), "tool_needed_acc50": bool(needed[i]),
                  "necessity_n": 10, "necessity_correct_count": 10 - 5 * int(needed[i])} for i in range(n)]
    (tmp_path / "questions_probe_labeled.json").write_text(json.dumps({"questions": questions}))
    acts = {}
    for name in ("activations", "activations_intervention"):
        a = rng.normal(size=(layers, n, hidden)).astype(np.float16)
        a[2, :, 0] += (3 * (needed - 0.5)).astype(np.float16)
        np.savez(tmp_path / f"{name}.npz", last_token=a, mean_pool=a, ids=np.array([q["id"] for q in questions]), meta="{}")
    for prompt, rate in (("baseline", 0.9), ("intervention", 0.5)):
        (tmp_path / f"action_local_{prompt}.json").write_text(json.dumps(
            [{"id": q["id"], "tool_called": bool(rng.random() < rate), "malformed": False} for q in questions]))
    (tmp_path / "run_info.json").write_text(json.dumps({"necessity": {"model": "m", "precision": "bf16", "gpu": "none", "torch": "x", "transformers": "y"}}))

    small = ["--outdir", str(tmp_path), "--folds", "3", "--repeats", "1", "--null-repeats", "1", "--n-boot", "30", "--n-jobs", "1", "--c-grid", "0.01,1"]
    run("run_probe.py", *small, "--activations", str(tmp_path / "activations.npz"))
    run("run_probe.py", *small, "--activations", str(tmp_path / "activations_intervention.npz"), "--tag", "_intervention", "--scopes", "all", "--no-mean-pool")
    run("knowing_doing.py", "--outdir", str(tmp_path))
    run("knowing_doing.py", "--outdir", str(tmp_path), "--prompt", "intervention", "--tag", "_intervention")
    run("make_figures.py", "--outdir", str(tmp_path))
    run("make_figures.py", "--outdir", str(tmp_path), "--tag", "_intervention")
    run("summarize.py", "--outdir", str(tmp_path))

    nested = json.loads((tmp_path / "probe_nested.json").read_text())["results"]
    pooled = next(r for r in nested if r["scope"] == "all" and r["feature"] == "last_token")
    assert pooled["nested"]["auroc"] > 0.8  # signal planted in layer 2
    assert pd.read_csv(tmp_path / "probe_layer_curve.csv").layer.nunique() == layers
    for f in ("probe_layer_curve.png", "four_cell.png", "calibration.png", "four_cell_intervention.png"):
        assert (tmp_path / "figures" / f).stat().st_size > 1000
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dataset"]["n"] == n and set(summary["knowing_doing"]) == {"baseline", "intervention"}
    assert "prompt_comparison" in summary and "Nested AUROC" in (tmp_path / "summary.md").read_text()
