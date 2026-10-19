"""Run all baselines (plan §7.4) and write evaluation/results/main/baselines.json.

- SBFL: Ochiai, DStar^2, DStar^3, Tarantula on the precomputed spectra.
- One-shot LLM: one tool-less call per bug; ranking completed with the
  spectrum tail exactly like the agent configs (equal footing, plan §5.9).
"""
from __future__ import annotations

import json
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import swebench as swe
from evaluation import baselines as bl
from lethe.agent import DEFAULT_BASE, DEFAULT_MODEL

PRECOMP = ROOT / "evaluation/data/precomputed"
OUT = ROOT / "evaluation/results/main/baselines.json"
SELECTED = ROOT / "evaluation/data/precomputed/_selected.json"


def main():
    keep = json.loads(SELECTED.read_text()) if SELECTED.exists() else sorted(
        p.stem for p in PRECOMP.glob("*.json") if not p.stem.startswith("_"))
    out = {}
    for iid in keep:
        doc = json.loads((PRECOMP / f"{iid}.json").read_text())
        bug = next(b for b in swe.load_dataset() if b.instance_id == iid)
        row = {"sbfl": {}}
        for formula in ("ochiai", "dstar2", "dstar3", "tarantula"):
            row["sbfl"][formula] = bl.sbfl_for_bug(doc, formula)
        try:
            os_res = bl.oneshot_for_bug(bug, doc, base=DEFAULT_BASE, model=DEFAULT_MODEL)
            row["oneshot"] = {
                "n_accepted": os_res.get("n_accepted", 0),
                "stmt": os_res.get("stmt", {}),
                "method": os_res.get("method", {}),
            }
        except Exception as e:  # LLM outage must not kill SBFL baselines
            row["oneshot"] = {"error": str(e)[:300]}
        out[iid] = row
        print(f"[baselines] {iid}: "
              f"ochiai-stmt fr={row['sbfl']['ochiai']['stmt']['first_rank']} "
              f"oneshot stmt fr={row['oneshot'].get('stmt', {}).get('first_rank')}",
              flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
