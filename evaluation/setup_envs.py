"""Build and verify SWE-bench Lite bug environments (parallel, resumable).

Each environment: repo clone at the environment commit + virtualenv +
editable install + test patch applied + reproducible-failure verification +
git re-init at the buggy state (leak protection).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.swebench import load_dataset, setup, CANDIDATES  # noqa: E402


def main():
    bugs = {b.instance_id: b for b in load_dataset() if b.instance_id in CANDIDATES}
    todo = []
    for iid, b in sorted(bugs.items()):
        stp = Path("evaluation/data/envs") / iid / "status.json"
        if stp.exists():
            st = json.loads(stp.read_text())
            if st.get("ok"):
                print(f"skip {iid} (already ok)")
                continue
        else:
            # wipe any partial env so setup can retry from scratch
            d = Path("evaluation/data/envs") / iid
            if d.exists():
                import shutil
                shutil.rmtree(d, ignore_errors=True)
        todo.append(b)
    print(f"setting up {len(todo)} envs: {[b.instance_id for b in todo]}", flush=True)

    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(setup, b): b.instance_id for b in todo}
        for fut in futs:
            iid = futs[fut]
            try:
                st = fut.result()
                print(f"DONE {iid} ok={st['ok']} steps={[k for k, v in st['steps'].items() if not v.get('ok')]}", flush=True)
            except Exception as e:
                print(f"CRASH {iid}: {type(e).__name__}: {e}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
