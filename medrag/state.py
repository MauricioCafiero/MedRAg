"""Per-run session state: runs/<run_id>/state.json, written atomically after every pipeline step so a crashed or
interrupted run resumes from the last finished step (same rule as the long MLIP jobs: results land incrementally).

candidates maps canonical SMILES -> record; each step adds its own key to a record:
    {"smiles", "sources": [{"source", ...}], "filter": {"status", "reasons", "props", "admet"},
     "dock": {...}, "validate": {...}}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class RunState:
    def __init__(self, run_id: str, root: str | Path = "runs"):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"
        self.data = dict(run_id=run_id, target={}, seed=None, candidates={}, steps=[])
        if self.path.is_file():
            self.data = json.loads(self.path.read_text())

    # --- access ---------------------------------------------------------------
    @property
    def candidates(self) -> dict:
        return self.data["candidates"]

    @property
    def target(self) -> dict:
        return self.data["target"]

    def set_target(self, **fields):
        self.data["target"].update(fields)
        self.save()

    def add_candidate(self, smiles: str, source: str, **meta) -> dict:
        rec = self.candidates.setdefault(smiles, dict(smiles=smiles, sources=[]))
        rec["sources"].append(dict(source=source, **meta))
        return rec

    def log_step(self, step: str, **info):
        self.data["steps"].append(dict(step=step, time=time.strftime("%Y-%m-%d %H:%M:%S"), **info))
        self.save()

    def step_done(self, step: str) -> bool:
        return any(s["step"] == step for s in self.data["steps"])

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)          # atomic: a crash mid-write cannot leave a half-written state.json
