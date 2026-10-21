"""Pass/warn/fail cut-offs for the feasibility checks.

The defaults are PROVISIONAL starting points, not validated numbers. calibrate.py runs every check on the
crystal pose of the seed ligand and derives target-specific cut-offs (never tighter than these defaults),
which are then saved next to the run and loaded by the pipeline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# name: (warn, fail), higher value = worse unless listed in LOWER_IS_WORSE
DEFAULTS: dict[str, tuple[float, float]] = {
    "ligand_strain":      (6.0, 12.0),    # kcal/mol, relaxed bound ligand vs vacuum minimum
    "relax_displacement": (1.0, 2.0),     # A, ligand heavy-atom RMSD docked -> relaxed in pocket
    "clash_count":        (1, 3),         # heavy-atom pairs closer than 0.75*(vdW sum) (polar pairs: 2.4 A)
    "force_residuals":    (3.0, 8.0),     # eV/A, max force on ligand / free pocket atoms before relaxing
    "md_stability":       (2.0, 3.5),     # A, max ligand heavy-atom RMSD during short MD
    "contact_persistence": (0.75, 0.5),   # fraction of docked polar contacts kept after relax (lower = worse)
    "pose_ensemble":      (0.6, 0.35),    # fraction of top-N relaxed poses in the dominant basin (lower = worse)
}
LOWER_IS_WORSE = {"contact_persistence", "pose_ensemble"}


@dataclass
class Thresholds:
    cuts: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULTS))
    source: str = "defaults"

    def grade(self, name: str, value: float) -> str:
        warn, fail = self.cuts[name]
        if name in LOWER_IS_WORSE:
            return "fail" if value < fail else "warn" if value < warn else "pass"
        return "fail" if value > fail else "warn" if value > warn else "pass"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"source": self.source, "cuts": self.cuts}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        p = Path(path)
        if not p.is_file():
            return cls()
        d = json.loads(p.read_text())
        return cls(cuts={k: tuple(v) for k, v in d["cuts"].items()}, source=d.get("source", str(p)))

    def relative_to_seed(self, seed_strain: float, margin: float = 3.0) -> "Thresholds":
        """Strain cut-offs no tighter than the seed's own strain plus a margin (warn: +margin, fail: +2*margin). Absolute strain
        cut-offs reject a flexible, charged lead and everything like it (HMGCR: rosuvastatin 29 kcal/mol against a 12 cut-off,
        largely because its vacuum reference folds onto itself); what matters for hit-to-lead is being clearly more strained
        than the lead."""
        w, f = self.cuts["ligand_strain"]
        cuts = dict(self.cuts)
        cuts["ligand_strain"] = (max(w, seed_strain + margin), max(f, seed_strain + 2 * margin))
        return Thresholds(cuts=cuts, source=f"{self.source}; strain relative to the seed ({seed_strain:.1f} kcal/mol + {margin:g})")

    @classmethod
    def from_reference(cls, ref: dict[str, float], warn_x: float = 1.5, fail_x: float = 2.5) -> "Thresholds":
        """Loosen (never tighten) the defaults so the crystal-pose reference itself passes with margin."""
        cuts = dict(DEFAULTS)
        for name, val in ref.items():
            if name not in cuts or val is None:
                continue
            w, f = cuts[name]
            if name in LOWER_IS_WORSE:
                cuts[name] = (min(w, val / warn_x), min(f, val / fail_x))
            else:
                cuts[name] = (max(w, val * warn_x), max(f, val * fail_x))
        return cls(cuts=cuts, source="calibrated from crystal pose")
