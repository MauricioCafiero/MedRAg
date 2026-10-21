"""Pose feasibility checks (MLIP-backed). Each returns a CheckResult; PoseContext carries the shared state
(relaxed geometry, MD frames) so later checks reuse earlier work.

Execution order (fail fast, cheap first):
    relax_caps (stage 1: end caps only)  ->  clash_count, force_residuals  ->  relax_displacement  ->  ligand_strain, contact_persistence
    pose_ensemble runs across poses of one ligand.
Dynamics is NOT routine: md_stability runs only through final_stability(), on finalist poses that survived the above.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms, units
from ase.constraints import FixAtoms, Hookean
from ase.io import Trajectory, write
from ase.data import vdw_radii, atomic_numbers
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

from .system import PoseSystem, graph_only, sym_rmsd
from .thresholds import Thresholds

EV_TO_KCAL = 23.0605
Status = Literal["pass", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    value: float
    unit: str
    status: Status
    threshold: tuple
    detail: str
    cost_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"[{self.status.upper():4}] {self.name}: {self.value:.2f} {self.unit} - {self.detail}"


@dataclass
class PoseContext:
    system: PoseSystem
    calc: object
    thresholds: Thresholds = field(default_factory=Thresholds)
    relax_steps: int = 150
    relax_fmax: float = 0.3           # eV/A; loose on purpose (UMADock uses 0.30), we want a basin, not a minimum
    relax_radius: float = 6.0         # A; pocket residues farther than this from the ligand are frozen (cluster edge)
    workdir: str | None = None        # checkpoints: relax.traj / relax.log / relaxed.xyz; a rerun resumes from relax.traj
    cap_steps: int = 100
    md_steps: int = 1000
    caps_prepared: bool = False
    prep: dict = field(default_factory=dict)
    results: dict[str, CheckResult] = field(default_factory=dict)
    relaxed: object = None            # ase.Atoms after relax_displacement
    md_lig_frames: list = field(default_factory=list)
    _frozen: list | None = field(default=None, repr=False)

    def frozen(self) -> list[int]:
        """Constrained indices: UMADock's C-alpha anchors plus the frozen cluster edge."""
        if self._frozen is None:
            self._frozen = sorted(set(self.system.fixed) | set(self.system.frozen_edge(self.relax_radius)))
        return self._frozen

    def fresh(self, relaxed: bool = False):
        a = (self.relaxed if relaxed and self.relaxed is not None else self.system.complex).copy()
        a.info.update(self.system.complex.info)
        a.calc = self.calc
        a.set_constraint(FixAtoms(indices=self.frozen()))
        return a

    def ligand_block(self, relaxed: bool = False) -> str:
        """Ligand as an SDF mol block (explicit H) at its docked or relaxed coordinates, for the report overlay."""
        src = self.relaxed if relaxed else self.system.complex
        if src is None:
            raise ValueError("no relaxed geometry yet: run relax_displacement first")
        mol = Chem.Mol(self.system.lig_mol)
        conf = mol.GetConformer()
        for i, p in enumerate(src.positions[self.system.lig_slice]):
            conf.SetAtomPosition(i, p.tolist())
        return Chem.MolToMolBlock(mol)

    def _done(self, name, value, unit, detail, t0, **extra) -> CheckResult:
        r = CheckResult(name, float(value), unit, self.thresholds.grade(name, float(value)),
                        self.thresholds.cuts[name], detail, time.time() - t0, extra)
        self.results[name] = r
        return r


def _vdw(sym: str) -> float:
    r = vdw_radii[atomic_numbers[sym]]
    return 1.7 if np.isnan(r) else float(r)


def _heavy_pairs(ctx: PoseContext, atoms):
    s = ctx.system
    sym = np.array(atoms.get_chemical_symbols())
    lig = np.array([i for i in range(s.n_pocket, s.n_pocket + s.n_lig) if sym[i] != "H"])
    poc = np.array([i for i in range(s.n_pocket) if sym[i] != "H"])
    d = np.linalg.norm(atoms.positions[lig][:, None] - atoms.positions[poc][None], axis=2)
    return sym, lig, poc, d


# --- static checks -------------------------------------------------------------------------

def _count_clashes(ctx: PoseContext, atoms):
    sym, lig, poc, d = _heavy_pairs(ctx, atoms)
    bad = []
    for a, i in enumerate(lig):
        for b, j in enumerate(poc):
            polar = sym[i] in "NO" and sym[j] in "NO"
            limit = 2.4 if polar else 0.75 * (_vdw(sym[i]) + _vdw(sym[j]))
            if d[a, b] < limit:
                bad.append((d[a, b], ctx.system.labels[i], ctx.system.labels[j]))
    return sorted(bad)


def clash_count(ctx: PoseContext) -> CheckResult:
    t0 = time.time()
    bad = _count_clashes(ctx, ctx.system.complex)
    worst = "; ".join(f"{a}-{b} {d:.2f}A" for d, a, b in bad[:3]) or "none"
    return ctx._done("clash_count", len(bad), "pairs", f"worst: {worst}", t0, pairs=[(a, b, d) for d, a, b in bad])


def force_residuals(ctx: PoseContext, near: float = 5.0) -> CheckResult:
    """Max MLIP force on the ligand and on free pocket atoms within `near` A of it, before pocket relaxation (caps
    already relaxed). Atoms farther out are the cluster edge: their forces come from the cut-out (backbone C/O next to
    a cap reached 10 eV/A at 6-8 A from the ligand), not from the pose, so they are left out."""
    t0 = time.time()
    a = ctx.fresh()
    f = np.linalg.norm(a.get_forces(apply_constraint=False), axis=1)
    lig = a.positions[ctx.system.lig_slice]
    dist = np.linalg.norm(a.positions[:, None, :] - lig[None, :, :], axis=2).min(axis=1)
    free = np.ones(len(a), bool)
    free[ctx.frozen()] = False
    idx = np.flatnonzero(free & (dist <= near))
    top = idx[np.argsort(-f[idx])[:3]]
    detail = (f"within {near:g} A of the ligand ({len(idx)} atoms); top atoms: "
              + ", ".join(f"{ctx.system.labels[i]} {f[i]:.1f}" for i in top))
    return ctx._done("force_residuals", f[idx].max(), "eV/A", detail, t0, energy_eV=float(a.get_potential_energy()),
                     ligand_max=float(f[ctx.system.lig_slice].max()))


# --- relaxation-based checks ---------------------------------------------------------------

def _bfgs(ctx: PoseContext, a, name: str, steps: int, fmax: float):
    """BFGS with checkpointing: <workdir>/<name>.traj + .log are written every step, and a rerun resumes from the
    last saved geometry. Returns (converged, dyn, steps_already_done)."""
    logfile, trajectory, done_steps = None, None, 0
    if ctx.workdir:
        wd = Path(ctx.workdir)
        wd.mkdir(parents=True, exist_ok=True)
        traj_path = wd / f"{name}.traj"
        if traj_path.exists() and traj_path.stat().st_size:
            with Trajectory(str(traj_path)) as old:
                done_steps = len(old)
                if done_steps:
                    a.positions = old[-1].positions
        logfile, trajectory = str(wd / f"{name}.log"), Trajectory(str(traj_path), "a", a)
    dyn = BFGS(a, logfile=logfile, trajectory=trajectory)      # the log is appended on a resume
    converged = dyn.run(fmax=fmax, steps=max(steps - done_steps, 1))
    if trajectory is not None:
        trajectory.close()
    return converged, dyn, done_steps


def relax_caps(ctx: PoseContext) -> dict:
    """Stage 1: relax only the ACE/NME end caps (everything else frozen). The caps are placed from idealised geometry
    and carry huge forces (60-100 eV/A) that are an artefact of the cut-out, not of the pose; UMADock relaxes them as
    part of its constrained optimisation. Their relaxed positions are written back into the system, so every later
    check (static forces, pocket relaxation) starts from sane caps. Idempotent."""
    if ctx.caps_prepared or not ctx.system.caps:
        ctx.caps_prepared = True
        return {}
    t0 = time.time()
    a = ctx.system.complex.copy()
    a.info.update(ctx.system.complex.info)
    a.calc = ctx.calc
    a.set_constraint(FixAtoms(indices=[i for i in range(len(a)) if i not in set(ctx.system.caps)]))
    e0 = float(a.get_potential_energy())
    converged, dyn, done = _bfgs(ctx, a, "caps", ctx.cap_steps, ctx.relax_fmax)
    ctx.system.complex.positions[ctx.system.caps] = a.positions[ctx.system.caps]
    ctx.caps_prepared = True
    fn = np.linalg.norm(a.get_forces(apply_constraint=False)[ctx.system.caps], axis=1)
    info = dict(n_caps=len(ctx.system.caps), steps=dyn.get_number_of_steps() + done, converged=bool(converged),
                energy_drop_eV=e0 - float(a.get_potential_energy()), cap_fmax=float(fn.max()), seconds=time.time() - t0)
    ctx.prep["relax_caps"] = info
    return info


def relax_displacement(ctx: PoseContext) -> CheckResult:
    t0 = time.time()
    a = ctx.fresh()
    start = a.positions[ctx.system.lig_slice].copy()          # the docked pose, before any resume
    converged, dyn, done_steps = _bfgs(ctx, a, "relax", ctx.relax_steps, ctx.relax_fmax)
    if ctx.workdir:
        write(str(Path(ctx.workdir) / "relaxed.xyz"), a)
    ctx.relaxed = a.copy()
    ctx.relaxed.info.update(ctx.system.complex.info)
    heavy = ctx.system.lig_heavy()
    shift = np.linalg.norm(a.positions[ctx.system.lig_slice][heavy] - start[heavy], axis=1)
    rmsd = float(np.sqrt((shift ** 2).mean()))
    # final forces (cached from the last BFGS step): max per-atom norm is ASE's fmax; RMS over the free atoms
    fn = np.linalg.norm(a.get_forces(), axis=1)
    free = np.ones(len(a), bool)
    free[ctx.frozen()] = False
    f_max, f_rms = float(fn.max()), float(np.sqrt((fn[free] ** 2).mean()))
    steps_txt = f"{dyn.get_number_of_steps()} steps" + (f" (+{done_steps} resumed)" if done_steps else "")
    detail = (f"max atom shift {shift.max():.2f}A, {steps_txt}, "
              f"{'converged' if converged else 'NOT converged (step cap)'}; final fmax {f_max:.2f}, "
              f"RMS force {f_rms:.3f} eV/A")
    return ctx._done("relax_displacement", rmsd, "A", detail, t0, converged=bool(converged),
                     max_shift=float(shift.max()), energy_eV=float(a.get_potential_energy()),
                     final_fmax=f_max, final_rms_force=f_rms)


def _energy_relaxed(ligand, calc, steps=150):
    ligand.calc = calc
    BFGS(ligand, logfile=None).run(fmax=0.05, steps=steps)
    return ligand.get_potential_energy()


def ligand_species(ctx: PoseContext) -> dict:
    """The ligand as it actually is in the pocket-relaxed complex, defined by bonds, not by atom index.

    Every H is assigned to its nearest heavy atom in the relaxed geometry. A proton that moved from the ligand to the
    pocket (or from the pocket to the ligand) is a real effect of the relaxation, so the ligand's protonation state and
    charge change with it. Returns the RDKit species mol, its atom positions in the DOCKED frame (retained ligand atoms
    at their docked coordinates; a transferred H is placed on its new heavy atom using the relaxed bond vector),
    the net charge and a description of each proton move."""
    s, rel = ctx.system, ctx.relaxed
    P, n = s.n_pocket, s.n_lig
    sym = np.array(rel.get_chemical_symbols())
    pos = rel.positions
    heavy_c = np.flatnonzero(sym != "H")
    h_c = np.flatnonzero(sym == "H")
    d = np.linalg.norm(pos[h_c][:, None, :] - pos[heavy_c][None, :, :], axis=2)
    owner = dict(zip(h_c.tolist(), heavy_c[d.argmin(1)].tolist()))
    lig_heavy_local = s.lig_heavy()
    lig_heavy_c = {P + i for i in lig_heavy_local}
    orig_owner = {P + a.GetIdx(): P + a.GetNeighbors()[0].GetIdx() for a in s.lig_mol.GetAtoms() if a.GetAtomicNum() == 1}
    new_H: dict[int, list[int]] = {}
    for h, o in owner.items():
        if o in lig_heavy_c:
            new_H.setdefault(o, []).append(h)
    moves = []
    for h in h_c.tolist():
        was, now = orig_owner.get(h), (owner[h] if owner[h] in lig_heavy_c else None)
        if was != now:
            src = s.labels[was] if was is not None else s.labels[h].split(":")[0] + " (pocket)"
            dst = s.labels[now] if now is not None else s.labels[owner[h]]
            moves.append(f"H {src} -> {dst}")

    dock = s.complex.positions
    rw = Chem.RWMol(s.lig_mol)
    for i in sorted((a.GetIdx() for a in s.lig_mol.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        rw.RemoveAtom(i)
    new_idx = {i: k for k, i in enumerate(lig_heavy_local)}          # ligand-local heavy index -> index in rw
    coords = [dock[P + i] for i in lig_heavy_local]
    for o in sorted(new_H):
        for h in new_H[o]:
            k = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(new_idx[o - P], k, Chem.BondType.SINGLE)
            coords.append(dock[h] if orig_owner.get(h) == o else dock[o] + (pos[h] - pos[o]))
    for i in lig_heavy_local:
        at = rw.GetAtomWithIdx(new_idx[i])
        n_old = sum(1 for nb in s.lig_mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetAtomicNum() == 1)
        at.SetFormalCharge(at.GetFormalCharge() + len(new_H.get(P + i, [])) - n_old)
        at.SetNoImplicit(True)
        at.SetNumExplicitHs(0)
        at.SetNumRadicalElectrons(0)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k, xyz in enumerate(coords):
        conf.SetAtomPosition(k, [float(v) for v in xyz])
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return dict(mol=mol, pos=np.array(coords), charge=Chem.GetFormalCharge(mol), moves=moves,
                symbols=[a.GetSymbol() for a in mol.GetAtoms()])


def _species_atoms(sp: dict, positions) -> Atoms:
    a = Atoms(sp["symbols"], positions=positions)
    a.info["charge"] = sp["charge"]
    a.info["spin"] = 1
    return a


def ligand_strain(ctx: PoseContext, n_confs: int = 6, k_restraint: float = 5.0) -> CheckResult:
    """E(ligand at its DOCKED geometry) - E(lowest vacuum minimum of the same species), kcal/mol.

    The pose is not relaxed in the pocket first: that would relax away the strain being measured. Two details:
      * protonation: the species is taken from the pocket-relaxed complex (ligand_species), so a proton that transferred
        during relaxation is part of the ligand being evaluated, against that species' own minima;
      * bond-length noise: raw docked/crystal geometries are not at the MLIP's bond lengths (crystal C-C ~0.1 A long
        cost ~40 kcal/mol alone), so the docked geometry is relaxed with every heavy atom harmonically restrained to its
        docked position (k_restraint eV/A^2). The restraint energy is reported in `extra` so the bias is visible.
    Floored at 0 (the conformer search is not exhaustive)."""
    t0 = time.time()
    if ctx.relaxed is None:
        relax_displacement(ctx)
    sp = ligand_species(ctx)
    heavy = [i for i, sy in enumerate(sp["symbols"]) if sy != "H"]
    pos0 = sp["pos"]

    bound = _species_atoms(sp, pos0)
    bound.calc = ctx.calc
    bound.set_constraint([Hookean(i, tuple(pos0[i]), k_restraint, rt=0.01) for i in heavy])
    BFGS(bound, logfile=None).run(fmax=0.1, steps=200)
    e_bound = bound.get_potential_energy(apply_constraint=False)
    shift = np.linalg.norm(bound.positions[heavy] - pos0[heavy], axis=1)
    restraint_kcal = 0.5 * k_restraint * float((shift ** 2).sum()) * EV_TO_KCAL

    free = _species_atoms(sp, bound.positions)
    energies = [_energy_relaxed(free, ctx.calc)]
    mol = Chem.Mol(sp["mol"])
    mol.RemoveAllConformers()
    for cid in AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=7):
        energies.append(_energy_relaxed(_species_atoms(sp, mol.GetConformer(cid).GetPositions()), ctx.calc))
    raw = (e_bound - min(energies)) * EV_TO_KCAL
    moved = ("; proton moves: " + ", ".join(sp["moves"])) if sp["moves"] else ""
    return ctx._done("ligand_strain", max(0.0, raw), "kcal/mol",
                     f"species charge {sp['charge']:+d}, docked geometry vs lowest of {len(energies)} vacuum minima "
                     f"(heavy-atom restraint {restraint_kcal:.1f} kcal/mol, max shift {shift.max():.2f}A){moved}", t0,
                     raw_kcal=raw, restraint_kcal=restraint_kcal, proton_moves=sp["moves"], species_charge=sp["charge"],
                     n_conformers=len(energies))


def _polar_contacts(ctx: PoseContext, atoms, cutoff: float = 3.5):
    sym, lig, poc, d = _heavy_pairs(ctx, atoms)
    return {(i, j) for a, i in enumerate(lig) for b, j in enumerate(poc)
            if sym[i] in "NO" and sym[j] in "NO" and d[a, b] < cutoff}


def contact_persistence(ctx: PoseContext, slack: float = 0.5) -> CheckResult:
    """Fraction of the docked pose's polar (N/O...N/O < 3.5A) contacts still within 3.5+slack A after relaxation;
    also the mean over MD frames when md_stability has run."""
    t0 = time.time()
    if ctx.relaxed is None:
        relax_displacement(ctx)
    docked = _polar_contacts(ctx, ctx.system.complex)
    if not docked:
        return ctx._done("contact_persistence", 1.0, "fraction", "no polar contacts in the docked pose to track", t0)
    cur = ctx.relaxed.positions
    kept = [i for i, j in docked if np.linalg.norm(cur[i] - cur[j]) < 3.5 + slack]
    frac = len(kept) / len(docked)
    lost = [f"{ctx.system.labels[i]}-{ctx.system.labels[j]}" for i, j in docked
            if np.linalg.norm(cur[i] - cur[j]) >= 3.5 + slack]
    detail = f"{len(kept)}/{len(docked)} polar contacts kept" + (f"; lost: {', '.join(lost[:3])}" if lost else "")
    return ctx._done("contact_persistence", frac, "fraction", detail, t0, lost=lost, n_docked=len(docked))


def md_stability(ctx: PoseContext, temperature_K: float = 300.0, stride: int = 10) -> CheckResult:
    """Short NVT (Langevin) run from the relaxed pose; reports max ligand heavy-atom RMSD to the start and the
    mean polar-contact persistence over the trajectory. Frozen backbone/caps keep the pocket frame fixed, so RMSD
    needs no alignment."""
    t0 = time.time()
    if ctx.relaxed is None:
        relax_displacement(ctx)
    a = ctx.fresh(relaxed=True)
    MaxwellBoltzmannDistribution(a, temperature_K=temperature_K)
    dyn = Langevin(a, 1.0 * units.fs, temperature_K=temperature_K, friction=0.02 / units.fs, logfile=None)
    heavy = ctx.system.lig_heavy()
    ls = ctx.system.lig_slice
    ref = a.positions[ls][heavy].copy()
    docked = _polar_contacts(ctx, ctx.system.complex)
    rmsds, persist, frames = [], [], []

    def record():
        p = a.positions
        rmsds.append(float(np.sqrt(((p[ls][heavy] - ref) ** 2).sum(axis=1).mean())))
        frames.append(p[ls].copy())
        if docked:
            persist.append(np.mean([np.linalg.norm(p[i] - p[j]) < 4.0 for i, j in docked]))

    dyn.attach(record, interval=stride)
    dyn.run(ctx.md_steps)
    ctx.md_lig_frames = frames
    com_drift = float(np.linalg.norm(np.array(frames)[:, heavy].mean(axis=1) - np.array(frames)[0, heavy].mean(axis=0), axis=1).max())
    pp = float(np.mean(persist)) if persist else 1.0
    return ctx._done("md_stability", max(rmsds), "A",
                     f"{ctx.md_steps} fs @ {temperature_K:.0f}K; mean RMSD {np.mean(rmsds):.2f}A, "
                     f"COM drift {com_drift:.2f}A, polar contacts present {pp:.0%} of frames", t0,
                     mean_rmsd=float(np.mean(rmsds)), com_drift=com_drift, contact_persistence_md=pp)


# --- across poses of one ligand ------------------------------------------------------------

def pose_ensemble(ctxs: list[PoseContext], cluster_A: float = 2.0) -> CheckResult:
    """Fraction of the (docking-score-ordered) relaxed poses that fall in the largest basin (heavy-atom RMSD <=
    cluster_A, symmetry-aware, leader clustering). Needs relax_displacement already run on each context."""
    t0 = time.time()
    th = ctxs[0].thresholds
    graph = graph_only(ctxs[0].system.lig_mol)
    coords = []
    for c in ctxs:
        if c.relaxed is None:
            relax_displacement(c)
        coords.append(c.relaxed.positions[c.system.lig_slice][c.system.lig_heavy()])
    leaders: list[list[int]] = []                     # poses share atom order (one ligand, one docking run)
    for i, xyz in enumerate(coords):
        for grp in leaders:
            if sym_rmsd(graph, xyz, coords[grp[0]]) <= cluster_A:
                grp.append(i)
                break
        else:
            leaders.append([i])
    biggest = max(leaders, key=len)
    frac = len(biggest) / len(mols)
    r = CheckResult("pose_ensemble", frac, "fraction", th.grade("pose_ensemble", frac), th.cuts["pose_ensemble"],
                    f"{len(leaders)} basin(s) among {len(mols)} relaxed poses; dominant holds poses {[i + 1 for i in biggest]}",
                    time.time() - t0, {"basins": [[i + 1 for i in g] for g in leaders]})
    for c in ctxs:
        c.results["pose_ensemble"] = r
    return r


# --- orchestration -------------------------------------------------------------------------

def verdict(results: dict[str, CheckResult]) -> tuple[str, list[str]]:
    """Any fail -> reject; two or more warns -> demote; else accept."""
    fails = [r for r in results.values() if r.status == "fail"]
    warns = [r for r in results.values() if r.status == "warn"]
    reasons = [r.line() for r in fails + warns]
    return ("reject" if fails else "demote" if len(warns) >= 2 else "accept"), reasons


def run_checks(ctx: PoseContext) -> tuple[str, list[str]]:
    """Routine checks, cheap first; stop as soon as a pose is clearly bad. No MD: dynamics is a final stability
    check on the finalists only (final_stability)."""
    relax_caps(ctx)
    clash_count(ctx)
    force_residuals(ctx)
    if ctx.results["clash_count"].status == "fail" and ctx.results["force_residuals"].status == "fail":
        return verdict(ctx.results)
    relax_displacement(ctx)
    ligand_strain(ctx)
    contact_persistence(ctx)
    return verdict(ctx.results)


def final_stability(ctx: PoseContext) -> tuple[str, list[str]]:
    """Short MLIP MD on a finalist pose (run after run_checks, only for poses that survived it). Expensive:
    ~one UMA force call per step on the full pocket."""
    if verdict(ctx.results)[0] == "reject":
        return verdict(ctx.results)
    md_stability(ctx)
    # fold MD contact persistence into the same check, keeping the stricter of the two
    md_p = ctx.results["md_stability"].extra.get("contact_persistence_md", 1.0)
    cp = ctx.results["contact_persistence"]
    if md_p < cp.value:
        cp.value, cp.status = md_p, ctx.thresholds.grade("contact_persistence", md_p)
        cp.detail += f"; over MD {md_p:.0%}"
    return verdict(ctx.results)
