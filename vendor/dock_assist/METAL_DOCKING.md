# Metal-site docking — integration notes and fixes (2026-10-14)

Engineering log for adding metal support to Modrag-dock: what was tried, what
was installed, what broke, and how each failure was fixed. The user-facing docs
live in [`README.md`](README.md#metal-site-docking-ad4zn); this file is the
"why it ended up this way" record. Tool behavior facts verified empirically on
this machine (Apple Silicon, macOS 26 / Darwin 25.5) are marked **[measured]**.

---

## 1. Decision: which metal approach

Options considered, in order of evaluation:

| Option | Verdict |
|---|---|
| AD4Zn (Scripps ADFR suite, AutoGrid + `AD4Zn.dat`) | **Chosen for Zn²⁺.** Zinc-specific, directionally calibrated (292 crystal complexes; Santos-Martins et al., JCIM 2014). |
| AutoDock-GPU / AutoDock4 | **Dropped.** Once the AD4Zn force field was understood, no second engine is needed: the zinc chemistry lives in the *maps*, and Vina ≥ 1.2 docks against them with `--scoring ad4 --maps`. One engine, native Apple Silicon build. |
| Keep Zn in plain Vina 1.1.2 | **Rejected as the primary path but kept as fallback for non-zinc.** Tested: Vina accepts (and *repurposes*) metal atom types, but treats them as generic vdW spheres. [measured] |
| MetalDock | Deferred. Third-party install, smaller community; costs more than AD4Zn for the same scope. |
| Calibrated TZ-analogues for Mg/Ca/Fe (literature params, new geometry) | Deferred. Published LJ parameters exist (Merz 12-6-4 etc.) but the *application* is unvalidated without a redocking benchmark — see §6. |

**Why the plain-Vina option is weak for Zn.** Probed with a hand-built
one-atom receptor PDBQT (dockstring's vendored Vina 1.1.2) plus rigid single-atom
ligands [measured]:

- Vina parses atom type `Zn` without complaint and *prefers* the site (pose
  ~3.0 Å from the ion vs ~15 Å in an empty box).
- But N and C ligand atoms settle at the identical distance (~3.0 vs 3.7 Å for
  carbon-typed) — no donor preference, no tetrahedral geometry, ~3 Å instead
  of the crystallographic 2.0–2.2 Å. A carbon and a nitrogen are "coordinated"
  equally, which is chemically wrong for a zinc site.

Conclusion: metal-aware *scoring* requires the AD4Zn maps; metal-aware *pose
selection* requires a post-check. Both were built.

---

## 2. Toolchain — what was installed, all user-space

No system installers were used anywhere (brew/docker/`curl|sh` all avoided):
this fits the repo's install policy and dodged both Gatekeeper and x86_64-binary
problems reported for the ADFR suite on recent macOS.

### 2.1 Vina ≥ 1.2 (native Apple Silicon)
```bash
conda create -n vina -c conda-forge vina
```
conda-forge ships the **CLI** (`vina`, `vina_split`) plus Python bindings,
native `osx-arm64`, currently 1.2.7. The dockstring-vendored 1.1.2 binary
supports neither `--scoring ad4` nor `--maps`.

### 2.2 ADFR suite (osx-64 under Rosetta 2)
```bash
CONDA_SUBDIR=osx-64 conda create -n adfr-suite -c hcc adfr-suite
```
The only conda `adfr-suite` anywhere (HCC channel) is **1.0, uploaded
April 2020** — old but sufficient: it ships exactly
**AutoGrid 4.2.7.x.2019-07-11**, the version the AD4Zn tutorial requires
(`nbp_r_eps` semantics changed in 2020 builds vs 4.2.6), plus `pythonsh`
(Python 2.7.3, MGLTools) that runs the AD4Zn scripts. The 2020 vintage means
the docker route people recommend for the Scripps tarball is unnecessary.

### 2.3 autogrid4 dylib patch (the one binary-level fix)
`autogrid4` from the conda package embeds build-machine paths and dies:

```
dyld: Library not loaded: /Users/Shared/mgltoolsDev/src/homebrew/opt/gcc/lib/gcc/8/libgomp.1.dylib
```

Only autogrid4 is affected — `agfr`, `obabel`, `pythonsh`, `adcp` in the same
package are clean [measured: `otool -L` sweep]. The patch (inside the conda
env, no system changes):

1. Extracted `libgomp.1.dylib` + `libgcc_s.1.dylib` from the old Anaconda
   **`free` channel `gcc-4.8.5` osx-64 conda tarball** (plain download, no
   install) into `$ENV/lib/`.
2. That libgomp **segfaulted [measured]** — and not on thread creation:
   139 even with `OMP_NUM_THREADS=1`.
3. Replaced it with `libgomp.1.dylib` from the current **Homebrew `gcc@14`
   bottle**: fetch an anonymous ghcr token
   (`https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/gcc/14:pull`)
   and pull the blob directly; no brew install. (`gcc@8` itself is gone from
   Homebrew — formula deleted, 404 from formulae.brew.sh.)
4. `install_name_tool -change <scripps-path> @executable_path/../lib/<dylib>`
   on the binary; original kept as `bin/autogrid4.orig`.

### 2.4 AD4Zn scripts + parameter file
The ADFR suite does **not** ship the zinc helpers. Three files are fetched from
the official [AutoDock-Vina repo](https://github.com/ccsb-scripps/AutoDock-Vina)
into `dock_assist/ad4zn/`:

- `example/autodock_scripts/zinc_pseudo.py` — places tetrahedral `TZ`
  pseudo-atoms around each ZN ion.
- `example/autodock_scripts/prepare_gpf4zn.py` — writes the GPF with the
  calibrated `nbp_r_eps` pair terms (NA–TZ, OA–Zn, SA–Zn, N–Zn…).
- `data/AD4Zn.dat` — the zinc parameter file. (Note: the copy under
  `example/docking_with_zinc_metalloproteins/data/` is a repo *symlink*;
  downloading it yields a 23-byte text file, not the real 121-line .dat.)

### 2.5 meeko (dock-env, pip)
`meeko` 0.8 (+`gemmi`, `prody` — the receptor-prep dependencies) preps the
AD4Zn receptor/ligand PDBQTs, matching the tutorial. In `requirements.txt`.

Locations are overridable by env vars so setups can move: `ADFR_PYTHONSH`,
`AUTOGRID_BIN`, `VINA_12_BIN`, `AD4ZN_DIR`.

---

## 3. Runtime fixes, in test order

Every one of these was hit and fixed during the 1s63 validation on the actual
integration path; each is now baked into `vina_dock.py` with a comment.

### 3.1 venv scripts, not PATH
`shutil.which("mk_prepare_*.py")` returned nothing when dock-env's interpreter
is invoked directly (venv bin not on PATH). → `_meeko_script()` resolves
against `os.path.dirname(sys.executable)` first, PATH second.

### 3.2 GPF references the receptor *relatively*
`prepare_gpf4zn.py` echoes `receptor`/`gridfld` from the `-r` path; when passed
an absolute path, the GPF still ends up resolved relative to pythonsh's cwd
(MolKit `Read()` asserts on the relative name). → generate the GPF and run
autogrid with `cwd=work`, with the TZ receptor copied in as `rec.pdbqt`, which
also makes the map prefix deterministic (`work/rec.<TYPE>.map`).

### 3.3 `AD4Zn.dat` is basenamed into the GPF → benign-looking SIGSEGV
The GPF's `parameter_file` line stores the *basename*. AutoGrid therefore
opens `AD4Zn.dat` **relative to its cwd**; if absent it calls
`stop(NULL)` → `fputs(NULL)` → SIGSEGV. The macOS crash report exposes it
(`stop(char const*)`, `fputs` symbolLocation=114) but the symptom looks like a
memory corruption bug, complete with directory-dependent reproducibility —
crashed in any dir without the file, "worked" in the dir where a copy happened
to sit from earlier manual tests. [measured]

Diagnostic lesson: for a "segfault" in old Scripps tooling, read the crash
report *frame* before blaming memory — here it was a missing-file error path.

Fix: `shutil.copy(AD4ZN_DAT, work/AD4Zn.dat)` before autogrid.

### 3.4 `gridcenter auto` is not the site
AutoGrid's `auto` centers the grid on the **receptor bounding box**, not the
ligand or the metal. First map run at auto-center: score **+1.5**, pose in
empty space, `INTER 0.000`; with the center pinned to the metal: **−12.2**.
The metal path always passes an explicit `gridcenter` (the ion's coordinates).

### 3.5 Vina 1.2 removed `--log`
Both `run_vina` (organic-docking branch of the metal path) and the ad4-maps
runner initially passed `--log`; 1.2.7 rejects it as unrecognised and 1.1.2
*needs* it. → `run_vina(..., no_log=True)` for 1.2 binaries, and the mode
table (which arrives on stdout) is written to `log_path` so the single
`parse_vina_log()` keeps working for both versions.

### 3.6 PDBQT column-assembly quirks (hand-built probes only)
Vina 1.1.2's parser is strict about fixed PDBQT columns; probes built with
sloppy column widths die with `ATOM syntax incorrect`. Assembling lines by
exact column positions (x at 31–38, y 39–46, z 47–54, type at 78–79) parsed
fine. Also: receptor PDBQTs must be flat `ATOM` records — `ROOT/TORSDOF`
records are ligand-only and rejected as "Unknown or inappropriate tag".

---

## 4. What was wired and how

- `vina_dock.py`: `detect_metal_sites()` / `detect_metal_sites_agent` (pure-PDB
  HETATM scan; contacts = any atom ≤ 2.5 Å), `dock_metal_site()` /
  `dock_metal_site_agent`. Engine chosen **in code** from the *detected*
  element (LLM never passes it — the tool-signature gotcha). The agent
  supplies only the site center; the code snaps it to the nearest ion ≤ 3 Å.
- Engines: `ad4zn` (zinc) vs `vina-metal` (everything else — metal kept in the
  receptor PDBQT via `build_receptor_pdbqt_with_metals()`, waters/cofactors
  still dropped, Vina 1.2 generic scoring; every pose post-checked: donor
  N/O/S–ion distance, ≤ 2.5 Å = coordination).
- Caches live next to the input PDB: `<stem>_mk.pdbqt` (meeko receptor),
  `<stem>_mkTZ.pdbqt` (+ TZ), `<stem>_metal.pdbqt`; maps/GPFs are per-ligand
  because `prepare_gpf4zn` derives ligand atom types per molecule. The tier
  0/1 engine's receptor cache is `<stem>_map.pdbqt`.
- `dock_assist.py`: both tools registered (tool list, dispatch map, chat
  `tools=`, 3×) plus a metalloprotein paragraph in the system prompt.
- `run_vina` gained the backward-compatible `no_log` flag (default preserves
  the 1.1.2 organic-path behavior exactly).

---

## 5. Validation record

| Test | Setup | Result |
|---|---|---|
| Vina metal semantics | 1-atom Zn receptor, rigid single-atom ligands | Zn accepted; C and N settle equally (~3.0 Å) → vdW sphere [measured] |
| Detection | 1s63 | ZN (17:18.142, 132.126, 5.224), contacts ASP B297 / CYS B299 / HIS B362 |
| AD4Zn, tutorial params | manual 40³ grid at site | −12.21 kcal/mol (tutorial ref ≈ −13.5) |
| AD4Zn, full tool | 22 Å box, SMILES-generated ligand, exhaustiveness 16 | **−12.08 kcal/mol; chelating N at 1.91 Å from Zn** (crystal Zn–N ≈ 2.0–2.2) |
| vina-metal engine | same receptor with ZN→MG, full tool | docks (Vina 1.2.7 accepts `Mg` type), coordination honestly reported absent (5.91 Å) |
| Error paths | center with no ion ≤ 3 Å | clean DockError pointing at `detect_metal_sites_agent` |

Re-verification one-liner (receptor at `/tmp/ad4zntest/proteinH.pdb`):

```python
import sys; sys.path.insert(0, 'code')
import vina_dock
print(vina_dock.dock_metal_site('/tmp/ad4zntest/proteinH.pdb',
    ['N#Cc1ccc(Cn2cncc2CN2CCN(c3cccc(Cl)c3)C(=O)C2)cc1'],
    center=(18.142, 132.126, 5.224), exhaustiveness=16, seed=1))
```

Expect: engine `ad4zn`, best ≈ −12 kcal/mol, "COORDINATES ZN at 1.91 A".

---

## 6. Known limitations and the path forward

1. **Scores from different engines aren't comparable.** AD4Zn (AutoGrid ad4
   maps) and the organic Vina scoring function are different force fields;
   never rank across them.
2. **Non-zinc metals are qualitative.** `vina-metal` keeps the ion but scores
   it as a sphere; the honest readout is the coordination post-check, not the
   kcal/mol number. For Mg²⁺/Ca²⁺ chemistry (spherical electrostatics) this
   is less wrong than for Zn.
3. **Literature-parameter pseudo-atom sets for other metals (Mg/Ca/Fe
   octahedra etc.) are deferred as *unproven*.** Published LJ parameters
   (Merz 12-6-4, etc.) exist, but no one has validated them as *docking pose
   ranks* inside the AD4 scoring landscape — that was Scripps' own move for
   AD4Zn (fit on 292 zinc complexes). To make such a set trustworthy, build a
   redocking benchmark: ~30–50 metal-bound crystal complexes from the PDB,
   redock, measure pose RMSD / coordination recovery. That script would also
   double as a regression test for this whole pipeline.
4. **Reproducibility.** Docking seeds are passed by default, but AutoGrid map
   generation and meeko prep are deterministic, so a full metal run is
   reproducible given the same tool versions pinned above.

---

## 7. Tiered metal parameters (tier 0 + tier 1, added 2026-10-14 PM)

Extending §6's option-3 discussion, parameter coverage is now tiered by how
much literature parameterization each metal has, with engines chosen in code:

| Engine | Metals | Parameters |
|---|---|---|
| `ad4zn` | Zn²⁺ | (unchanged) full AD4Zn: TZ pseudo-atoms + pairwise terms fit by Scripps on 292 complexes |
| `ad4map` (tier 0) | Mg²⁺, Ca²⁺, Mn²⁺, Fe²⁺ | **stock values already in `AD4Zn.dat`** — zero new parameters; ion docked through AutoGrid maps instead of Vina's vdW sphere |
| `ad4merz` (tier 1) | Cu²⁺, Co²⁺, Ni²⁺, Cd²⁺, Hg²⁺, Na⁺, K⁺ | metals stock `AD4Zn.dat` doesn't parameterize at all: **Li & Merz 12-6-4 set (TIP3P, Li & Merz, JCTC 2014, 10, 289)** Rmin/2+ε for divalent; **Joung & Cheatham (TIP3P, JPCB 2008)** for monovalent. Written into `ad4zn/AD4Merz.dat` (Rii = 2×Rmin/2, epsii = ε; solper −0.00110; vol mirroring the nearest stock metal) |
| `vina-metal` | anything else | unchanged fallback |

### What tier 1 conversion does and doesn't carry over

The 12-6-4 potential is E = A/r¹² − B/r⁶ − C₄/r⁴; the r⁻⁴ induced-dipole term
is what pulls ion–donor equilibria in to ~2.0–2.1 Å, and AutoGrid pairwise
terms cannot express an r⁻⁴ shape. So `AD4Merz.dat` carries the exact 12-6
part and drops C₄: bare metal–donor contacts would run longer than
crystallographic coordination — the same wall the Scripps authors hit
(why they fit directional TZ terms for Zn), and the explicit advice in
[AutoDock-Vina issue #368](https://github.com/ccsb-scripps/AutoDock-Vina/issues/368)
is "follow the AutoDock4Zn procedure to derive your own parameters" for Mg —
i.e., tier 1 is a provenance improvement (real literature values vs. missing
types), not a zinc-grade force field. C₄'s geometry/depth contribution is now
absorbed into the effective `nbp_r_eps` donor wells (see "Why the tier maps
needed…"), which is an estimate until a redocking benchmark validates the
well depths; the benchmark remains the future step.
`AD4Merz.dat` is otherwise a copy of `AD4Zn.dat` (zinc calibrated terms kept,
Zn/Mg/Ca/Mn/Fe metal lines replaced by Merz values), so the zinc engine is
not affected by it.

### Prep-tool findings that shaped the tier-0/1 receptor

Neither existing prep tool could produce an AD4-grade metal receptor:

- **obabel** types ions correctly *if the PDB element column is right*
  (`Mg`, `Fe`, `Cu`, `Na`, …) but is useless for a charged metal receptor
  [measured on openbabel-wheel 3.1.0, the obabel the CLI resolves on PATH]:
  default flags write **charge 0.000 on every atom**, and with
  `--partialcharge gasteiger` a molecule containing a metal ion converts to
  *nothing* ("0 molecules converted", exit 0, empty output). The adfr-suite
  obabel 2.4.1 computes Gasteiger charges unasked (~7200 nonzero) but the
  tier paths can't depend on which obabel is first on PATH. A zero-charged
  receptor is also fatal to AutoGrid directly: autogrid *refuses* it outright
  ("No partial atomic charges were found"). Final fix: the tier prep is
  **meeko on the protein only** (metal HETATMs stripped first — the same
  meeko prep the validated zinc path uses), then each metal ion is appended
  as a hand-built receptor line (charge col 70:76, type col 77:79, layout
  byte-verified against meeko's own Zn line) with the formal charge
  (`+2.000 Mg`, `+1.000 Na`, …; Fe assumed Fe2+).
- Case matters twice: obabel writes the acceptor nitrogen as `NA` and sodium
  as `Na` — the charge fixer matches **case-exactly**, or protein N's get
  +1 (measured on natest). And a broken test mutation (element shifted one
  column) also made obabel type Cu as `C` and Na+ as acceptor-`NA` plus a
  spurious hydrogen — the original garbage, since it reproduced only when the
  element column was misplaced; the corrected files type properly.
- **meeko** templates Zn ions (the zinc path's `+2.000 Zn` charge comes from
  its templates) but **fails template matching on any other ion** — which is
  why the tier prep filters the metal HETATMs out before meeko and splices
  them back in afterward (previous finding).
- The zinc GPF generator (`prepare_gpf4zn.py`) hardcodes
  `parameter_file=AD4Zn.dat` and appends the calibrated Zn/TZ `nbp_r_eps`
  lines unconditionally — so the tier paths use a plain in-repo GPF writer
  (`_build_ad4map_gpf`) reproducing exactly the structure autogrid+Vina
  accepted on the zinc path, with the metal's own site terms (below).

### Why the tier maps needed zinc-style `nbp_r_eps` site terms [measured]

**First: AutoGrid's 12-6 pair well is ε/6, not ε.** Verified with single-atom
autogrid probes, on both stock and `nbp_r_eps` pairs — every probe fit the
same law, so the earlier "−3.85 at 2.10 Å" reading of the zinc calibration was
wrong by exactly 6×:

- **Stock pairs** mix as Rmin_pair = **(Rii_a + Rii_b)/2** with well
  **√(εii_a·εii_b)/6** — stock Zn–OA probe: measured minimum −0.054 kcal/mol;
  predicted √(0.550·0.200)/6 = 0.0553 at 2.34 Å, and the full radial curve
  matches at every sampled distance. Stock C–OA probe: measured −0.0300 at
  3.60 Å = (4.00+3.20)/2; predicted √(0.150·0.200)/6 = 0.0289.
- **`nbp_r_eps` pairs** take R = the given r directly (wall and minimum land
  exactly there) with well exactly linear in the given ε: writing ε = 1/4/16
  gave depths −0.165/−0.664/−2.658 = ε/6, and R = 2.0/2.5 moved the minimum
  without changing the depth.

So the Scripps zinc numbers are calibrated **with** these semantics:
`nbp_r_eps OA Zn 2.1 3.8453` is a **−0.64** well, and the deep zinc binding
comes from the TZ terms — `nbp_r_eps NA TZ 0.25 23.2135` is a **−3.87 well per
donor** at the tetrahedral pseudo-atoms, plus the metal's electrostatics (the
+2 potential in the e-map reaches ≈ +11.4/r² per unit ligand charge, i.e.
≈ −1.4 per charged donor at 2 Å). The zinc engine must not be "corrected" —
it works as fit.

**Why the tier runs still failed with zinc's raw numbers.** The first tier
implementation gave the metals no site terms at all: all three test metals
(Mg, Cu, Na) scored ≈ −13.7 with **no coordination (~8 Å)** — with plain
atom_par wells of −0.05, donors can never reach the ion and the ligand docks
in a neighboring pocket. Copying the zinc `nbp_r_eps` numbers verbatim gave
effective wells of only −0.64 (OA) ≈ +0 (NA/N) — and since the tier engines
have **no TZ pseudo-atoms**, nothing else pulls: the rerun scored ≈ −6.9 with
poses still ~8.4 Å away, essentially ignoring the ion. A mid-diagnosis scare
that the ligand PDBQT was uncharged turned out to be a stale hand-artifact;
fresh `mk_prepare_ligand.py` output carries proper Gasteiger charges (the
ligand path was never broken).

Fix: `NBP_DONOR_BASE` now holds the **intended effective well depths** and
`_build_ad4map_gpf` writes ε = 6×the well (offsetting AutoGrid's measured ÷6),
each at the metal's crystal-chemical contact distance (`METAL_SITE_R` — Mg
2.07, Ca 2.35, Mn 2.19, Fe 2.10, Cu 2.15, Co 2.09, Ni 2.06, Cd 2.28, Hg 2.20,
Na 2.36, K 2.81), with S-donors +0.15 Å (zinc fit: OA 2.1/SA 2.25), NA/N
−0.10, HD excluded. Wells: OA −3.85 (the zinc-anchored intended donor pull),
SA −2.5, NA −1.5, N −2.0 — softness-scaled estimates pending the redocking
benchmark; the 12-6-4 C₄ term stays absorbed into the effective wells. The
metal's charge-dependence is *not* folded into the wells: it arrives via the
electrostatics map (Mg²⁺ doubles Na⁺'s potential), so monovalent ions get the
same wells as divalent — an earlier charge² ×0.25 scaling left Na⁺ with a
~−1.3 per-donor pull that could not beat a −6.9 neighboring pocket.

### Validation so far

Detection + receptor prep unit-checked for the corrected Cu/Na/Mg mutants of
the 1s63 receptor (element parsed, charge column +2/+1 written, case
distinction Na/NA held). Iterative validation, in order:

1. **No site terms:** all three metals ≈ −13.7 kcal/mol, no coordination
   (~8.1 Å) — the ion contributed nothing.
2. **Zinc's raw nbp numbers as site terms:** ≈ −6.9, still no coordination
   (~8.4 Å) — without TZ atoms the effective wells (ε/6 semantics, above)
   were ~60× too weak to compete with the neighboring pocket.
3. **Intended wells ×6 in the writer (current):** all three **coordinate**:

| metal | engine | best score | best-mode coordination |
|---|---|---|---|
| Mg²⁺ (tier 0) | ad4map | −10.27 | **1.96 Å** |
| Cu²⁺ (tier 1) | ad4merz | −10.17 | **2.02 Å** |
| Na⁺ (tier 1)  | ad4merz | −10.06 | **2.43 Å** |

vs crystal-chemical means Mg–O 2.07 / Cu–O 2.15 / Na–O 2.36 — each within
~0.1–0.2 Å. Tier scores are not comparable across engines or to the −12.1
zinc number; the coordination post-check remains the shared readout.