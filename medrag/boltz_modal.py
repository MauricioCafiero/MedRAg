"""Modal app: Boltz-2 protein-ligand cofolding + binding affinity on a GPU (the "modal" option next to running Boltz locally).

Built to be cheap: the weights are baked into the image at build time (no persistent volume, so no storage billing and no
re-download per run), one GPU call handles one input, a hard 10-minute timeout caps the cost of a single call, and the MSA is passed
in (computed on the caller's side) so the GPU never sits waiting on the alignment server.

usage: modal run medrag/boltz_modal.py --yaml-path input.yaml [--out-dir outputs/boltz_modal] [--gpu A10G]
The yaml is a normal Boltz-2 input; a `msa: ./file.csv` entry is uploaded next to it. Results (cif, confidence and affinity json,
npz arrays) are written to <out-dir>/<name>/.
"""
from pathlib import Path

import modal

app = modal.App("medrag-boltz")


def _bake_weights():
    from pathlib import Path
    from boltz.main import download_boltz2
    Path("/boltz_cache").mkdir(parents=True, exist_ok=True)
    download_boltz2(Path("/boltz_cache"))           # Boltz-2 structure + affinity weights and the CCD data


image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("boltz==2.2.1")
         .run_function(_bake_weights))


@app.function(image=image, gpu="A10G", timeout=600)      # 10 min x A10G is the worst case for one call
def cofold(yaml_text: str, files: dict[str, bytes], name: str) -> dict[str, bytes]:
    import subprocess
    import tempfile
    from pathlib import Path
    work = Path(tempfile.mkdtemp())
    (work / f"{name}.yaml").write_text(yaml_text)
    for rel, data in files.items():
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    cmd = ["boltz", "predict", str(work / f"{name}.yaml"), "--cache", "/boltz_cache", "--out_dir", str(work / "out"),
           "--accelerator", "gpu", "--num_workers", "2", "--no_kernels"]
    if "msa:" not in yaml_text:
        cmd.append("--use_msa_server")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)       # boltz resolves a relative `msa:` path from the cwd
    print(r.stdout[-1500:], r.stderr[-1500:])
    r.check_returncode()
    pred = work / "out" / f"boltz_results_{name}" / "predictions" / name
    out = {str(p.relative_to(pred)): p.read_bytes() for p in pred.rglob("*") if p.is_file()}
    if not out:                                                              # boltz exits 0 when it skips a bad input
        raise RuntimeError("boltz produced no predictions; see the log above")
    for p in (work / "out" / f"boltz_results_{name}" / "msa").glob("*_0.csv"):      # the alignment the server made, so the caller can reuse it
        out[f"_msa/{p.name}"] = p.read_bytes()
    return out


@app.local_entrypoint()
def main(yaml_path: str, out_dir: str = "outputs/boltz_modal", gpu: str = "A10G"):
    ypath = Path(yaml_path)
    text = ypath.read_text()
    files = {}
    for line in text.splitlines():
        if "msa:" in line:
            ref = line.split("msa:", 1)[1].strip().strip("'\"")
            files[ref] = (ypath.parent / ref).read_bytes()
    print(f"submitting {ypath.stem} on {gpu} ({len(files)} msa file(s)) ...")
    out = cofold.with_options(gpu=gpu).remote(text, files, ypath.stem)
    dest = Path(out_dir) / ypath.stem
    for rel, data in out.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(data)
    print(f"wrote {len(out)} files to {dest}")
