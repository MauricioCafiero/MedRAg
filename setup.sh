#!/usr/bin/env bash
# medrag environment: uv venv (py3.12), CPU-only torch 2.8.0, fairchem (UMA) + mace + ase + rdkit
# + LLM clients. Mirrors cafquantum/setup.sh (same pins, same --no-deps trick for the e3nn conflict).
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 --allow-existing .venv
PY=.venv/bin/python

# CPU-only torch (the default PyPI wheel drags in ~3 GB of CUDA libs this machine can't use)
uv pip install --python $PY "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu

uv pip install --python $PY \
    "numpy>=2.0,<2.5" "e3nn==0.4.4" "ase==3.29.0" rdkit scipy pandas scikit-learn \
    opt_einsum torch-ema prettytable matscipy h5py torchmetrics python-hostlist configargparse \
    GitPython pyYAML tqdm lmdb orjson "ase-db-backends>=0.10.0" "clusterscope==0.0.18" \
    "huggingface-hub>=0.27.1" hydra-core "monty>=2026.2.18" "numba>=0.62.0" requests \
    "setuptools<81.0.0" submitit torchtnt websockets "ray[serve]>=2.53.0" wandb \
    ollama anthropic rich psutil pytest \
    pubchempy openbabel-wheel dockstring meeko transformers lightgbm openmm pdbfixer "graph-longrange==0.4.0" \
    pymol-open-source chembl_webresource_client rcsb-api dimorphite-dl pdb2pqr

uv pip install --python $PY --no-deps "fairchem-core==2.21.0" "mace-torch==0.3.16"

# generate/filter extras. admet-ai would downgrade rdkit (2026.9.1 -> 2026.3.6) unless it is held with a constraint;
# py3Dmol is imported by GenMask; oddt (bioisostere_assist shape descriptors) is built from the copy of its tarball in medrag/lib/bioisostere_assist without
# isolation (it needs six). CafChemShape (borrowed from bioisostere_assist) blocks the incompatible openbabel bindings itself.
echo "rdkit==$($PY -c 'import rdkit; print(rdkit.__version__)')" > /tmp/medrag_constraints.txt
uv pip install --python $PY -c /tmp/medrag_constraints.txt admet-ai py3Dmol six setuptools wheel
uv pip install --python $PY -c /tmp/medrag_constraints.txt --no-build-isolation medrag/lib/bioisostere_assist/oddt-0.7.tar.gz
# GenMask model (433 MB) is fetched on first use, or ahead of time with:
#   $PY -c "from huggingface_hub import snapshot_download; snapshot_download('cafierom/bert-base-cased-ChemTok-ZN250K-V1')"

echo "--- import probe ---"
$PY -c "
import torch, ase, rdkit, ollama, anthropic, psutil
from torch.serialization import add_safe_globals; add_safe_globals([slice])
import fairchem.core, mace.calculators
print('torch', torch.__version__, '| ase', ase.__version__, '| fairchem+mace OK')"
