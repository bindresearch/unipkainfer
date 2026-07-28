# unipkainfer

`unipkainfer` is a standalone, inference-only packaging of Uni-pKa, Uni-Mol,
and the minimal pure-Python Uni-Core runtime they require. It supports CPU and
single-GPU inference through PyTorch and does not build Uni-Core's optional
fused CUDA extensions.

This distribution contains code derived from
[Uni-pKa](https://github.com/dptech-corp/Uni-pKa) (Apache-2.0), and code
derived from [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) and
[Uni-Core](https://github.com/dptech-corp/Uni-Core) (MIT). Copyright and
license notices are preserved in the accompanying license and
third-party notice files.

## Installation

```bash
python -m pip install unipkainfer
```

The selected model fold is downloaded from Hugging Face on first use and
verified by its SHA-256 checksum. The default is fold 1. Checkpoints are cached
in the user data directory, normally:

```text
~/.local/share/unipkainfer/models
```

The download can be prepared ahead of time, for example when building a
container:

```bash
unipka-download-model --fold 1
```

Use multiple values to prefetch several folds:

```bash
unipka-download-model --fold 0 1 2 3 4 --output-folder /path/to/models
```

## Python API

```python
from rdkit import Chem
from unipkainfer import predict_standard_free_energy

free_energy = predict_standard_free_energy(Chem.MolFromSmiles("CCO"))
```

Use `UnipkaFreeEnergyConfig(folds=...)` to select folds and `model_dir=...` to
override the cache location. A CUDA-enabled PyTorch installation and compatible
NVIDIA driver are required for GPU inference.

See `THIRD_PARTY_NOTICES.md`, `unipkainfer/unicoreinfer/UPSTREAM.md`, and the
included licenses for upstream provenance.
