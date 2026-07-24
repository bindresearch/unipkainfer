# unipkainfer

`unipkainfer` is a standalone, inference-only packaging of Uni-pKa, Uni-Mol,
and the minimal pure-Python Uni-Core runtime they require. It supports CPU and
single-GPU inference through PyTorch and does not build Uni-Core's optional
fused CUDA extensions.

## Installation

```bash
python -m pip install unipkainfer
unipka-download-model
```

By default, model checkpoints are downloaded to the operating system's user
data directory. On Linux this is normally:

```text
~/.local/share/unipkainfer/models
```

Choose another location when needed:

```bash
unipka-download-model --output-folder /path/to/models
```

## Python API

```python
from rdkit import Chem
from unipkainfer import predict_standard_free_energy

free_energy = predict_standard_free_energy(Chem.MolFromSmiles("CCO"))
```

Use `UnipkaFreeEnergyConfig(model_dir=...)` to override the default model
location. A CUDA-enabled PyTorch installation and compatible NVIDIA driver are
required for GPU inference.

See `unipkainfer/unicoreinfer/UPSTREAM.md` and the included licenses for
upstream provenance.
