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
unipka-download-model
```

By default, model checkpoints are downloaded to the user data directory.
On Linux this is normally:

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

See `THIRD_PARTY_NOTICES.md`, `unipkainfer/unicoreinfer/UPSTREAM.md`, and the
included licenses for upstream provenance.
