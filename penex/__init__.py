from packaging.version import Version
import warnings

# numpy is a hard dependency in pyproject, so this import should succeed:
import numpy as _np

# Torch is optional; only check compatibility if it's present.
try:
    import torch as _torch
    _tv = Version(_torch.__version__.split("+")[0])  # drop local build tags
    _nv = Version(_np.__version__)
    if _nv.major >= 2 and _tv < Version("2.3"):
        raise ImportError(
            f"penex detected NumPy {_nv} with PyTorch {_tv}. "
            f"PyTorch < 2.3 is not compatible with NumPy 2.x. "
            f"Either downgrade NumPy: `pip install 'numpy<2'` "
            f"or upgrade PyTorch (>=2.3). "
            f"Alternatively install penex with NumPy 1.x defaults: `pip install penex`."
        )
except ModuleNotFoundError:
    # torch not installed; that's okay until user hits code that needs it
    pass