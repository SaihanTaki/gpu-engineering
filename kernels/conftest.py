import os
import shutil

# Without an NVIDIA GPU, run Numba kernels on its CUDA simulator and Triton
# kernels in its interpreter. Both read these variables at import time, and
# pytest loads this file before any test module imports numba or triton.
if shutil.which("nvidia-smi") is None:
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
    os.environ.setdefault("TRITON_INTERPRET", "1")
