import os
import sys
from pathlib import Path

QUEST_CODE = str(Path(__file__).resolve().parent.parent)   # this repo

MODEL = "quest-semantic"
DEVICE = "cuda"

import warnings
# matplotlib prints an Axes3D import warning on this box; nothing here is 3D
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

if QUEST_CODE not in sys.path:
    sys.path.insert(0, QUEST_CODE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats("retina")
except ImportError:                     
    pass
