"""Step 4A.0.G — Capture environment versions."""
import json
import platform
import subprocess

def safe_version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"IMPORT_FAILED: {e}"

import torch
import neuralop
import numpy as np
import scipy

info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "neuralop": safe_version("neuralop"),
    "numpy": safe_version("numpy"),
    "scipy": safe_version("scipy"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_device_vram_MB": (torch.cuda.get_device_properties(0).total_memory / 1e6
                            if torch.cuda.is_available() else None),
}
try:
    info["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
         "--format=csv,noheader"], text=True
    ).strip()
except Exception as e:
    info["nvidia_smi"] = f"FAILED: {e}"

with open("results/phase_4/phase_4a/preflight/environment.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
print("\n=== 4A.0.G Environment: captured ===")
