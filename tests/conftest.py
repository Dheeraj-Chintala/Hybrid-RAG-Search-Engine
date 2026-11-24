import sys
import os
from pathlib import Path

# On Windows Python 3.8+, register faiss_cpu.libs directory for OpenMP DLL resolution
if sys.platform == "win32":
    for p in sys.path:
        libs_dir = Path(p) / "faiss_cpu.libs"
        if libs_dir.is_dir():
            try:
                os.add_dll_directory(str(libs_dir.resolve()))
            except Exception:
                pass

try:
    import torch  # noqa: F401
except Exception:
    pass
