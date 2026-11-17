"""Environment probe run by the GUI in a separate process.

Prints one JSON line describing torch/CUDA/WhisperX/ffmpeg. Loads no models
and never fails just because a GPU is missing.
"""

import importlib.metadata
import json
import os
import shutil
import sys


def main() -> int:
    info = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": None, "cuda_build": None, "cuda_available": False,
        "gpu": None, "vram_gb": None, "whisperx": None,
        "ffmpeg": shutil.which("ffmpeg"), "error": None,
    }
    try:
        info["whisperx"] = importlib.metadata.version("whisperx")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        # Same DLL lookup as transcript.setup_cuda_dlls, without importing rich.
        if hasattr(os, "add_dll_directory"):
            for path in sys.path:
                for library in ("cublas", "cudnn"):
                    directory = os.path.join(path, "nvidia", library, "bin")
                    if os.path.isdir(directory):
                        try:
                            os.add_dll_directory(directory)
                        except OSError:
                            pass
        import torch

        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(total / 1024 ** 3, 1)
    except Exception as error:  # diagnostic only
        info["error"] = f"{type(error).__name__}: {error}"
    print("@@PROBE " + json.dumps(info, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
