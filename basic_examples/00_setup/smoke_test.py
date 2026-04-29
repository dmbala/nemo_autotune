"""Container smoke test. Confirms:
  * nemo_automodel imports
  * torch + CUDA work and a matmul runs
  * Automodel's distributed config dataclasses are importable (so Module 03's YAML keys resolve)
Run via: shared/launch.sh python 00_setup/smoke_test.py
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import sys


def _report(name: str) -> None:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"[FAIL] import {name}: {type(exc).__name__}: {exc}")
        return
    ver = getattr(mod, "__version__", "?")
    print(f"[ OK ] {name}: {ver}")


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '<unset>')}")

    for name in [
        "nemo_automodel",
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "huggingface_hub",
    ]:
        _report(name)

    import torch

    print(f"cuda available: {torch.cuda.is_available()}  devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device 0: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        print(f"bf16 matmul ok, |y|={float(y.abs().mean()):.4f}")
    print(f"torch.distributed available: {torch.distributed.is_available()}")

    # Parallelism knobs that the workshop configs reference.
    from nemo_automodel.components.distributed.config import FSDP2Config, DDPConfig

    print("FSDP2Config fields:", [f.name for f in dataclasses.fields(FSDP2Config)])
    print("DDPConfig fields:  ", [f.name for f in dataclasses.fields(DDPConfig)])

    # Surface which recipe the CLI dispatches to.
    from nemo_automodel._cli import app as cli_app

    print("CLI COMMAND_ALIASES:", cli_app.COMMAND_ALIASES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
