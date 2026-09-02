"""Print host hardware relevant to the 4GB-VRAM floor (non-negotiable #5).

REAL script (not a stub). Reports CPU, total RAM, PyTorch/CUDA status, and each
CUDA GPU's name and VRAM. Used on Phase 1 Day 1 (P3) to post the hardware
profile in the team chat and to decide Head C (needs VRAM >= 8GB, §2.2.5).

Run:  python scripts/check_hardware.py
Depends only on the standard library + torch (already in requirements.txt).
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys

VRAM_FLOOR_GB = 4.0
HEAD_C_VRAM_THRESHOLD_GB = 8.0
_BYTES_PER_GB = 1024 ** 3


def total_ram_gb() -> float | None:
    """Return total physical RAM in GB, or None if it cannot be determined.

    Uses only the standard library (no psutil): GlobalMemoryStatusEx on Windows,
    sysconf on POSIX.
    """
    try:
        if platform.system() == "Windows":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / _BYTES_PER_GB

        # POSIX
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / _BYTES_PER_GB
    except Exception:  # noqa: BLE001 - best-effort probe
        return None
    return None


def report() -> None:
    """Print the full hardware report to stdout."""
    print("=" * 60)
    print("HARDWARE CHECK  -  mgt-detect")
    print("=" * 60)

    # Platform / CPU
    print(f"Platform     : {platform.platform()}")
    print(f"Python       : {platform.python_version()} ({sys.executable})")
    print(f"Processor    : {platform.processor() or 'unknown'}")
    print(f"Logical CPUs : {os.cpu_count()}")

    ram = total_ram_gb()
    print(f"Total RAM    : {ram:.1f} GB" if ram is not None else "Total RAM    : unknown")

    # Torch / CUDA
    print("-" * 60)
    try:
        import torch
    except ImportError:
        print("PyTorch      : NOT INSTALLED  (pip install -r requirements.txt)")
        print("=" * 60)
        return

    print(f"PyTorch      : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA build   : {torch.version.cuda or 'cpu-only build'}")
    print(f"CUDA usable  : {cuda_available}")

    if not cuda_available:
        print("GPU          : none visible - running on CPU.")
        print(f"NOTE         : Head B/curvature will be slow on CPU; "
              f"VRAM floor is {VRAM_FLOOR_GB:.0f} GB (non-negotiable #5).")
        print("=" * 60)
        return

    n = torch.cuda.device_count()
    print(f"GPU count    : {n}")
    max_vram = 0.0
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / _BYTES_PER_GB
        max_vram = max(max_vram, vram_gb)
        print(f"  [{i}] {props.name}  |  VRAM {vram_gb:.1f} GB  |  CC {props.major}.{props.minor}")

    print("-" * 60)
    print(f"Max VRAM     : {max_vram:.1f} GB")
    print(f"VRAM floor   : {'OK' if max_vram >= VRAM_FLOOR_GB else 'BELOW FLOOR'} "
          f"(need >= {VRAM_FLOOR_GB:.0f} GB)")
    print(f"Head C       : {'eligible' if max_vram >= HEAD_C_VRAM_THRESHOLD_GB else 'SKIP'} "
          f"(needs >= {HEAD_C_VRAM_THRESHOLD_GB:.0f} GB, §2.2.5)")
    print("=" * 60)


if __name__ == "__main__":
    report()
