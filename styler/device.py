from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CudaStatus:
    available: bool
    message: str
    device_count: int = 0
    device_name: Optional[str] = None
    torch_cuda_version: Optional[str] = None


def get_cuda_status() -> CudaStatus:
    """
    Returns a human-friendly CUDA availability status.
    Works even on CPU-only builds.
    """
    try:
        is_avail = torch.cuda.is_available()
    except Exception as e:
        return CudaStatus(
            available=False,
            message=f"CUDA check failed: {e}",
            device_count=0,
            device_name=None,
            torch_cuda_version=getattr(torch.version, "cuda", None),
        )

    torch_cuda_ver = getattr(torch.version, "cuda", None)

    if not is_avail:
        msg = (
            "GPU (CUDA) недоступен. "
            "Причины: нет NVIDIA GPU, или установлен CPU-only PyTorch, или не установлены CUDA-драйверы/библиотеки."
        )
        return CudaStatus(
            available=False,
            message=msg,
            device_count=0,
            device_name=None,
            torch_cuda_version=torch_cuda_ver,
        )

    # CUDA available
    try:
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count > 0 else None
    except Exception:
        count = 0
        name = None

    msg = "CUDA доступен."
    if name:
        msg += f" Устройство: {name}"

    return CudaStatus(
        available=True,
        message=msg,
        device_count=count,
        device_name=name,
        torch_cuda_version=torch_cuda_ver,
    )


def resolve_device(requested: str) -> torch.device:
    """
    Strict device resolver: if user requested 'cuda' but it's unavailable -> raise RuntimeError.
    """
    req = (requested or "cpu").strip().lower()
    if req == "cuda":
        st = get_cuda_status()
        if not st.available:
            raise RuntimeError(st.message)
        return torch.device("cuda")
    return torch.device("cpu")
