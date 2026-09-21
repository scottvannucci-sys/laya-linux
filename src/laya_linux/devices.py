"""Centralized device and dtype behavior (architecture §5.7, requirements §10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .errors import DeviceUnavailableError, DtypeUnsupportedError

SUPPORTED_DTYPES = ("auto", "float32", "float16", "bfloat16")


@dataclass(frozen=True)
class DevicePlan:
    device: torch.device
    backend: str  # "cuda" | "rocm" | "cpu"
    dtype: torch.dtype
    explanation: str

    def describe(self) -> dict[str, Any]:
        """Structured description suitable for `doctor` — no user-identifying paths."""
        return {
            "device": str(self.device),
            "backend": self.backend,
            "dtype": str(self.dtype).replace("torch.", ""),
            "explanation": self.explanation,
        }


def _is_rocm_build() -> bool:
    torch_version = getattr(torch, "version", None)
    return bool(getattr(torch_version, "hip", None))


def select_device(
    device: str | None,
    dtype: str = "auto",
    *,
    cpu_default_dtype: torch.dtype = torch.float32,
) -> DevicePlan:
    """Resolve an explicit or automatic device request into a concrete plan.

    - ``device="auto"`` selects an available accelerator, else CPU.
    - Explicit requests either use that device or fail with ``DEVICE_UNAVAILABLE``;
      they never silently move work elsewhere (requirements §10.2).
    """
    requested = device or "auto"
    if requested == "auto":
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            backend = "rocm" if _is_rocm_build() else "cuda"
            name = torch.cuda.get_device_name(0)
            plan_dtype = _resolve_dtype(dtype, torch.float16, allow_bfloat16=_cuda_bf16_ok())
            return DevicePlan(
                torch.device("cuda"),
                backend,
                plan_dtype,
                f"auto-selected cuda:0 ({name}; {count} device(s) visible; backend={backend})",
            )
        return DevicePlan(
            torch.device("cpu"),
            "cpu",
            cpu_default_dtype,
            "auto-selected cpu: no CUDA-compatible accelerator is available to PyTorch",
        )

    target = None
    try:
        target = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise DeviceUnavailableError(
            f"Unsupported device {requested!r}; supported values are 'auto', 'cpu', and 'cuda[:n]'."
        ) from exc
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                "CUDA was requested but this PyTorch build reports no available CUDA device "
                "(torch.cuda.is_available() is False). Verify the CUDA-enabled PyTorch build and driver."
            )
        backend = "rocm" if _is_rocm_build() else "cuda"
        plan_dtype = _resolve_dtype(dtype, torch.float16, allow_bfloat16=_cuda_bf16_ok())
        return DevicePlan(target, backend, plan_dtype, f"explicit request honored: {target} (backend={backend})")
    if target.type == "cpu":
        if dtype in ("float16", "bfloat16"):
            raise DtypeUnsupportedError(
                f"{dtype} is not supported on the CPU runtime in this version; "
                f"CPU inference requires float32 (see requirements §10.2)."
            )
        return DevicePlan(target, "cpu", cpu_default_dtype, "explicit request honored: cpu")
    raise DeviceUnavailableError(
        f"Unsupported device {requested!r}; supported values are 'auto', 'cpu', and 'cuda[:n]'."
    )


def _cuda_bf16_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return False


def _resolve_dtype(dtype: str, accelerator_default: torch.dtype, *, allow_bfloat16: bool) -> torch.dtype:
    if dtype == "auto":
        return accelerator_default
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        if not allow_bfloat16:
            raise DtypeUnsupportedError(
                "bfloat16 was requested but this device does not advertise BF16 support (compute capability < 8.0)."
            )
        return torch.bfloat16
    raise DtypeUnsupportedError(
        f"Unknown dtype {dtype!r}; supported values are 'auto', 'float32', 'float16', and 'bfloat16'."
    )
