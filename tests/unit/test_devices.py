"""Device and dtype selection (unit, CPU-only assertions)."""

import pytest
import torch

from laya_linux.devices import select_device
from laya_linux.errors import DeviceUnavailableError, DtypeUnsupportedError


def test_auto_selects_cpu_without_cuda():
    plan = select_device("auto")
    if not torch.cuda.is_available():
        assert plan.device.type == "cpu"
        assert plan.backend == "cpu"
        assert plan.dtype == torch.float32  # CPU defaults to FP32


def test_explicit_cpu():
    plan = select_device("cpu")
    assert plan.device.type == "cpu"
    assert plan.dtype == torch.float32


def test_explicit_unavailable_fails_never_falls_back():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; unavailable-fallback path unreachable")
    with pytest.raises(DeviceUnavailableError):
        select_device("cuda:0")


def test_cpu_bf16_rejected():
    with pytest.raises(DtypeUnsupportedError):
        select_device("cpu", "bfloat16")


def test_unknown_device_rejected():
    with pytest.raises(DeviceUnavailableError):
        select_device("量子 accelerator")


def test_explanation_is_safe():
    plan = select_device("cpu")
    assert plan.explanation
    assert "scott" not in plan.explanation.lower()


def test_describe_shape():
    d = select_device("cpu").describe()
    assert set(d) == {"device", "backend", "dtype", "explanation"}


def test_checkpoint_amp_dtype_hint_normalized():
    """The checkpoint's amp_dtype spelling ("bf16") drives the auto GPU default."""
    from laya_linux.devices import _capability_default

    assert _capability_default("bf16", allow_bfloat16=True) is torch.bfloat16
    assert _capability_default("bfloat16", allow_bfloat16=True) is torch.bfloat16
    assert _capability_default("bf16", allow_bfloat16=False) is torch.float16
    assert _capability_default("float16", allow_bfloat16=True) is torch.float16
    assert _capability_default(None, allow_bfloat16=True) is torch.float16
    assert _capability_default("garbage", allow_bfloat16=True) is torch.float16
