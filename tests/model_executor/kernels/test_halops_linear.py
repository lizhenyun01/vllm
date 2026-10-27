# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HALOps MXFP8 linear backend: registration, gating, weight preparation and
the custom-op boundary.

These tests run on CPU and fake the `halops` package, so they do not need the
wheel from the PaddlePaddle index to be installed.
"""

import sys
import types

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_MXFP8_KERNELS,
    HALOpsMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp8 import halops as halops_mod
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    swizzle_mxfp8_scale,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def test_halops_kernel_is_registered_for_its_backend() -> None:
    assert HALOpsMxfp8LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["halops"]
    assert HALOpsMxfp8LinearKernel in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]


def test_halops_backend_selection(monkeypatch) -> None:
    """`--linear-backend halops` selects the HALOps kernel."""
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(
        linear_mod, "_get_linear_backend", lambda *, quantization: "halops"
    )
    monkeypatch.setitem(
        linear_mod._POSSIBLE_MXFP8_KERNELS,
        PlatformEnum.CUDA,
        [HALOpsMxfp8LinearKernel],
    )
    monkeypatch.setattr(
        HALOpsMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        HALOpsMxfp8LinearKernel,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )

    kernel = init_mxfp8_linear_kernel(weight_shape=(2048, 2048))

    assert isinstance(kernel, HALOpsMxfp8LinearKernel)
    assert kernel.config == Mxfp8LinearLayerConfig(weight_shape=(2048, 2048))


def _fake_platform(*, cuda: bool, blackwell: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        is_cuda=lambda: cuda,
        is_device_capability_family=lambda family: blackwell and family == 100,
    )


def test_halops_is_supported_requires_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        halops_mod, "current_platform", _fake_platform(cuda=False, blackwell=False)
    )

    assert HALOpsMxfp8LinearKernel.is_supported() == (
        False,
        "HALOps dense_gemm requires a CUDA device.",
    )


def test_halops_is_supported_requires_blackwell(monkeypatch) -> None:
    monkeypatch.setattr(
        halops_mod, "current_platform", _fake_platform(cuda=True, blackwell=False)
    )

    assert HALOpsMxfp8LinearKernel.is_supported() == (
        False,
        "HALOps dense_gemm requires Blackwell (SM100).",
    )


def test_halops_is_supported_reports_missing_package(monkeypatch) -> None:
    monkeypatch.setattr(
        halops_mod, "current_platform", _fake_platform(cuda=True, blackwell=True)
    )

    def _raise_import_error():
        raise ImportError("halops is not installed")

    monkeypatch.setattr(halops_mod, "_import_halops", _raise_import_error)

    assert HALOpsMxfp8LinearKernel.is_supported() == (
        False,
        "halops is not installed",
    )


def test_halops_is_supported_when_package_imports(monkeypatch) -> None:
    monkeypatch.setattr(
        halops_mod, "current_platform", _fake_platform(cuda=True, blackwell=True)
    )
    monkeypatch.setattr(halops_mod, "_import_halops", lambda: (object(), object()))

    assert HALOpsMxfp8LinearKernel.is_supported() == (True, None)


def test_halops_import_error_includes_install_hint(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "halops", None)

    with pytest.raises(ImportError, match=r'uv pip install "vllm\[halops\]" -f'):
        halops_mod._import_halops()


def test_halops_rejects_grouped_bmm() -> None:
    can_implement, reason = HALOpsMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig(weight_shape=(128, 128), bmm_batch_size=2)
    )

    assert not can_implement
    assert reason == "HALOps dense_gemm does not implement grouped BMM."


@pytest.mark.parametrize(
    ("weight_shape", "reason"),
    [
        ((128, 64), "HALOps requires K divisible by 128, got 64."),
        ((64, 128), "HALOps requires N divisible by 128, got 64."),
    ],
)
def test_halops_rejects_unaligned_shapes(weight_shape, reason) -> None:
    can_implement, actual_reason = HALOpsMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    )

    assert not can_implement
    assert actual_reason == reason


def test_halops_accepts_aligned_shape() -> None:
    assert HALOpsMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig(weight_shape=(128, 256))
    ) == (True, None)


def _make_layer(n: int, k: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    weight = torch.randn(n, 2 * k).to(torch.float8_e4m3fn)[:, ::2]
    assert not weight.is_contiguous()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // MXFP8_BLOCK_SIZE), dtype=torch.uint8),
        requires_grad=False,
    )
    return layer


def test_halops_process_weights_swizzles_scales_and_keeps_loader() -> None:
    n, k = 128, 256
    layer = _make_layer(n, k)
    expected_weight = layer.weight.detach().contiguous()
    expected_scale = swizzle_mxfp8_scale(
        layer.weight_scale.detach().clone(), M=n, K=k
    ).view(n, k // MXFP8_BLOCK_SIZE)
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale.weight_loader = scale_loader
    kernel = object.__new__(HALOpsMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.is_contiguous()
    assert torch.equal(
        layer.weight.view(torch.uint8), expected_weight.view(torch.uint8)
    )
    assert layer.weight_scale.shape == (n, k // MXFP8_BLOCK_SIZE)
    assert layer.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(layer.weight_scale, expected_scale)
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader
    assert layer.weight_block_size == [1, MXFP8_BLOCK_SIZE]


class _FakeHalops:
    """Records the calls the kernel makes into the `halops` package."""

    def __init__(self) -> None:
        self.quant_calls: list[tuple[torch.Tensor, str, bool]] = []
        self.gemm_calls: list[dict] = []

    def group_wise_quant(self, x: torch.Tensor, *, mode: str, use_swizzle: bool):
        self.quant_calls.append((x, mode, use_swizzle))
        m, k = x.shape
        m_sf = (m + 127) // 128 * 128
        return (
            torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device),
            torch.zeros((m_sf, k // 32), dtype=torch.uint8, device=x.device),
        )

    def dense_gemm(self, **kwargs) -> None:
        self.gemm_calls.append(kwargs)
        kwargs["output"].fill_(3.0)


@pytest.fixture
def fake_halops(monkeypatch) -> _FakeHalops:
    fake = _FakeHalops()
    monkeypatch.setattr(
        halops_mod, "_import_halops", lambda: (fake.dense_gemm, fake.group_wise_quant)
    )
    return fake


def _weights(n: int = 128, k: int = 64):
    return (
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.zeros((n, k // MXFP8_BLOCK_SIZE), dtype=torch.uint8),
    )


def test_halops_linear_allocates_real_m(fake_halops) -> None:
    """The output keeps the real M; only the SfAtom scales are padded."""
    weight, weight_scale = _weights()
    x = torch.randn((2, 3, 64), dtype=torch.bfloat16)
    bias = torch.ones(128, dtype=torch.bfloat16)

    out = halops_mod._halops_mxfp8_linear(x, weight, weight_scale, bias)

    assert len(fake_halops.quant_calls) == 1
    assert fake_halops.gemm_calls[0]["output"].shape == (6, 128)
    (gemm,) = fake_halops.gemm_calls
    assert gemm["bias"] is bias
    assert gemm["sfb"] is weight_scale
    assert gemm["alpha"] is None
    assert gemm["sm_count"] == -1
    assert gemm["mode"] == "w8a8-mxfp8"
    assert out.shape == (2, 3, 128)
    assert out.dtype is torch.bfloat16
    torch.testing.assert_close(out, torch.full_like(out, 3.0))


def test_halops_linear_without_bias_passes_none(fake_halops) -> None:
    weight, weight_scale = _weights()

    halops_mod._halops_mxfp8_linear(
        torch.randn((1, 64), dtype=torch.bfloat16), weight, weight_scale, None
    )

    assert fake_halops.gemm_calls[0]["bias"] is None


def test_halops_linear_zero_tokens_skips_the_kernel(fake_halops) -> None:
    weight, weight_scale = _weights()

    out = halops_mod._halops_mxfp8_linear(
        torch.empty((0, 64), dtype=torch.bfloat16), weight, weight_scale, None
    )

    assert out.shape == (0, 128)
    assert fake_halops.quant_calls == []
    assert fake_halops.gemm_calls == []


def test_halops_linear_quantizes_contiguous_bf16(fake_halops) -> None:
    weight, weight_scale = _weights()
    storage = torch.randn((4, 128), dtype=torch.bfloat16)
    x = storage[:, ::2]
    assert not x.is_contiguous()

    out = halops_mod._halops_mxfp8_linear(x, weight, weight_scale, None)

    q_input, mode, use_swizzle = fake_halops.quant_calls[0]
    assert q_input.is_contiguous()
    assert torch.equal(q_input, x)
    assert mode == "w8a8-mxfp8"
    assert use_swizzle
    assert out.dtype is torch.bfloat16


def test_halops_linear_upcasts_fp32_activations(fake_halops) -> None:
    weight, weight_scale = _weights()
    x = torch.randn((4, 64), dtype=torch.float32)

    out = halops_mod._halops_mxfp8_linear(x, weight, weight_scale, None)

    q_input = fake_halops.quant_calls[0][0]
    assert q_input.dtype is torch.bfloat16
    assert q_input.shape == (4, 64)
    assert out.dtype is torch.float32


def test_halops_custom_op_fake_matches_the_real_shape() -> None:
    weight, weight_scale = _weights()
    x = torch.empty((2, 3, 64), dtype=torch.bfloat16)

    out = halops_mod._halops_mxfp8_linear_fake(x, weight, weight_scale, None)

    assert out.shape == (2, 3, 128)
    assert out.dtype is torch.bfloat16
    assert out.device == x.device


def test_halops_custom_op_is_registered_with_a_fake_impl() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    schema = str(torch.ops.vllm.halops_mxfp8_linear.default._schema)
    assert "halops_mxfp8_linear" in schema
    assert "Tensor? bias" in schema

    with FakeTensorMode():
        weight, weight_scale = _weights()
        x = torch.empty((2, 3, 64), dtype=torch.bfloat16)
        out = torch.ops.vllm.halops_mxfp8_linear(x, weight, weight_scale, None)

    assert out.shape == (2, 3, 128)
    assert out.dtype is torch.bfloat16
