import torch
import torch.nn as nn

from ._ops import ops


def _supports_strided_rows(x: torch.Tensor) -> bool:
    """Last dimension is contiguous and leading dims collapse to one row stride.

    This is the layout of ``tensor.chunk(..., dim=-1)``. Fully arbitrary
    strides (a transposed last dimension, or a gap that is not one row stride)
    return False.
    """
    if x.ndim < 1:
        return False
    if x.size(-1) > 1 and x.stride(-1) != 1:
        return False
    if x.ndim == 1:
        return True
    inner = 1
    row_stride = None
    for i in range(x.ndim - 2, -1, -1):
        size = x.size(i)
        if size == 1:
            continue
        stride = x.stride(i)
        if stride < 0:
            return False
        if row_stride is None:
            row_stride = stride
            inner = size
            continue
        if inner != 0 and row_stride > (2**63 - 1) // inner:
            return False
        if stride != row_stride * inner:
            return False
        if size > (2**63 - 1) // inner:
            return False
        inner *= size
    # Metal indexes the row with a uint32 stride. CUDA uses int64.
    if (
        x.device.type == "mps"
        and row_stride is not None
        and row_stride > 0xFFFFFFFF
    ):
        return False
    return True


def _prepare_input(x: torch.Tensor) -> torch.Tensor:
    """Pass through layouts the CUDA and Metal kernels index themselves.

    Contiguous tensors stay as they are. On CUDA (including ROCm) and MPS, a
    contiguous last dimension with a single row stride is also passed through.
    Every other layout, and every other backend, is copied so the kernel sees
    a contiguous tensor.
    """
    if x.is_contiguous():
        return x
    if x.device.type in ("cuda", "mps") and _supports_strided_rows(x):
        return x
    return x.contiguous()


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out

class Silu(nn.Module):
    """An activation function for SiLU.

    The function computes x -> silu(x).

    Shapes:
        x: (num_tokens, d) or (batch_size, seq_len, d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.silu(out, x)
        return out

class Gelu(nn.Module):
    """An activation function for GELU.

    The function computes x -> gelu(x).

    Shapes:
        x: (num_tokens, d) or (batch_size, seq_len, d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.gelu(out, x)
        return out

class GeluTanh(nn.Module):
    """An activation function for GELU with `tanh` approximation.

    The function computes x -> gelu_tanh(x).

    Shapes:
        x: (num_tokens, d) or (batch_size, seq_len, d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.gelu_tanh(out, x)
        return out


class MulAndSilu(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> x[:d] * silu(x[d:]) where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x)
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.mul_and_silu(out, x)
        return out


class GeluAndMul(nn.Module):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_and_mul(out, x)
        return out


class GeluTanhAndMul(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.gelu_tanh_and_mul(out, x)
        return out


class FatreluAndMul(nn.Module):
    """An activation function for FATReLU.

    The function computes x -> FATReLU(x[:d]) * x[d:] where
    d = x.shape[-1] // 2.
    This is used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor):
        x = _prepare_input(x)
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.fatrelu_and_mul(out, x, self.threshold)
        return out


class FastGELU(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.gelu_fast(out, x)
        return out


class NewGELU(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.gelu_new(out, x)
        return out


class QuickGELU(nn.Module):
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _prepare_input(x)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.gelu_quick(out, x)
        return out
