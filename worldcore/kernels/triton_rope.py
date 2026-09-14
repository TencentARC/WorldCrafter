import torch
import triton
import triton.language as tl

from .utils import calculate_settings, torch_gpu_device


# ------------------------------- replace funtion -------------------------------


def apply_rotary_emb_transposed_flash(x, freqs_cis):
    return Flash_RoPE_Transposed.apply(x, freqs_cis)


def replace_rope_with_flash_rope():
    from ..diffusers import transformer

    transformer.apply_rotary_emb_transposed = apply_rotary_emb_transposed_flash
    print("Patched Flash_RoPE globally\n")


# ------------------------------- layer norm -------------------------------


@triton.jit
def _apply_rope_transposed_kernel(
    X,
    Out,
    cos,
    sin,
    n_heads: tl.constexpr,
    stride_x: tl.constexpr,
    stride_out: tl.constexpr,
    stride_freq: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    freq_row_idx = row_idx // n_heads

    half_head_dim = head_dim // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim

    x_ptr = X + row_idx * stride_x
    out_ptr = Out + row_idx * stride_out
    cos_ptr = cos + freq_row_idx * stride_freq
    sin_ptr = sin + freq_row_idx * stride_freq

    x_real = tl.load(x_ptr + col_offsets * 2, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)
    cos_even = tl.load(cos_ptr + col_offsets * 2, mask=mask, other=0.0)
    sin_odd = tl.load(sin_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)

    out_even = x_real * cos_even - x_imag * sin_odd
    out_odd = x_real * sin_odd + x_imag * cos_even

    tl.store(out_ptr + col_offsets * 2, out_even, mask=mask)
    tl.store(out_ptr + col_offsets * 2 + 1, out_odd, mask=mask)


class Flash_RoPE_Transposed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        # x: [B, seq_len, n_heads, head_dim]
        # freqs_cis: [B, seq_len, head_dim*2]

        B, seq_len, n_heads, head_dim = x.shape

        x_flat = x.reshape(-1, head_dim).contiguous()
        device = x_flat.device
        out = torch.empty_like(x_flat)

        freqs_flat = freqs_cis.reshape(B * seq_len, -1).contiguous()
        half_dim = freqs_flat.shape[-1] // 2
        cos = freqs_flat[:, :half_dim].contiguous()  # [B*seq_len, head_dim]
        sin = freqs_flat[:, half_dim:].contiguous()  # [B*seq_len, head_dim]

        n_rows = x_flat.shape[0]  # B*seq_len*n_heads
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)

        with torch_gpu_device(device):
            _apply_rope_transposed_kernel[(n_rows,)](
                x_flat,
                out,
                cos,
                sin,
                n_heads,
                x_flat.stride(0),
                out.stride(0),
                cos.stride(0),
                head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        out = out.reshape(B, seq_len, n_heads, head_dim)

        ctx.save_for_backward(cos, sin)
        ctx.n_heads = n_heads
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim

        return out

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors

        B, seq_len, n_heads, head_dim = grad_output.shape
        grad_flat = grad_output.reshape(-1, head_dim).contiguous()
        device = grad_flat.device
        grad_x = torch.empty_like(grad_flat)

        sin_neg = -sin

        n_rows = grad_flat.shape[0]

        with torch_gpu_device(device):
            _apply_rope_transposed_kernel[(n_rows,)](
                grad_flat,
                grad_x,
                cos,
                sin_neg,
                ctx.n_heads,
                grad_flat.stride(0),
                grad_x.stride(0),
                cos.stride(0),
                ctx.head_dim,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        grad_x = grad_x.reshape(B, seq_len, n_heads, head_dim)
        return grad_x, None

