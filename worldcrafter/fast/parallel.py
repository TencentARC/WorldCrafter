"""Query-sharded attention with replicated weights and unchanged key ordering."""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as collectives


def install_query_parallel():
    from ..diffusers import transformer
    from . import attention

    rank, size = dist.get_rank(), dist.get_world_size()
    dispatch = transformer.dispatch_attention_fn
    camera_attention = attention.flash_attention

    def gather(output, length, padded):
        if output.shape[1] < padded:
            padding = output.new_zeros((output.shape[0], padded - output.shape[1], *output.shape[2:]))
            output = torch.cat((output, padding), dim=1)
        full = collectives.wait_tensor(
            collectives.all_gather_tensor(output.contiguous(), 1, dist.group.WORLD)
        )
        return full[:, :length]

    def parallel_dispatch(query, key, value, **kwargs):
        length = query.shape[1]
        padded = (length + size - 1) // size
        start, end = rank * padded, min((rank + 1) * padded, length)
        if length < size:
            return dispatch(query, key, value, **kwargs)
        mask = kwargs.get("attn_mask")
        if mask is not None and mask.shape[-2] != 1:
            kwargs["attn_mask"] = mask[..., start:end, :]
        output = dispatch(query[:, start:end], key, value, **kwargs)
        return gather(output, length, padded)

    def parallel_camera(q, k, v, num_heads, compatibility_mode=False):
        # The original helper expects equal Q/K lengths, so reshape explicitly.
        import torch.nn.functional as F
        batch, length, channels = q.shape
        if length < size:
            return camera_attention(q, k, v, num_heads, compatibility_mode)
        padded = (length + size - 1) // size
        start, end = rank * padded, min((rank + 1) * padded, length)
        def heads(x):
            return x.reshape(batch, -1, num_heads, channels // num_heads).transpose(1, 2)
        output = F.scaled_dot_product_attention(heads(q[:, start:end]), heads(k), heads(v), is_causal=False)
        output = output.transpose(1, 2).reshape(batch, end - start, channels)
        return gather(output, length, padded)

    transformer.dispatch_attention_fn = parallel_dispatch
    attention.flash_attention = parallel_camera
