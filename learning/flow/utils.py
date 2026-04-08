import random

import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    del num_left_chunks
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode="trunc") + 1) * chunk_size
    return pos_idx.unsqueeze(0) < block_value.unsqueeze(1)


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
) -> torch.Tensor:
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1,)).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size, num_left_chunks, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    elif static_chunk_size > 0:
        chunk_masks = subsequent_chunk_mask(
            xs.size(1), static_chunk_size, num_decoding_left_chunks, xs.device
        )
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks

    if (chunk_masks.sum(dim=-1) == 0).any():
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    return (1.0 - mask.to(dtype)) * -1.0e10


def set_all_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
