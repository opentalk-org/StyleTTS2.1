import torch
import triton
import triton.language as tl


@triton.jit
def _maximum_path_forward_kernel(
    scores,
    text_lengths,
    mel_lengths,
    directions,
    workspace,
    score_stride_b,
    score_stride_t,
    score_stride_m,
    direction_stride_b,
    direction_stride_t,
    direction_stride_m,
    workspace_stride_b,
    max_mel_steps,
    block_text: tl.constexpr,
):
    batch = tl.program_id(0)
    text_length = tl.load(text_lengths + batch)
    mel_length = tl.load(mel_lengths + batch)
    text_offsets = tl.arange(0, block_text)
    negative_infinity = -float("inf")
    previous = tl.full((block_text,), negative_infinity, tl.float32)
    previous = tl.where(text_offsets == 0, 0.0, previous)
    tl.store(
        workspace + batch * workspace_stride_b + text_offsets,
        previous,
        mask=text_offsets < text_length,
    )
    tl.debug_barrier()

    for mel_index in range(0, max_mel_steps):
        valid_mel = mel_index < mel_length
        valid_text = (
            (text_offsets < text_length)
            & (text_offsets <= mel_index)
            & (text_offsets >= text_length + mel_index - mel_length)
        )
        current_predecessor = tl.load(
            workspace + batch * workspace_stride_b + text_offsets,
            mask=text_offsets < text_length,
            other=negative_infinity,
        )
        previous_predecessor = tl.load(
            workspace + batch * workspace_stride_b + text_offsets - 1,
            mask=(text_offsets > 0) & (text_offsets < text_length),
            other=negative_infinity,
        )
        moved = previous_predecessor >= current_predecessor
        best = tl.maximum(current_predecessor, previous_predecessor)
        score = tl.load(
            scores
            + batch * score_stride_b
            + text_offsets * score_stride_t
            + mel_index * score_stride_m,
            mask=valid_mel & valid_text,
            other=negative_infinity,
        ).to(tl.float32)
        current = score + best
        current = tl.where(valid_mel & valid_text, current, negative_infinity)
        tl.store(
            directions
            + batch * direction_stride_b
            + text_offsets * direction_stride_t
            + mel_index * direction_stride_m,
            moved,
            mask=valid_mel & valid_text,
        )
        tl.store(
            workspace + batch * workspace_stride_b + text_offsets,
            current,
            mask=text_offsets < text_length,
        )
        tl.debug_barrier()


@triton.jit
def _maximum_path_backward_kernel(
    directions,
    text_lengths,
    mel_lengths,
    path,
    direction_stride_b,
    direction_stride_t,
    direction_stride_m,
    path_stride_b,
    path_stride_t,
    path_stride_m,
    max_mel_steps,
):
    batch = tl.program_id(0)
    text_index = tl.load(text_lengths + batch) - 1
    mel_length = tl.load(mel_lengths + batch)

    for reverse_index in range(0, max_mel_steps):
        mel_index = max_mel_steps - reverse_index - 1
        valid = mel_index < mel_length
        tl.store(
            path
            + batch * path_stride_b
            + text_index * path_stride_t
            + mel_index * path_stride_m,
            1,
            mask=valid,
        )
        moved = tl.load(
            directions
            + batch * direction_stride_b
            + text_index * direction_stride_t
            + mel_index * direction_stride_m,
            mask=valid,
            other=0,
        )
        text_index -= moved.to(tl.int32)


def maximum_path(
    scores: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_text_steps, max_mel_steps = scores.shape
    directions = torch.empty(
        scores.shape,
        dtype=torch.uint8,
        device=scores.device,
    )
    workspace = torch.empty(
        (batch_size, max_text_steps),
        dtype=torch.float32,
        device=scores.device,
    )
    path = torch.zeros_like(scores)
    block_text = triton.next_power_of_2(max_text_steps)
    _maximum_path_forward_kernel[(batch_size,)](
        scores,
        text_lengths,
        mel_lengths,
        directions,
        workspace,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        directions.stride(0),
        directions.stride(1),
        directions.stride(2),
        workspace.stride(0),
        max_mel_steps,
        block_text,
        num_warps=8 if block_text >= 256 else 4,
    )
    _maximum_path_backward_kernel[(batch_size,)](
        directions,
        text_lengths,
        mel_lengths,
        path,
        directions.stride(0),
        directions.stride(1),
        directions.stride(2),
        path.stride(0),
        path.stride(1),
        path.stride(2),
        max_mel_steps,
    )
    return path
