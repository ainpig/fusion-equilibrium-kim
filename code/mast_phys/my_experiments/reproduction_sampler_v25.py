"""Optional higher-precision sampling gradients for repeatable inverse solves.

Forward values stay on the original float32 CUDA path. Accumulating sampling
gradients in float64 reduces the sensitivity of the 900-step inverse solve to
CUDA atomic-add order.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

ORIGINAL = F.grid_sample


class Sampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, grid, mode, padding, align):
        ctx.save_for_backward(inputs, grid)
        ctx.mode, ctx.padding, ctx.align = mode, padding, align
        return ORIGINAL(inputs, grid, mode=mode, padding_mode=padding, align_corners=align)

    @staticmethod
    def backward(ctx, grad):
        inputs, grid = ctx.saved_tensors
        modes = {'bilinear':0, 'nearest':1, 'bicubic':2}
        pads = {'zeros':0, 'border':1, 'reflection':2}
        need = [ctx.needs_input_grad[0], ctx.needs_input_grad[1]]
        a, b = torch.ops.aten.grid_sampler_2d_backward(grad.double(), inputs.double(), grid.double(),
            modes[ctx.mode], pads[ctx.padding], ctx.align, need)
        return a.to(inputs.dtype) if need[0] else None, b.to(grid.dtype) if need[1] else None, None, None, None


def sampled(inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    if inputs.is_cuda and inputs.dtype == torch.float32 and torch.is_grad_enabled() and (inputs.requires_grad or grid.requires_grad):
        return Sampler.apply(inputs, grid, mode, padding_mode, bool(align_corners))
    return ORIGINAL(inputs, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


@contextmanager
def precision(enabled):
    if not enabled:
        yield
        return
    previous = F.grid_sample
    F.grid_sample = sampled
    try:
        yield
    finally:
        F.grid_sample = previous
