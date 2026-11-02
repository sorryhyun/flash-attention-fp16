"""Runs an unpacked wheel on an sm_120 card: head dims 64 and 128 give finite gradients, 96 raises.

    python smoke.py <folder the wheel was unpacked to>
"""

import os
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

import flash_attn_2_cuda  # noqa: E402
from flash_attn import flash_attn_func  # noqa: E402

assert Path(flash_attn_2_cuda.__file__).parent == root, flash_attn_2_cuda.__file__
assert torch.cuda.get_device_capability(0)[0] == 12, "not an sm_120 card"


def grads(heads, head_dim):
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 512, heads, head_dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(3)
    )
    out = flash_attn_func(q, k, v)
    out.float().square().sum().backward()
    for t in (out, q.grad, k.grad, v.grad):
        assert t.isfinite().all(), "non-finite output or gradient"
    return torch.cat([t.grad.float().flatten() for t in (q, k, v)])


grads(16, 64)
print("head dim 64: finite")

os.environ["FLASH_ATTN_FP16ACC"] = "0"
stock = grads(24, 128)
os.environ["FLASH_ATTN_FP16ACC"] = "1"
fp16acc = grads(24, 128)
distance = ((fp16acc - stock).norm() / stock.norm()).item()
print(f"head dim 128: finite, fp16acc is {distance:.4f} from stock")
assert distance < 0.01

try:
    grads(16, 96)
except RuntimeError as e:
    print(f"head dim 96 raises: {e}")
else:
    raise AssertionError("head dim 96 ran")
print("ok")
