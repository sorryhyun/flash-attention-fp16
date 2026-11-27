"""Runs an unpacked wheel on an sm_120 card: head dims 64, 72 (the 96 kernel), 128 and 256 give
finite gradients, 72 and 256 (causal, GQA, varlen) match SDPA, the fp16acc forward is on by default
above 1024 keys, 160 raises.

    python smoke.py <folder the wheel was unpacked to>
"""

import os
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

import flash_attn_2_cuda  # noqa: E402
from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: E402

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

# PaddleOCR-VL's tower: 16 heads of 72, padded to the 96 kernel
torch.manual_seed(0)
q, k, v = (
    torch.randn(1, 729, 16, 72, device="cuda", dtype=torch.bfloat16).requires_grad_()
    for _ in range(3)
)
out = flash_attn_func(q, k, v)
out.float().square().sum().backward()
fa = torch.cat([t.flatten().float() for t in (out, q.grad, k.grad, v.grad)])
q2, k2, v2 = (t.detach().float().transpose(1, 2).requires_grad_() for t in (q, k, v))
ref = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)
ref.square().sum().backward()
ref = torch.cat(
    [t.transpose(1, 2).flatten() for t in (ref.detach(), q2.grad, k2.grad, v2.grad)]
)
assert fa.isfinite().all(), "non-finite output or gradient at head dim 72"
distance = ((fa - ref).norm() / ref.norm()).item()
print(f"head dim 72: finite, {distance:.4f} from fp32 SDPA")
assert distance < 0.02

# OvisOCR3's LM: 8 Q heads over 2 K/V heads of 256, causal, rows left-padded as two varlen
# segments (pad, then real); the real segments are checked against fp32 SDPA
torch.manual_seed(0)
S, pads = 640, (0, 200)
q = torch.randn(2 * S, 8, 256, device="cuda", dtype=torch.bfloat16).requires_grad_()
k, v = (
    torch.randn(2 * S, 2, 256, device="cuda", dtype=torch.bfloat16).requires_grad_()
    for _ in range(2)
)
cu = torch.tensor([0, pads[0], S, S + pads[1], 2 * S], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, k, v, cu, cu, S, S, causal=True)
real = torch.cat([torch.arange(i * S + p, (i + 1) * S) for i, p in enumerate(pads)]).cuda()
out[real].float().square().sum().backward()
fa = torch.cat([t[real].flatten().float() for t in (out, q.grad, k.grad, v.grad)])
refs = []
for i, p in enumerate(pads):
    rows = slice(i * S + p, (i + 1) * S)
    q2, k2, v2 = (t.detach()[rows].float().transpose(0, 1).requires_grad_() for t in (q, k, v))
    o2 = torch.nn.functional.scaled_dot_product_attention(
        q2, k2, v2, is_causal=True, enable_gqa=True
    )
    o2.square().sum().backward()
    refs.append([t.transpose(0, 1) for t in (o2.detach(), q2.grad, k2.grad, v2.grad)])
ref = torch.cat([torch.cat([r[j] for r in refs]).flatten() for j in range(4)])
assert fa.isfinite().all(), "non-finite output or gradient at head dim 256"
distance = ((fa - ref).norm() / ref.norm()).item()
print(f"head dim 256 (causal GQA varlen): finite, {distance:.4f} from fp32 SDPA")
assert distance < 0.02

os.environ["FLASH_ATTN_FP16ACC"] = "0"
stock = grads(24, 128)
os.environ["FLASH_ATTN_FP16ACC"] = "1"
fp16acc = grads(24, 128)
distance = ((fp16acc - stock).norm() / stock.norm()).item()
print(f"head dim 128: finite, fp16acc is {distance:.4f} from stock")
assert distance < 0.01


def fwd(lk):
    torch.manual_seed(0)
    q = torch.randn(1, 2048, 24, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(1, lk, 24, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    return flash_attn_func(q, k, v).float()


outs = {}
for flag in ("0", None, "all"):
    os.environ.pop("FLASH_ATTN_FP16ACC_FWD", None)
    if flag is not None:
        os.environ["FLASH_ATTN_FP16ACC_FWD"] = flag
    outs[flag] = (fwd(2048), fwd(512))
os.environ.pop("FLASH_ATTN_FP16ACC_FWD", None)
for flag in (None, "all"):
    long, short = outs[flag]
    assert long.isfinite().all(), "non-finite fp16acc forward"
    assert torch.equal(short, outs["0"][1]), "under 1024 keys the forward must stay bf16"
    assert not torch.equal(long, outs["0"][0]), f"FLASH_ATTN_FP16ACC_FWD={flag} took the bf16 forward"
    distance = ((long - outs["0"][0]).norm() / outs["0"][0].norm()).item()
    print(f"forward {flag or 'pv (default)'}: {distance:.4f} from bf16")
    assert distance < 0.01

try:
    grads(16, 160)
except RuntimeError as e:
    print(f"head dim 160 raises: {e}")
else:
    raise AssertionError("head dim 160 ran")
print("ok")
