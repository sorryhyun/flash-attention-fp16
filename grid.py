"""fp16acc vs stock FA2 against fp32: L x attention temperature x V scale x dO outlier.

Arms: stock bf16 (both flags 0), fp16acc (defaults), fp16-input stock (q/k/v half, dO / amax, fp32 acc).
Metric: relative L2 distance of dQ, dK, dV (and O) from an fp32 SDPA reference on the same bf16 inputs.
Ends with one |V| element past fp16's limit.

    python grid.py [result.json]
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

torch.backends.cuda.matmul.allow_tf32 = False
H, D = 4, 128


def rel(a, b):
    return ((a.float() - b).norm() / b.norm().clamp_min(1e-30)).item()


def ref32(q, k, v, do):
    q, k, v = (t.detach().float().transpose(1, 2).requires_grad_() for t in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        o = F.scaled_dot_product_attention(q, k, v)
    o.backward(do.float().transpose(1, 2))
    p_max = None
    with torch.no_grad():
        s = (q[:, :, :256] @ k.transpose(-1, -2)) / D**0.5
        p_max = s.softmax(-1).amax(-1).mean().item()
    return [o.detach().transpose(1, 2)] + [t.grad.transpose(1, 2) for t in (q, k, v)], p_max


def fa(q, k, v, do, arm):
    os.environ["FLASH_ATTN_FP16ACC"] = "0" if arm == "stock" else "1"
    os.environ["FLASH_ATTN_FP16ACC_FWD"] = "0" if arm == "stock" else "pv"
    if arm == "fp16in":
        s = do.float().abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-30)
        q, k, v = (t.detach().half().requires_grad_() for t in (q, k, v))
        o = flash_attn_func(q, k, v)
        o.backward((do.float() / s).half())
        return [o.detach()] + [t.grad.float() * s for t in (q, k, v)]
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o = flash_attn_func(q, k, v)
    o.backward(do)
    return [o.detach()] + [t.grad for t in (q, k, v)]


rows = []
for L in (1024, 4096, 8192):
    for temp in (0.05, 0.5, 1.0, 1.5):
        for vscale in (1.0, 0.01):
            for outlier in (False, True):
                g = torch.Generator(device="cuda").manual_seed(0)
                mk = lambda s: (torch.randn(1, L, H, D, device="cuda", generator=g) * s).bfloat16()  # noqa: E731
                q, k, v = mk(temp), mk(temp), mk(vscale)
                do = torch.randn(1, L, H, D, device="cuda", generator=g) * 1e-7
                if outlier:
                    do[:, 7] *= 1000.0
                do = do.bfloat16()
                ref, p_max = ref32(q, k, v, do)
                row = dict(L=L, temp=temp, vscale=vscale, outlier=outlier, p_max=p_max)
                for arm in ("stock", "fp16acc", "fp16in"):
                    out = fa(q, k, v, do, arm)
                    row[arm] = [rel(a, b) for a, b in zip(out, ref)]
                    row[arm + "_finite"] = all(t.isfinite().all().item() for t in out)
                rows.append(row)
                r = lambda a: "/".join(f"{x:.2e}" for x in row[a][1:])  # noqa: E731
                ratio = "/".join(f"{a / b:.2f}" for a, b in zip(row["fp16acc"][1:], row["stock"][1:]))
                ratio16 = "/".join(f"{a / b:.2f}" for a, b in zip(row["fp16acc"][1:], row["fp16in"][1:]))
                print(
                    f"L={L:5d} t={temp:4.2f} v={vscale:4.2f} out={int(outlier)} pmax={p_max:.1e} "
                    f"stock {r('stock')} | acc/stock {ratio} | acc/fp16in {ratio16} "
                    f"| O acc/stock {row['fp16acc'][0] / row['stock'][0]:.2f} "
                    f"fin={row['stock_finite']}{row['fp16acc_finite']}{row['fp16in_finite']}",
                    flush=True,
                )

# range: one |V| element past fp16's limit
q = torch.randn(1, 2048, H, D, device="cuda").bfloat16()
k = torch.randn_like(q)
v = torch.randn_like(q)
v[0, 5, 0, 3] = 1e5
do = (torch.randn_like(q.float()) * 1e-3).bfloat16()
for arm in ("stock", "fp16acc"):
    out = fa(q, k, v, do, arm)
    print(f"|V|=1e5 {arm}: finite O/dQ/dK/dV =", [t.isfinite().all().item() for t in out], flush=True)

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump(rows, f, indent=1)
