# flash-attention-fp16

FlashAttention 2.8.3 for RTX 50-series cards (sm_120), built for MOSS-Image and Anima:

- the 128×32 forward tile at head dim 128, in place of the A100 / H100 128×64 tile FA2 picks on cc 12;
- a backward with dP, dV, dK and dQ accumulated in fp16, for bf16 input;
- a forward with PV accumulated in fp16;
- `flash_attn_f16in_func`, which takes the caller's fp16 copies of Q, K and V in the forward and the
  backward, so under `torch.compile` the casts fuse into the kernel that wrote them;
- one wheel for every torch from 2.12 up, on torch's stable ABI.

Only sm_120 and head dims 64 and 128 are built (Anima's `llm_adapter` runs 64). Any other head dim
raises `this FA2 build holds head dims 64 and 128 only`. Dropout raises too.

## Install

The release `v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-f16in-cu132-torch2.12plus` has one `cp310-abi3`
wheel for Linux x86_64. The Windows (`win_amd64`) wheel is still the previous release's,
`v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.12plus`, without `f16in`. Both load on any torch
≥ 2.12 built for CUDA 13 and CPython ≥ 3.10. The ops are `torch.ops.flash_attn_2.*`.

```bash
gh release download v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-f16in-cu132-torch2.12plus \
  -R sorryhyun/flash-attention-fp16 -p '*linux_x86_64.whl'
uv pip install --no-deps --force-reinstall ./flash_attn-*.whl
```

`uv sync` puts the pinned wheel back.

## What the fp16 paths cover

| path | taken for | switch |
|---|---|---|
| fp16acc backward | bf16, head dim 128, no dropout, causal, window, alibi, softcap or `deterministic` | `FLASH_ATTN_FP16ACC=0` takes the stock backward |
| PV fp16acc forward | the same calls, with at least 1024 keys | `FLASH_ATTN_FP16ACC_FWD=0` takes the bf16 forward; `=all` also accumulates QKᵀ in fp16, forward and backward |
| fp16 input | `flash_attn_f16in_func`: head dim 128, as many K/V heads as Q heads, no causal or dropout; the PV forward at any key count | `flash_attn_f16in_takes` is false with `FLASH_ATTN_FP16ACC=0` or `FLASH_ATTN_FP16ACC_FWD=0` at import |

Every other call takes the stock path. S keeps fp32 accumulation in the backward. For the PV forward,
Q, K and V are cast to fp16, PV is accumulated in fp16 within each 32-key tile and added into the fp32
O, and O comes back in bf16. Under 1024 keys the casts cost more than the kernel saves on an RTX
5070 Ti; the threshold is that card's and is not re-tuned per device.

## Measured

RTX 5070 Ti, bf16, ms per call. Each table changes one thing.

fp16acc backward:

| | stock backward | fp16acc backward | |
|---|---|---|---|
| self, q `[1, 4131, 16, 128]` | 3.787 | 2.973 | 1.274x |
| cross, Lk 512 | 0.607 | 0.619 | 0.980x |
| self, q `[1, 4600, 24, 128]`, random input | 6.992 | 5.628 | 1.242x |

dQ, dK, dV sit at 0.74–0.97x the stock bf16 backward's median distance from fp32 on 48 captured
calls. Against the stock fp16 kernel on fp16-cast inputs with fp32 accumulation, the fp16acc gradients
are 4.4–5.5x further from fp32 (`grid.py`): fp16 storage of P and dS gains more over bf16 than fp16
accumulation loses. A MOSS-Image training step goes from 1071–1082 ms to 1022 ms at the same peak
memory.

128×32 forward tile, against FA2's 128×64:

| | forward | forward + backward |
|---|---|---|
| self, q `[1, 4131, 16, 128]` | 1.582 → 1.499 (1.056x) | 5.332 → 5.244 (1.017x) |
| cross, Lk 512 | 0.209 → 0.187 (1.118x) | 0.808 → 0.785 (1.029x) |
| self, q `[1, 4600, 24, 128]`, random input | 2.831 → 2.684 (1.055x) | 9.848 → 9.670 (1.018x) |

Median distance from fp32 on 48 captured calls: self 0.00170 → 0.00169, cross 0.00181 → 0.00175.

PV fp16acc forward:

| | bf16 forward | PV fp16acc | |
|---|---|---|---|
| self, q `[1, 4608, 24, 128]`, random input | 2.801 | 2.436 | 1.150x |
| self, q `[1, 4131, 16, 128]` | 1.550 | 1.327 | 1.168x |

O and dQ, dK, dV sit at 0.96–1.00x the bf16 forward's median distance from fp32 on 48 captured calls.
A MOSS-Image 1024² render (int8 base, 50 steps, CFG 3) goes from 10.6–10.9 s to 10.3 s.

fp16 input, against the same call on bf16 input:

| | bf16 input | `flash_attn_f16in_func` | |
|---|---|---|---|
| self, q `[1, 4600, 24, 128]`, backward | 5.49 | 5.19 | 1.058x |
| cross, q `[1, 4096, 16, 128]`, Lk 512, forward + backward | 0.823 | 0.770 | 1.069x |

The backward drops its three casts and `abs` + `amax` on dO (one kernel now, 44 µs). On the self
shape O, dQ, dK and dV are bitwise equal to the bf16 path. At 512 keys the call now takes the PV
forward, and O's distance from fp32 goes from 2.18e-3 to 1.69e-3. The op
`flash_attn::_flash_attn_forward_f16in` returns only O, the logsumexp and the RNG state: torch's
activation-budget partitioner prices a node at the sum of its outputs, and an op that also returned
its fp16 copies was recomputed in every block's backward. In a compiled training step the casts fuse
into RoPE: MOSS-Image 297.7 + 666.3 → 289.1 + 655.1 ms (forward + backward medians, same peak), Anima
392 → 380 ms.

The stable ABI costs nothing measurable on a MOSS-Image training step ([`notes.md`](notes.md)).

## Limits

These are the shapes and value ranges of MOSS-Image and Anima, both with QK-norm. Other models can
fall outside them:

- **Range.** Q, K and V are cast to fp16 with no check. An element past ±65504 gives a non-finite O,
  dQ and dK. MOSS peaks at |V| 304 and |Q|, |K| 77.
- **dS underflow.** dS is stored in fp16 with no gain. When dP is small (small V, or a head whose dO
  rows are far below its largest one) and attention is spread over many keys, dS falls into fp16's
  subnormals and dQ, dK lose precision. dV keeps it, since P carries a gain of 256. The stock fp16
  kernel does the same on fp16 input; the stock bf16 kernel does not.
- **dO scale per (batch, head).** dO is divided by its max over each (batch, head). One outlier row
  pushes the others toward underflow. A per-row scale cannot be undone in dK and dV, which sum over
  rows.

`grid.py` measures these against fp32 (H 4, head dim 128), as multiples of stock bf16's error:

| | dQ | dK | dV |
|---|---|---|---|
| L 1024–8192, any attention spread, V scale 1, plain dO | 0.75–0.80x | 0.75–0.81x | 0.71–0.73x |
| L 4096, uniform attention, V scale 0.01 | 6.7x | 6.7x | 0.72x |
| L 8192, uniform attention, V scale 1, one dO row ×1000 | 7.2x | 6.9x | 0.74x |
| L 8192, uniform attention, V scale 0.01, one dO row ×1000 | 38x | 48x | 0.74x |

The last row is a relative error of about 0.1. `FLASH_ATTN_FP16ACC=0` takes the stock backward.

## Build

```bash
./build.sh /path/to/.venv/bin/python   # torch ≥ 2.12, nvcc on PATH, CUDA_HOME set, ninja
```

`build.sh` applies, in order:

| patch | |
|---|---|
| `sm120.patch` | `is_sm8x` also true on cc 12 in `run_mha_fwd_hdim128`, for the 128×32 tile sm86 / 89 use; head dim 128 only; `-std=c++20` for torch's headers |
| `fp16acc.patch` | the fp16-accumulate backward |
| `hdim64.patch` | head dim 64 alongside 128 |
| `fwd_fp16acc.patch` | the PV fp16-accumulate forward |
| `stable_abi.patch` | torch's stable ABI (`TORCH_TARGET_VERSION` 2.12), ops under `torch.ops.flash_attn_2` |
| `f16in.patch` | fp16 Q, K, V with a bf16 `out` in `fwd` and with bf16 O / dO in `bwd`; the one-kernel dO max; `flash_attn_f16in_func` |

The Windows wheel adds `patches/windows.patch` (host C++20, `_USE_MATH_DEFINES`, an exported
`PyInit`) and is built with MSVC 2022 and `DISTUTILS_USE_SDK=1`.

`smoke.py` checks an unpacked wheel on the card. `grid.py` measures the installed wheel's error against
fp32 over sequence length, attention spread, V scale and a dO outlier row.

[`notes.md`](notes.md) has the ncu profile of the fp16acc backward and what was tried and dropped.

## Licence

FlashAttention is BSD-3-Clause, copyright its contributors; see `LICENSE` and `AUTHORS`.
