# flash-attention-fp16

FlashAttention 2.8.3 built for RTX 50-series cards (sm_120), with the 128×32 forward tile.

On compute capability 12, FA2's `run_mha_fwd_hdim128` falls to the A100 / H100 tile, 128×64, and
exposes no option to pick another. `patches/sm120.patch` sends cc 12 to the tile sm86 / 89 use, and
limits the build to what these wheels are for:

- `is_sm8x` also true on `cc_major == 12`, in `run_mha_fwd_hdim128` only.
- sm_120 only, head dims 64 and 128 (MOSS-Image and Anima, whose `llm_adapter` runs 64). Any other
  head dim raises `this FA2 build holds head dims 64 and 128 only`.
- `-std=c++20`, which torch 2.14's headers need.

`patches/fp16acc.patch` adds a backward with dP, dV, dK and dQ accumulated in fp16, for bf16 input at
head dim 128 with no dropout, causal, window, alibi, softcap or `deterministic`. Other calls take the
stock path, and `FLASH_ATTN_FP16ACC=0` forces it. S keeps fp32 accumulation.

`patches/fwd_fp16acc.patch` accumulates the forward's PV in fp16 inside each 32-key tile, added into
the fp32 O, for the same bf16, head dim 128 calls with at least 1024 keys. Q, K and V are cast to fp16
for it; O comes back in bf16. It is on by default; `FLASH_ATTN_FP16ACC_FWD=0` takes the bf16 forward,
and `FLASH_ATTN_FP16ACC_FWD=all` accumulates QKᵀ in fp16 as well, in the forward and the backward.
Under 1024 keys the casts cost more than the kernel saves on an RTX 5070 Ti, so those calls keep the
bf16 forward. The threshold is that card's and is not re-tuned per device.

`patches/stable_abi.patch` builds on torch's stable ABI (`TORCH_TARGET_VERSION` 2.12): one
`cp310-abi3` wheel loads on every torch from 2.12 up, for the CUDA major it was built with. The
ops are `torch.ops.flash_attn_2.*`. Dropout raises, since the stable ABI has no generator.

`patches/precast.patch` hands the fp16acc forward's fp16 copies of Q, K and V back to the caller
(`fwd`'s last three outputs, empty when the bf16 forward ran) and lets `bwd` take fp16 Q, K, V with
bf16 O and dO, so `flash_attn_func`'s backward skips its three casts and gets the same bytes. The
gradients come back in O's dtype. It also replaces `amax(abs(dO))` with one reduction kernel. The
kernels are unchanged; dK and dV are bitwise those of the self-cast path.

## Measured

RTX 5070 Ti, bf16, ms per call, against the wheel's 128×64:

| | forward | forward + backward |
|---|---|---|
| self, q `[1, 4131, 16, 128]` | 1.582 → 1.499 (1.056x) | 5.332 → 5.244 (1.017x) |
| cross, Lk 512 | 0.209 → 0.187 (1.118x) | 0.808 → 0.785 (1.029x) |
| self, q `[1, 4600, 24, 128]`, random input | 2.831 → 2.684 (1.055x) | 9.848 → 9.670 (1.018x) |

Median distance from fp32 on 48 captured calls: self 0.00170 → 0.00169, cross 0.00181 → 0.00175.

With `fp16acc.patch`, same card:

| | stock backward | fp16acc backward | backward |
|---|---|---|---|
| self, q `[1, 4131, 16, 128]` | 3.787 | 2.973 | 1.274x |
| cross, Lk 512 | 0.607 | 0.619 | 0.980x |
| self, q `[1, 4600, 24, 128]`, random input | 6.992 | 5.628 | 1.242x |

dQ, dK, dV sit at 0.74–0.97x stock's median distance from fp32 on 48 captured calls.
The baseline is the stock bf16 backward, the path these calls would otherwise take. Against
the stock fp16 kernel on fp16-cast inputs with fp32 accumulation, the fp16acc gradients are 4.4–5.5x
further from fp32 (`grid.py`): fp16 storage of P and dS gains more over bf16 than fp16 accumulation
loses. A real MOSS-Image training step goes from 1071–1082 ms to 1022 ms at the same peak memory.

With `fwd_fp16acc.patch`, same card, forward ms per call:

| | bf16 forward | PV fp16acc | |
|---|---|---|---|
| self, q `[1, 4608, 24, 128]`, random input | 2.801 | 2.436 | 1.150x |
| self, q `[1, 4131, 16, 128]` | 1.550 | 1.327 | 1.168x |

O and dQ, dK, dV sit at 0.96–1.00x the bf16 forward's median distance from fp32 on 48 captured calls.
A MOSS-Image 1024² render (int8 base, 50 steps, CFG 3) goes from 10.6–10.9 s to 10.3 s.

With `precast.patch`, same card, the MOSS self shape `[1, 4600, 24, 128]`, forward + backward per call
(torch profiler, boost clocks): 7.95 → 7.73 ms; the backward's three `to(half)` (3 × 81 µs) are gone
and `abs` + `amax` (48 + 50 µs) became a 68 µs kernel.

Under ncu with locked clocks the fp16acc backward kernel runs the tensor pipe 74% of the time against
91% for the stock bf16 kernel (5.81 against 8.23 ms). HMMA on the fp16-accumulate path is 2x, but the
per-tile drain of the fp16 partials into fp32 (`cvt` + `fadd`, 3.6 G each on this shape) raises FMA-pipe
instructions 5.5x and is not hidden with two warps per scheduler. Draining dK and dV every second tile
(fp16 partials kept across tiles) cut those instructions 20% and still ran 5.5% slower, with or without
the loop unrolled to make the drain a compile-time branch: the partial kept across tiles chains each
tile's first HMMA on the previous tile's last, and the tensor pipe fell to 72%. Not shipped.

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

## Install

Releases up to `v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.14` carry a wheel per torch
minor, for CPython 3.13, and load only against the torch they were built on. `build.sh` now builds
the stable-ABI wheel, with `precast.patch`. Linux x86_64:

```bash
gh release download <tag> -R sorryhyun/flash-attention-fp16 -p '*.whl'
uv pip install --no-deps --force-reinstall ./flash_attn-*.whl
```

`uv sync` puts the pinned wheel back.

## Build

```bash
./build.sh /path/to/.venv/bin/python
```

The Windows wheel of the stable-ABI build is not built yet; [`windows.md`](windows.md) is the build
request, with the MSVC steps.

`smoke.py` checks an unpacked wheel on the card. `grid.py` measures the installed wheel's error against
fp32 over sequence length, attention spread, V scale and a dO outlier row.

## Licence

FlashAttention is BSD-3-Clause, copyright its contributors; see `LICENSE` and `AUTHORS`.
