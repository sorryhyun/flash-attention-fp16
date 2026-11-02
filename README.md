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

dQ, dK, dV sit at 0.74–0.97x stock's median distance from fp32 on 48 captured calls. A real MOSS-Image
training step goes from 1071–1082 ms to 1022 ms at the same peak memory.

## Install

Releases carry a wheel per torch minor, matched to the tag, for CPython 3.13. Linux x86_64:

```bash
gh release download <tag> -R sorryhyun/flash-attention-fp16 -p '*.whl'
uv pip install --no-deps --force-reinstall ./flash_attn-*.whl
```

A wheel loads only against the torch it was built on. `uv sync` puts the pinned wheel back.

## Build

```bash
./build.sh /path/to/.venv/bin/python
```

Windows: [windows.md](windows.md). `smoke.py` checks an unpacked wheel on the card.

## Licence

FlashAttention is BSD-3-Clause, copyright its contributors; see `LICENSE` and `AUTHORS`.
