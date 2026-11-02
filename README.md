# flash-attention-sm120

FlashAttention 2.8.3 built for RTX 50-series cards (sm_120), with the 128×32 forward tile.

On compute capability 12, FA2's `run_mha_fwd_hdim128` falls to the A100 / H100 tile, 128×64, and
exposes no option to pick another. `patches/sm120.patch` sends cc 12 to the tile sm86 / 89 use, and
limits the build to what these wheels are for:

- `is_sm8x` also true on `cc_major == 12`, in `run_mha_fwd_hdim128` only.
- sm_120 only, head dim 128 only. Any other head dim raises
  `this FA2 build holds head dim 128 only`.
- `-std=c++20`, which torch 2.14's headers need.

## Measured

RTX 5070 Ti, bf16, ms per call, against the wheel's 128×64:

| | forward | forward + backward |
|---|---|---|
| self, q `[1, 4131, 16, 128]` | 1.582 → 1.499 (1.056x) | 5.332 → 5.244 (1.017x) |
| cross, Lk 512 | 0.209 → 0.187 (1.118x) | 0.808 → 0.785 (1.029x) |
| self, q `[1, 4600, 24, 128]`, random input | 2.831 → 2.684 (1.055x) | 9.848 → 9.670 (1.018x) |

Median distance from fp32 on 48 captured calls: self 0.00170 → 0.00169, cross 0.00181 → 0.00175.

## Install

Releases carry a wheel per torch minor, matched to the tag. Linux x86_64, CPython 3.13:

```bash
gh release download <tag> -R sorryhyun/flash-attention-sm120 -p '*.whl'
uv pip install --no-deps --force-reinstall ./flash_attn-*.whl
```

A wheel loads only against the torch it was built on. `uv sync` puts the pinned wheel back.

## Build

```bash
./build.sh /path/to/.venv/bin/python
```

## Licence

FlashAttention is BSD-3-Clause, Copyright (c) 2022 the Princeton / Stanford authors; see `LICENSE`
and `AUTHORS`.
