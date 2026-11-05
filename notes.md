# Notes

Findings behind the current build, and what was tried and left out.

## The fp16acc backward under ncu

Under ncu with locked clocks the fp16acc backward kernel runs the tensor pipe 74% of the time against
91% for the stock bf16 kernel (5.81 against 8.23 ms). HMMA on the fp16-accumulate path is 2x, but the
per-tile drain of the fp16 partials into fp32 (`cvt` + `fadd`, 3.6 G each on this shape) raises FMA-pipe
instructions 5.5x and is not hidden with two warps per scheduler.

Draining dK and dV every second tile (fp16 partials kept across tiles) cut those instructions 20% and
still ran 5.5% slower, with or without the loop unrolled to make the drain a compile-time branch: the
partial kept across tiles chains each tile's first HMMA on the previous tile's last, and the tensor
pipe fell to 72%. Not shipped.

## precast.patch (dropped)

Released once, in `v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-precast-cu132-torch2.12plus` (2026-11-04),
and removed from the build the same day. The patch is kept in
[`patches/dropped/precast.patch`](patches/dropped/precast.patch); it applies after `stable_abi.patch`.

### What it did

`fwd` returned its fp16 copies of Q, K and V (empty when the bf16 forward ran), `FlashAttnFunc` saved
them, and `bwd` took them with bf16 O and dO, so the backward skipped its three casts. `amax(abs(dO))`
became one kernel. Kernels unchanged; dK and dV bitwise the self-cast path.

Per call, MOSS self shape `[1, 4600, 24, 128]`, forward + backward: 7.95 → 7.73 ms.

### Why it was dropped

A MOSS-Image LoRA training step (anima_lora `configs/moss/lora.toml`: int8 base, fp32 stream,
torch.compile, `activation_memory_budget` 0.3), RTX 5070 Ti, forward and backward summed over steps
20–40, one run each:

| wheel | PV forward | forward | backward | total | peak |
|---|---|---|---|---|---|
| mjun0812 stock (two runs) | – | 7.27 / 7.24 s | 15.66 / 15.59 s | 22.92 / 22.83 s | 12.85 GB |
| fwdpv, per-torch | on | 6.87 | 14.52 | 21.38 | 12.85 |
| fwdpv, stable ABI | on | 6.87 | 14.55 | 21.42 | 12.85 |
| precast | on | 6.87 | 15.64 | 22.51 | 13.01 |
| precast | off | 7.12 | 14.44 | 21.56 | 12.85 |

With the PV forward on, precast's backward is back at stock (+55 ms a step) and the peak is 0.16 GB
higher. With it off, precast has nothing to hand over and matches the other builds. The stable ABI
costs nothing. Likely cause: the three extra fp16 tensors saved per attention call change what the
compiled graph keeps and what it recomputes under the activation budget. Not separated further.
