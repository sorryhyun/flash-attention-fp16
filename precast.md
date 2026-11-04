# precast.patch (dropped)

Released once, in `v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-precast-cu132-torch2.12plus` (2026-11-04),
and removed from the build the same day. The patch is kept in
[`patches/dropped/precast.patch`](patches/dropped/precast.patch); it applies after `stable_abi.patch`.

## What it did

`fwd` returned its fp16 copies of Q, K and V (empty when the bf16 forward ran), `FlashAttnFunc` saved
them, and `bwd` took them with bf16 O and dO, so the backward skipped its three casts. `amax(abs(dO))`
became one kernel. Kernels unchanged; dK and dV bitwise the self-cast path.

Per call, MOSS self shape `[1, 4600, 24, 128]`, forward + backward: 7.95 → 7.73 ms.

## Why it was dropped

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
