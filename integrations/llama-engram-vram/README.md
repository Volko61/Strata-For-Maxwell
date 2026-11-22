# Engram table in VRAM for DeepSeek V4.1 Flash (llama.cpp)

`engram-vram.diff` applies to llama.cpp master `0b51bb2` (2026-11-22).

## Why

The Engram table (`per_layer_token_embd.weight`, IQ4_NL, 256 x 768,022,850 rows, 110.6 GB) is one
tensor that llama.cpp keeps on the host and reads row by row (48 rows per token: 2 Engram layers x
24 hash heads). A machine with little RAM reads it from disk at every token. It is also larger than
any single 80 GB card, so no existing option can place it in VRAM.

## What the patch does

With `LLAMA_DSV41_ENGRAM_DEVICES` set, the table is copied into device memory after load, cut into
one piece per listed device. Cuts fall between hash heads, so each head gathers from one piece, and
the gathered rows are concatenated in the original head order. Without the variable nothing changes.

    LLAMA_DSV41_ENGRAM_DEVICES=CUDA1,CUDA2          # two equal pieces (one Engram layer each, 51.5 GiB)
    LLAMA_DSV41_ENGRAM_DEVICES=CUDA0,CUDA1,CUDA2    # three pieces of 16 heads (34.3 GiB)
    LLAMA_DSV41_ENGRAM_DEVICES=CUDA0:1,CUDA1:2      # weighted shares

The copy is streamed in 64 MiB steps from the mapped file, so host RAM does not need to hold the
table. The pieces are counted by the memory-fit pass. Keep the default `--lazy-mode auto` so the host
side stays a lazy mapping.

## Verified

- CPU build, tiny `deepseek41` model from `test-llama-archs -o`: logits are bit-identical with the
  table uncut and cut in 2, 3 and 5 pieces (cuts inside an Engram layer included), lazy mode on and
  off, ubatch 64 and 16 (`engram_equiv_test.cpp`).
- `test-llama-archs -a '^deepseek41$'` passes.
- Real file headers: head dim 256 (a multiple of the 256-element block the CUDA quantized gather
  needs), 48 contiguous non-overlapping heads of about 16 M rows, row ids below 2^31.

## Not verified

No GPU run yet. To check on the first rental: the load log lines `Engram piece N ... -> CUDAx`,
`nvidia-smi` memory per card, and identical output against a run without the variable at temp 0.

## Suggested layout on 3 x 80 GB

    LLAMA_DSV41_ENGRAM_DEVICES=CUDA1,CUDA2 llama-server -ngl 99 -ts 70,14,14 -cram 0 ...

Card 0 takes most of the 67 GiB transformer, cards 1 and 2 one Engram layer each plus the rest.
About 65 GiB stays free for context. Tune `-ts` from the `nvidia-smi` numbers after the first load.
