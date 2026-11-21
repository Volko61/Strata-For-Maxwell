# DeepSeek V4.1 image input on the GGUF route (1Cat-vLLM, PR 1370 tree)

Port of 1Cat-vLLM #1561 (`VLLM_V41_VISION`, FP8 native route only) to the GGUF route used on the V100:

- vision tower weights from llama.cpp's projector `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` (`VLLM_V41_VISION_MMPROJ`)
- MoE image routing bias from `exp_probs_b_vl.f32` of ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (`VLLM_V41_VISION_BIAS_VL`, F32 [40, 384])
- everything else as #1561: image spans of `image_token_id`, VL bias on image rows, no Engram n-grams through image spans, OpenAI image blocks in chat

Off unless `VLLM_V41_VISION=1`. `apply.sh` dry-runs `vision-gguf.diff` with `patch -p1` (the box tree has local changes beyond PR 1370), backs the files up and adds the env line to `serve-onecat.sh`; `revert.sh` undoes it. `vision-gguf.diff` is the full change.

Installed on the V100 2026-11-21 13:47 UTC; `imgtest.sh` (carrots.jpeg straight to vLLM) answers "carrots, orange".
