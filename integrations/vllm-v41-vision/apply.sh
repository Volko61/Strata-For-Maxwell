#!/bin/bash
# installs the GGUF vision patch into /workspace/onecat/src1370 (backup in /workspace/onecat/vision-backup)
set -e
S=/workspace/onecat/src1370; P=/workspace/onecat/vision-patch; B=/workspace/onecat/vision-backup
cd $S
patch -p1 --dry-run < $P/vision-gguf.diff >/dev/null || { echo "PATCH DOES NOT APPLY"; exit 3; }
mkdir -p $B; tar czf $B/orig-src.tgz vllm/envs.py vllm/models/deepseek_v41/model.py vllm/models/deepseek_v41/dspark.py vllm/renderers/deepseek_v41.py vllm/tokenizers/deepseek_v41.py vllm/v1/worker/gpu_model_runner.py
cp /workspace/onecat/serve-onecat.sh $B/serve-onecat.sh
patch -p1 < $P/vision-gguf.diff
cp $P/exp_probs_b_vl.f32 /workspace/models/full/exp_probs_b_vl.f32
grep -q VLLM_V41_VISION /workspace/onecat/serve-onecat.sh || sed -i 's|^export CUDA_DEVICE_ORDER=|export VLLM_V41_VISION=1 VLLM_V41_VISION_MMPROJ=/workspace/models/coder/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf VLLM_V41_VISION_BIAS_VL=/workspace/models/full/exp_probs_b_vl.f32\nexport CUDA_DEVICE_ORDER=|' /workspace/onecat/serve-onecat.sh
echo APPLIED
