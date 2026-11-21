#!/bin/bash
# installs the GGUF vision patch into /workspace/onecat/src1370 (backup in /workspace/onecat/vision-backup)
set -e
S=/workspace/onecat/src1370; P=/workspace/onecat/vision-patch; B=/workspace/onecat/vision-backup
cd $S
if ! md5sum -c $P/orig.md5 --quiet; then echo "SOURCE DIFFERS FROM PR1370 HEAD"; exit 3; fi
mkdir -p $B; tar czf $B/orig-src.tgz $(awk '{print $2}' $P/orig.md5); cp /workspace/onecat/serve-onecat.sh $B/serve-onecat.sh
tar xzf $P/patch.tgz -C $S
cp $P/exp_probs_b_vl.f32 /workspace/models/full/exp_probs_b_vl.f32
grep -q VLLM_V41_VISION /workspace/onecat/serve-onecat.sh || sed -i 's|^export CUDA_DEVICE_ORDER=|export VLLM_V41_VISION=1 VLLM_V41_VISION_MMPROJ=/workspace/models/coder/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf VLLM_V41_VISION_BIAS_VL=/workspace/models/full/exp_probs_b_vl.f32\nexport CUDA_DEVICE_ORDER=|' /workspace/onecat/serve-onecat.sh
find $S/vllm -name __pycache__ -path "*deepseek_v41*" -prune -exec rm -rf {} + 2>/dev/null || true
grep -n VLLM_V41_VISION /workspace/onecat/serve-onecat.sh
echo APPLIED
