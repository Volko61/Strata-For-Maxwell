#!/bin/bash
# One-shot install on the V100 (Vast 60961367): fetch this folder, apply the GGUF vision patch, restart vLLM.
#   curl -fsSL https://raw.githubusercontent.com/Volko61/Strata-For-Maxwell/claude/project-thread-scwwe7/integrations/vllm-v41-vision/install.sh | bash
set -e
R=https://raw.githubusercontent.com/Volko61/Strata-For-Maxwell/claude/project-thread-scwwe7/integrations/vllm-v41-vision
P=/workspace/onecat/vision-patch; mkdir -p $P
for f in vision-gguf.diff apply.sh revert.sh exp_probs_b_vl.f32 imgreq.json imgtest.sh; do curl -fsSL --retry 5 -o $P/$f $R/$f; done
bash $P/apply.sh
supervisorctl restart onecat
echo "vLLM restarting with images on; log: /workspace/logs/onecat-server.log (about 6 min)"
