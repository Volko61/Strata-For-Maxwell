#!/bin/bash
# undo apply.sh (then: supervisorctl restart onecat)
set -e
S=/workspace/onecat/src1370; B=/workspace/onecat/vision-backup
tar xzf $B/orig-src.tgz -C $S; rm -f $S/vllm/models/deepseek_v41/vision.py
cp $B/serve-onecat.sh /workspace/onecat/serve-onecat.sh
echo REVERTED
