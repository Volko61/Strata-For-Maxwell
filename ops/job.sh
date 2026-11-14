# ops job run on the GPU box (see /workspace/tools/job.sh there). Holds no secrets.
cd /workspace
supervisorctl status strata strata-worker engram-ram | cut -c1-70
echo "--- run 2 (drafter + profiler):"; grep -E "^decode:|^dspark:|GPU per token|prompt:" "logs/cliprof/err_auto_DS4_PROF=1.txt" | cut -c1-330
