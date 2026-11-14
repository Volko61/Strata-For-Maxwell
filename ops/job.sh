# ops job run on the GPU box (see /workspace/tools/job.sh there). Holds no secrets.
cd /workspace; supervisorctl status strata strata-worker engram-ram | cut -c1-70
echo "--- build:"; grep -E "error|Error|BUILD rc|FAILED|\[[0-9]+/[0-9]+\]" logs/build-tp.log | tail -8 | cut -c1-220; ls -la strata/engine/ | cut -c24-
