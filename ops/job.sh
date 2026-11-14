# ops job run on the GPU box (see /workspace/tools/job.sh there). Holds no secrets.
# builds strata-dsv41 from this branch's src/dsv41/ds4_model.cu into engine/strata-dsv41.tp (production binary untouched)
cd /workspace/strata; B=claude/keen-volta-duvuj9; R=https://raw.githubusercontent.com/Volko61/Strata-For-Maxwell/$B
supervisorctl status strata strata-worker | cut -c1-70
[ -f src/dsv41/ds4_model.cu.orig ] || cp src/dsv41/ds4_model.cu src/dsv41/ds4_model.cu.orig
for i in 1 2 3 4 5 6 7 8; do curl -fsSL -m 60 "$R/src/dsv41/ds4_model.cu?t=$(date +%s)" -o /tmp/ds4_model.cu.new && break; sleep 3; done
grep -c DS4_TP_UNEVEN /tmp/ds4_model.cu.new && cp /tmp/ds4_model.cu.new src/dsv41/ds4_model.cu
(setsid nohup bash -c 'cd /workspace/strata; t=$(date +%s); .venv/bin/cmake --build build-dsv41-cuda12 --target strata-dsv41 -j 40 > /workspace/logs/build-tp.log 2>&1; rc=$?; [ $rc = 0 ] && cp build-dsv41-cuda12/strata-dsv41 engine/strata-dsv41.tp; echo "BUILD rc=$rc in $(( $(date +%s) - t )) s" >> /workspace/logs/build-tp.log' > /dev/null 2>&1 < /dev/null &)
echo "build started"; ls build-dsv41-cuda12 | head -5; df -h / | tail -1
