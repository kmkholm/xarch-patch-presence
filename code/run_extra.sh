#!/bin/bash
# Runs after run_grid.sh: binary-level split protocol (fullbin), then fusion of all configs.
cd "$(dirname "$0")"
mkdir -p ../logs ../runs
LOG=../logs/grid.log
until grep -q "GRID DONE" $LOG; do sleep 60; done
run() { echo "$(date +%H:%M:%S) START $*" >> $LOG; python -u "$@" >> $LOG 2>&1; echo "$(date +%H:%M:%S) END $* rc=$?" >> $LOG; }
for pass in 1 2; do
  for f in 0 1 2 3 4; do [ -f ../runs/fullbin/fold$f.csv ] || run finetune.py --config fullbin --fold $f; done
done
python fuse.py full x86ref xarch vulnonly nopretrain nogate valuegate sharedgate fullbin >> $LOG 2>&1
python evaluate.py full+fused x86ref+fused xarch+fused vulnonly+fused nopretrain+fused nogate+fused valuegate+fused sharedgate+fused fullbin fullbin+fused >> $LOG 2>&1
echo "$(date +%H:%M:%S) EXTRA DONE" >> $LOG
