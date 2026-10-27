#!/bin/bash
# Fixed experiment grid, one GPU job at a time (laptop power budget).
# Main protocol: same-architecture references (full and its ablations);
# x86ref: amd64 references (cross-architecture references).
# Resumable: finished outputs are skipped; a second pass retries any fold that
# failed (for example after a GPU reset).
cd "$(dirname "$0")"
mkdir -p ../logs ../runs
LOG=../logs/grid.log
run() { echo "$(date +%H:%M:%S) START $*" >> $LOG; python -u "$@" >> $LOG 2>&1; echo "$(date +%H:%M:%S) END $* rc=$?" >> $LOG; }

for pass in 1 2; do
  [ -f ../runs/pretrain_sdpa.pt ] || run pretrain.py --gate sdpa
  for cfg in full x86ref xarch vulnonly nopretrain; do
    for f in 0 1 2 3 4; do [ -f ../runs/$cfg/fold$f.csv ] || run finetune.py --config $cfg --fold $f; done
  done
  python evaluate.py full x86ref xarch vulnonly nopretrain overlap_same overlap_x86 >> $LOG 2>&1
  for g in none value shared; do [ -f ../runs/pretrain_$g.pt ] || run pretrain.py --gate $g; done
  for cfg in nogate valuegate sharedgate; do
    for f in 0 1 2 3 4; do [ -f ../runs/$cfg/fold$f.csv ] || run finetune.py --config $cfg --fold $f; done
  done
done
python evaluate.py full x86ref xarch vulnonly nopretrain nogate valuegate sharedgate overlap_same overlap_x86 >> $LOG 2>&1
echo "$(date +%H:%M:%S) GRID DONE" >> $LOG
