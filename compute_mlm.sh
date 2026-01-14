#!/usr/bin/env bash
set -euo pipefail

START=400
END=900
STEP=100

LOG="runs/tmp2.log"
TMUX_LOG_DIR="runs/tmux_logs"
mkdir -p "$(dirname "$LOG")" "$TMUX_LOG_DIR"

for x in $(seq "$START" "$STEP" "$END"); do
  y=$((x + 99))
  idx=$(((x - START) / STEP))
  gpu=$((idx % 8))
  session="mlm_${x}_${y}_gpu${gpu}"
  sess_log="${TMUX_LOG_DIR}/${session}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session exists, skipping: $session"
    continue
  fi

  tmux new-session -d -s "$session" "bash --noprofile --norc -c '
    set -eo pipefail

    # log everything
    mkdir -p \"$(printf %q "$TMUX_LOG_DIR")\"
    exec > >(tee -a \"$(printf %q "$sess_log")\") 2>&1

    echo \"[$session] START \$(date)\"

    # keep pane open on failure for debugging
    trap '\''echo \"[$session] ERROR at \$(date)\"; exec bash'\'' ERR

    cd ~/DNALM/ModernBERT/

    # bring conda into this clean shell
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
      source /opt/conda/etc/profile.d/conda.sh
    elif [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then
      source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
    else
      echo \"conda.sh not found (checked /opt/conda and ~/miniconda3)\" >&2
      exit 1
    fi

    conda activate bert24
    set -euo pipefail

    for i in {${x}..${y}}; do
      date
      echo \"$session: \$i --> \$(date)\" >> \"$(printf %q "$LOG")\"
      CUDA_VISIBLE_DEVICES=${gpu} python src/data/fast_compute_MLM_metrics.py \
        --job_number \"\${i}\" \
        --batch_size 96 \
        --num_workers 8 \
        --mlm_efficiency_path /dev/shm/mlm_outputs/ \
        --append_mlm_efficiency \
        --data_dir /home/jovyan/DNALM/ModernBERT/data/
    done

    echo \"[$session] DONE \$(date)\"
    exec bash
  '"

  echo "Started: $session (GPU $gpu, jobs ${x}..${y})"
done

echo "tmux ls"
echo "logs: $TMUX_LOG_DIR/"

