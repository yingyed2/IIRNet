#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

train_model() {
    local depth=$1 width=$2
    local name="update3_cd_${depth}x${width}"
    local checkpoint="checkpoints/${name}.ckpt"

    if [[ ! -f "$checkpoint" ]]; then
        python train.py \
            --experiment_name "$name" \
            --model_name mlp --filter_method cd \
            --num_layers "$depth" --hidden_dim "$width" \
            --num_points 512 --model_order 10 --max_train_order 10 \
            --max_root_radius 0.9 \
            --num_train_examples 2000 --num_val_examples 250 \
            --batch_size 32 --max_epochs 30 --lr 0.001 \
            --shuffle --seed 14 --num_workers 0 --log_every_n_steps 10 --accelerator gpu --devices 1

        # Lightning keeps the best validation checkpoint for each run.
        local run_dir="logs/$name/epochs=30_filter-method=cd_filter-order=10_hidden-dim=$width"
        local latest_version
        latest_version=$(find "$run_dir/lightning_logs" -maxdepth 1 -type d \
            -name 'version_*' | sort -V | tail -1)
        local best=( "$latest_version"/checkpoints/*.ckpt )
        if [[ ${#best[@]} -ne 1 || ! -f "${best[0]}" ]]; then
            echo "Expected one best checkpoint under $latest_version/checkpoints" >&2
            return 1
        fi
        cp "${best[0]}" "$checkpoint"
    fi

    score_model "${depth}x${width}" "$checkpoint"
}

score_model() {
    local label=$1 checkpoint=$2
    python eval_cd.py \
        --checkpoint "$checkpoint" \
        --seed 20261122 --num-examples 250 \
        --output-dir "results/update3/validation/$label"
}

score_model 3x32 checkpoints/cd_mlp_3x32.ckpt
train_model 3 64
train_model 3 16
train_model 1 32
train_model 5 32
