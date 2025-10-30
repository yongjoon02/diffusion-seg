#!/bin/bash

# OCTA500 6M Supervised Training Script
# Usage: ./train_supervised_octa_6m.sh

models=("cenet" "csnet" "aacaunet" "unet3plus" "vesselnet" "transunet" "dscnet")

echo "Starting OCTA500 6M supervised training for all models..."

for model in "${models[@]}"; do
    echo "Training $model on OCTA500 6M..."
    
    # Run training in background
    uv run python script/train_supervised_models.py fit \
        --config configs/octa500_6m_supervised_models.yaml \
        --arch_name $model \
        > logs/train_octa500_6m_${model}.log 2>&1 &
    
    echo "Started training $model (PID: $!)"
    sleep 2  # Small delay between starts
done

echo "All training jobs started!"
echo "Check logs in logs/ directory for progress"
echo "Use 'ps aux | grep train_supervised_models' to see running processes"