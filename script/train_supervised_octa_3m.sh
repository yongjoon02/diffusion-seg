#!/bin/bash
# Train multiple supervised models in parallel

# Base config
CONFIG="configs/octa500_3m_supervised_models.yaml"

# Models to train
MODELS=("cenet" "csnet" "aacaunet" "unet3plus" "vesselnet" "transunet" "dscnet")

# Run each model in parallel
for model in "${MODELS[@]}"; do
    echo "Starting training for ${model}..."
    
    # Run in background
    uv run python script/train_supervised_models.py fit \
        --config ${CONFIG} \
        --arch_name ${model} \
        > logs/train_octa500_3m_${model}.log 2>&1 &
    
    echo "Started ${model} (PID: $!)"
    
    # Wait a bit to avoid overwhelming the system
    sleep 5
done

echo "All models started. Check logs/ directory for outputs."
echo "To monitor: tail -f logs/train_*.log"

