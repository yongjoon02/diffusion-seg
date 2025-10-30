#!/bin/bash
# Train diffusion models in parallel

# Base config
CONFIG="configs/octa500_3m_diffusion_models.yaml"

# Models to train
MODELS=("segdiff" "medsegdiff" "berdiff" "colddiff" "maskdiff" "maskdiff_v2")

# Run each model in parallel
for model in "${MODELS[@]}"; do
    # Check if checkpoint exists
    CKPT_PATH="lightning_logs/octa500_3m/${model}/checkpoints/last.ckpt"
    
    if [ -f "$CKPT_PATH" ]; then
        echo "⏭️  Skipping ${model} - checkpoint already exists: $CKPT_PATH"
        continue
    fi
    
    echo "🚀 Starting training for ${model}..."
    
    # Run in background
    uv run python script/train_diffusion_models.py fit \
        --config ${CONFIG} \
        --arch_name ${model} \
        > logs/train_octa500_3m_${model}.log 2>&1 &
    
    echo "✓ Started ${model} (PID: $!)"
    
    # Wait a bit to avoid overwhelming the system
    sleep 5
done

echo ""
echo "=================================================="
echo "All models started. Check logs/ directory for outputs."
echo "To monitor: tail -f logs/train_octa500_3m_*.log"

