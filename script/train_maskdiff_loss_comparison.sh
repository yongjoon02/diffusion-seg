#!/bin/bash
# Train maskdiff with different loss types on ROSSA dataset
# Comparison: hybrid vs mse loss with timesteps=50

# Base config
CONFIG="configs/rossa_diffusion_models.yaml"

# Loss types to compare
LOSS_TYPES=("hybrid" "mse")

# Run each loss type
for loss_type in "${LOSS_TYPES[@]}"; do
    # Check if checkpoint exists
    CKPT_PATH="lightning_logs/rossa/maskdiff_${loss_type}/checkpoints/last.ckpt"
    
    if [ -f "$CKPT_PATH" ]; then
        echo "⏭️  Skipping maskdiff_${loss_type} - checkpoint already exists: $CKPT_PATH"
        continue
    fi
    
    echo "🚀 Starting training for maskdiff with loss_type=${loss_type}..."
    
    # Run in background
    uv run python script/train_diffusion_models.py fit \
        --config ${CONFIG} \
        --arch_name maskdiff \
        --model.timesteps 50 \
        --model.loss_type ${loss_type} \
        --trainer.logger.init_args.name rossa \
        --trainer.logger.init_args.version maskdiff_${loss_type} \
        > logs/train_rossa_maskdiff_${loss_type}.log 2>&1 &
    
    echo "✓ Started maskdiff_${loss_type} (PID: $!)"
    
    # Wait a bit to avoid overwhelming the system
    sleep 5
done

echo ""
echo "=================================================="
echo "All maskdiff models started with different loss types."
echo "Check logs/ directory for outputs."
echo "To monitor: tail -f logs/train_rossa_maskdiff_*.log"
echo ""
echo "Experiments:"
echo "  - maskdiff_hybrid: BCE+Dice + weighted Focal L1"
echo "  - maskdiff_mse: Focal L1 only"
echo "  - Both with timesteps=50 for faster training"
