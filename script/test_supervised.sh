#!/bin/bash
# Test multiple supervised models

# Base config
CONFIG="configs/octa500_3m_supervised_models.yaml"

# Models to test
MODELS=("cenet" "csnet" "aacaunet" "unet3plus" "vesselnet" "transunet" "dscnet")

# Run each model test
for model in "${MODELS[@]}"; do
    echo "Starting test for ${model}..."
    
    # Find the latest checkpoint for this model
    checkpoint=$(find lightning_logs -name "*.ckpt" -path "*/${model}/*" | sort -V | tail -1)
    
    if [ -n "$checkpoint" ]; then
        echo "Found checkpoint: $checkpoint"
        
        # Run test
        uv run python script/train_supervised_models.py test \
            --config ${CONFIG} \
            --model.arch_name ${model} \
            --ckpt_path "$checkpoint" \
            > logs/test_octa500_3m_${model}.log 2>&1 &
        
        echo "Started test for ${model} (PID: $!)"
    else
        echo "No checkpoint found for ${model}"
    fi
    
    sleep 5
done

echo "All tests started. Check logs/ directory for outputs."

