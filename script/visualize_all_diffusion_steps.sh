#!/bin/bash
# Visualize diffusion sampling steps for all models

# Models to visualize
MODELS="segdiff,medsegdiff,colddiff,berdiff,maskdiff"

# Number of samples to visualize per model
NUM_SAMPLES=5

echo "Visualizing diffusion sampling steps for all models..."
echo "Models: ${MODELS}"
echo "Samples per model: ${NUM_SAMPLES}"

# Create output directory
mkdir -p results/diffusion_model/visualization

# Visualize for each model
for model in $(echo $MODELS | tr ',' ' '); do
    echo "Processing ${model}..."
    
    # Visualize multiple samples for each model
    for sample_idx in $(seq 0 $((NUM_SAMPLES-1))); do
        echo "  Sample ${sample_idx}..."
        
        uv run python script/visualize_diffusion_steps.py \
            --model_name ${model} \
            --data_name octa500_3m \
            --sample_idx ${sample_idx} \
            --steps "0,10,20,30,40,50,60,70,80,90,100" \
            --output_dir results/diffusion_model/visualization \
            > logs/visualize_${model}_sample_${sample_idx}.log 2>&1 &
        
        # Wait a bit to avoid overwhelming the system
        sleep 2
    done
done

echo "All visualizations started!"
echo "Check results/diffusion_model/visualization/ directory for results"
echo "Monitor logs with: tail -f logs/visualize_*.log"
