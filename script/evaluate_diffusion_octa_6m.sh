#!/bin/bash
# Evaluate all diffusion models on OCTA500_6M dataset

# Models to evaluate
MODELS="segdiff,medsegdiff,berdiff,colddiff,maskdiff"

echo "Evaluating all diffusion models on OCTA500_6M dataset..."

# Run evaluation for all models at once
uv run python script/evaluate_diffusion_models.py \
    --data_name octa500_6m \
    --models "${MODELS}" \
    --output_dir results/octa500_6m_diffusion \
    > logs/evaluate_octa500_6m_diffusion.log 2>&1 &

echo "Started evaluation (PID: $!)"
echo "Monitor progress with: tail -f logs/evaluate_octa500_6m_diffusion.log"
