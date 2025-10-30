#!/bin/bash
# Evaluate all models on OCTA500_6M dataset

# Models to evaluate
MODELS="cenet,csnet,aacaunet,unet3plus,vesselnet,transunet,dscnet"

echo "Evaluating all models on OCTA500_6M dataset..."

# Run evaluation for all models at once
uv run python script/evaluate_supervised_models.py \
    --data_name octa500_6m \
    --models "${MODELS}" \
    --output_dir results/octa500_6m \
    > logs/evaluate_octa500_6m.log 2>&1 &

echo "Started evaluation (PID: $!)"
echo "Monitor progress with: tail -f logs/evaluate_octa500_6m.log"
