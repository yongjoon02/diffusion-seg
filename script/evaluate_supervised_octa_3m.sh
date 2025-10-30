#!/bin/bash
# Evaluate all models on OCTA500_3M dataset

# Models to evaluate
MODELS="cenet,csnet,aacaunet,unet3plus,vesselnet,transunet,dscnet"

echo "Evaluating all models on OCTA500_3M dataset..."

# Run evaluation for all models at once
uv run python script/evaluate_supervised_models.py \
    --data_name octa500_3m \
    --models "${MODELS}" \
    --output_dir results/octa500_3m \
    > logs/evaluate_octa500_3m.log 2>&1 &

echo "Started evaluation (PID: $!)"
echo "Monitor progress with: tail -f logs/evaluate_octa500_3m.log"
