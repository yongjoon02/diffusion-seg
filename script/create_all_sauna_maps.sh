#!/bin/bash
# Create SAUNA maps for all datasets

echo "🚀 Creating SAUNA maps for all datasets..."
echo "=================================================="

# Datasets to process
DATASETS=("OCTA500_3M" "OCTA500_6M" "ROSSA")

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "📁 Processing ${dataset}..."
    echo "----------------------------------------"
    
    if [ "$dataset" = "ROSSA" ]; then
        # ROSSA has different folder structure
        subdirs=("train_manual" "train_sam" "val" "test")
    else
        # OCTA500_3M and OCTA500_6M have standard structure
        subdirs=("train" "val" "test")
    fi
    
    for subdir in "${subdirs[@]}"; do
        input_dir="data/${dataset}/${subdir}/label"
        output_dir="data/${dataset}/${subdir}/label_sauna"
        
        if [ -d "$input_dir" ]; then
            echo "  Processing ${subdir}..."
            uv run python script/create_sauna_maps_v2.py --input-dir "$input_dir" --output-dir "$output_dir"
        else
            echo "  ⚠️  Input directory not found: $input_dir"
        fi
    done
    
    echo "✅ ${dataset} completed"
done

echo ""
echo "=================================================="
echo "🎉 All SAUNA map generation completed!"
echo ""
echo "Generated SAUNA maps in:"
echo "  - data/OCTA500_3M/*/label_sauna/"
echo "  - data/OCTA500_6M/*/label_sauna/"
echo "  - data/ROSSA/*/label_sauna/"
