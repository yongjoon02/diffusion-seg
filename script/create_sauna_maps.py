"""
Create SAUNA vessel probability maps from binary label images using original uncertainty extraction.

Example usage:
    python script/create_sauna_maps.py --input-dir data/OCTA500_3M/test/label --output-dir data/OCTA500_3M/test/label_sauna
"""

import autorootcwd
import os
import numpy as np
import click
from PIL import Image
from tqdm import tqdm
from src.data.generate_uncertainty import (
    extract_boundary_uncertainty_map,
    extract_thickness_uncertainty_map,
    extract_combined_uncertainty_map,
    ensure_binary_gt
)


@click.command()
@click.option('--input-dir', required=True, help='Directory containing input label images')
@click.option('--output-dir', required=True, help='Directory to save generated SAUNA maps')
def main(input_dir, output_dir):
    """Generate SAUNA vessel probability maps from binary label images."""
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files (BMP and PNG)
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".bmp", ".png"))]
    
    # Process each label file with progress bar
    for filename in tqdm(image_files, desc="Processing image files", unit="file"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load the label image
        label_image = np.array(Image.open(input_path))

        # Ensure binary ground truth
        gt = ensure_binary_gt(label_image)

        # Generate boundary uncertainty map
        gt_b, norm_b = extract_boundary_uncertainty_map(gt, transform_function=None)
        
        # Generate thickness uncertainty map
        gt_t, norm_t = extract_thickness_uncertainty_map(
            gt,
            tr=None,
            target_c_label="h",  # Use "h" for combined boundary + thickness
            kernel_ratio=1.0,
        )
        
        # Generate combined uncertainty map (SAUNA)
        sauna_map = extract_combined_uncertainty_map(gt_b, gt_t, target_c_label="h")
        
        # Convert to vessel probability: map [-1,1] to [0,1]
        vessel_prob = (sauna_map + 1.0) / 2.0
        
        # Convert to 8-bit image format
        vessel_prob_image = (vessel_prob * 255).astype(np.uint8)

        # Save SAUNA map (keep original format)
        Image.fromarray(vessel_prob_image).save(output_path)

    print(f"Processed all image labels and saved SAUNA maps to {output_dir}")


if __name__ == "__main__":
    main()
