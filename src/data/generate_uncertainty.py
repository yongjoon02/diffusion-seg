import os
import numpy as np
import cv2
import torch
import torch.nn.functional as functional
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import click
from PIL import Image
from skimage.morphology import skeletonize, thin
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import (
    erosion, dilation, opening, closing,
    white_tophat, disk, square
)
from tqdm import tqdm
from pathlib import Path
import matplotlib
import matplotlib.patches as patches

# Ensure input is grayscale
def ensure_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Ensure ground truth is binary (0 or 1) and uint8
def ensure_binary_gt(gt):
    gt = ensure_grayscale(gt)
    gt = (gt > 0).astype(np.uint8)  # Convert to binary (0 or 1)
    return gt

# Resize mask to a standard shape
def resize_mask(mask, std_shape):
    return cv2.resize(mask, std_shape, interpolation=cv2.INTER_NEAREST)

def count_nb(values):
    c = values.size // 2
    return (values.sum() - values[c]) * values[c]


def new_func(neighbor_map):
    inter_dist = distance_transform_edt(neighbor_map, cv2.DIST_L2, 5)


def generate_constant_soft_labels(gt, epsilon=0.5, nc=1, boundary=False, k=5):
    gt = torch.tensor(gt, dtype=int).view(1, gt.shape[0], gt.shape[1])
    label_one_hot = functional.one_hot(
        gt, nc + 1).permute(0, 3, 1, 2).to(torch.float)
    label_one_hot, label_ignore_one_hot = label_one_hot[:,
                                                        :-1, :, :], label_one_hot[:, -1, :, :]

    label_pool = (-functional.max_pool2d(-label_one_hot,
                  kernel_size=k, stride=1, padding=k//2)).to(torch.bool)
    label_ignore_pool = functional.max_pool2d(
        label_ignore_one_hot, kernel_size=k, stride=1, padding=k//2).to(torch.bool)

    label_boundary = torch.any(label_one_hot != label_pool, dim=1)
    # label_boundary[label_ignore_pool] = 0

    batch_size, crop_h, crop_w = gt.shape
    label_boundary = label_boundary.unsqueeze(
        1).expand(batch_size, nc + 1, crop_h, crop_w)

    soft_label = functional.one_hot(
        gt, nc + 1).permute(0, 3, 1, 2).to(torch.float)
    soft_label = torch.flip(soft_label, dims=[1])
    if boundary:
        soft_label[label_boundary] *= (1 - epsilon)
        soft_label[label_boundary] += (epsilon / nc)
    else:
        soft_label *= (1 - epsilon)
        soft_label += (epsilon / nc)
    soft_label = soft_label[0, 0, :, :].cpu().numpy()
    return soft_label


def compute_distance_transform(image):
    image = ensure_grayscale(image)
    return cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)


def transform_distance_map(dist, t):
    return dist


def extract_boundary_uncertainty_map(gt, transform_function=None):
    gt = ensure_binary_gt(gt)  # Ensure binary ground truth

    # Determine the distance transform.
    img_fg_dist = compute_distance_transform(gt)
    img_fg_dist = transform_distance_map(img_fg_dist, transform_function)

    fg_max = 11  # Fixed value for consistent normalization
    gt_bg = 1 - gt
    img_bg_dist = compute_distance_transform(gt_bg)
    img_bg_dist = transform_distance_map(img_bg_dist, transform_function)
    img_bg_dist = -img_bg_dist
    img_bg_dist[img_bg_dist <= -fg_max] = -fg_max
    img_dist = (img_fg_dist + img_bg_dist) / (fg_max + 1e-6)

    # Ensure img_dist does not contain all zeros
    if np.all(img_dist == 0):
        return np.full_like(img_dist, -1.0), fg_max

    return img_dist, fg_max


def extract_nodes(gt):
    # skeleton
    skeleton = skeletonize(gt) * 1.0

    kernel3 = square(3)
    skeleton = closing(skeleton, kernel3)

    footprint = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])

    neighbor_map = ndimage.generic_filter(
        skeleton, count_nb, footprint=footprint)
    leaf = (neighbor_map == 1) * 255
    inter = (neighbor_map >= 3) * 255

    element = disk(10)
    inter_dilated = dilation(inter, element)

    element = square(8)
    leaf_dilated = dilation(leaf, element)

    # ske_inter = dist + inter_dilated + leaf_dilated
    return inter_dilated, leaf_dilated


def do_max_pooling(img_dist, kernel_size=None, kernel_ratio=1):
    thickness_max = img_dist.max()
    kernel_size = thickness_max if kernel_size is None else kernel_size
    kernel_size = int(np.ceil(kernel_size))
    kernel_size = int(kernel_size * kernel_ratio)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    # print(kernel_size, thickness_max)

    # proposed stroke width transform
    img_dist = torch.tensor(img_dist).unsqueeze(0).unsqueeze(0)

    maxpool = torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=1, padding=kernel_size//2)
    img_maxpool = maxpool(img_dist)[0, 0, :, :].numpy()
    return img_maxpool


def extract_thickness_uncertainty_map(
    gt, kernel_size=None,
    tr=None,
    target_c_label="b",
    kernel_ratio=1,
):
    gt = ensure_binary_gt(gt)  # Ensure binary ground truth

    # Determine the distance transform.
    img_dist = compute_distance_transform(gt)
    img_dist = transform_distance_map(img_dist, tr)
    thickness_max = 9  # Fixed value for consistent normalization
    fg_max = thickness_max
    # print("raw t:", np.min(img_dist), np.max(img_dist))
    img_maxpool = do_max_pooling(img_dist, kernel_size=kernel_size)

    if target_c_label in ["hh"]:
        img_thick_pos = (gt > 0) * img_maxpool / (thickness_max + 1e-6)
        img_maxpool = do_max_pooling(
            img_dist, kernel_size=kernel_size, kernel_ratio=kernel_ratio)
        img_thick_neg = (gt <= 0) * img_maxpool / (thickness_max + 1e-6)
        img_swt = np.where(gt > 0, img_thick_pos, img_thick_neg)
    elif target_c_label in ["h"]:
        img_thick_pos = (gt > 0) * img_maxpool / (thickness_max + 1e-6)
        img_bg_dist = compute_distance_transform(1 - gt)
        img_bg_dist = transform_distance_map(img_bg_dist, tr)
        img_bg_dist[img_bg_dist >= fg_max] = fg_max
        img_bg_maxpool = do_max_pooling(img_bg_dist, kernel_size=kernel_size)
        img_thick_neg = (gt <= 0) * img_bg_maxpool / (thickness_max + 1e-6)
        img_thick_neg = np.clip(img_thick_neg, a_min=0.0, a_max=1.0)
        img_swt = np.where(gt > 0, img_thick_pos, img_thick_neg)
    else:
        img_thick_pos = (gt > 0) * img_maxpool / (thickness_max + 1e-6)
        img_swt = img_thick_pos

    return img_swt, thickness_max


def extract_combined_uncertainty_map(gt_b, gt_t, target_c_label):
    fg = gt_b > 0
    gt_c = gt_b.copy()
    gt_t_abs = np.abs(gt_t)

    if target_c_label == "c":
        gt_c[fg] = gt_b[fg] * gt_t[fg]
    elif target_c_label == "h":
        fg = (gt_t_abs > 0) & (gt_b > 0)
        gt_c[fg] = gt_b[fg] + (1.0 - gt_t[fg])
        bg = (gt_t_abs > 0) & (gt_b < 0)
        gt_c[bg] = gt_b[bg] - (1.0 - gt_t[bg])
    elif target_c_label == "hh":
        center = 1.0
        fg = (gt_t_abs > 0) & (gt_b > 0)
        gt_c[fg] = gt_b[fg] + (center - gt_t[fg])
        bg = (gt_t_abs > 0) & (gt_b < 0)
        bg_zero_thickness = (gt_t_abs == 0) & (gt_c < 0)
        gt_c[bg] = gt_b[bg] - (center - gt_t[bg])
        gt_c[bg_zero_thickness] = -1.0

    gt_c = np.clip(gt_c, a_min=-1.0, a_max=1.0)
    return gt_c


def extract_relative_location_map(gt):
    gt = ensure_binary_gt(gt)  # Ensure binary ground truth

    skeleton = skeletonize(ensure_grayscale(gt) / 1.0) * 1.0
    skeleton = 1.0 - skeleton
    skeleton = skeleton.astype(np.uint8)

    dist_to_skeleton = compute_distance_transform(skeleton)
    dist_to_skeleton[gt == 0] = -1.0

    max_dist = dist_to_skeleton.max()
    if max_dist <= 0.0:
        return np.full_like(dist_to_skeleton, -1.0)

    dist_to_skeleton[gt > 0] = dist_to_skeleton[gt > 0] / max_dist
    return dist_to_skeleton


def convert_gt(root, in_dir, ext=".png", std_shape=(512, 512)):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)

    filenames = sorted(os.listdir(os.path.join(root, in_dir)))
    out_dir_c = os.path.join(root, in_dir + "_c")

    os.makedirs(out_dir_c, exist_ok=True)
    all_norm_b, all_norm_t = [], []
    target_c_label = "h"
    tr = None
    kernel_ratio = 1.0
    file_index = 0
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    for filename in tqdm(filenames, total=len(filenames), desc="Process:"):
        if filename.endswith(ext):
            fullname = os.path.join(root, in_dir, filename)
            image_name = Path(fullname).stem
            gt = Image.open(fullname).convert("L")
            gt = np.array(gt)
            gt = resize_mask(gt, std_shape)

            gt_b, norm_b = extract_boundary_uncertainty_map(gt, tr)
            gt_t, norm_t = extract_thickness_uncertainty_map(
                gt,
                tr=tr,
                target_c_label=target_c_label,
                kernel_ratio=kernel_ratio,
            )
            assert not np.any(np.isnan(gt_b)), image_name
            assert not np.any(np.isnan(gt_t)), image_name
            all_norm_b.append(norm_b)
            all_norm_t.append(norm_t)

            gt_c = extract_combined_uncertainty_map(gt_b, gt_t, target_c_label)

            fullname_c = os.path.join(out_dir_c, image_name + ".npy")
            with open(fullname_c, "wb") as f:
                np.save(f, gt_c)

            gt_r = extract_relative_location_map(gt)
            file_index += 1
            if file_index < 5:
                visualize_new_labels(fullname, gt, gt_t,
                                     gt_b, gt_c, gt_r, output_dir=temp_dir)

    os.rename(out_dir_c, out_dir_c[:-1] + target_c_label + f"_{std_shape[0]}")

    all_norm_b = np.array(all_norm_b)
    all_norm_t = np.array(all_norm_t)
    print(all_norm_b.min(), all_norm_b.mean(), np.median(
        all_norm_b), np.var(all_norm_b), all_norm_b.max())
    print(all_norm_t.min(), all_norm_t.mean(), np.median(
        all_norm_t), np.var(all_norm_t), all_norm_t.max())


@click.command()
@click.option("--root")
@click.option("--in_dir")
def main(root: str, in_dir: str):
    std_resize_to = (512, 512)
    convert_gt(
        root=root,
        in_dir=in_dir,
        ext=".png",
        std_shape=std_resize_to,
    )


if __name__ == "__main__":
    main()