#!/usr/bin/env python3
import os, numpy as np, click
from PIL import Image
from tqdm import tqdm
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt

def load_gray(path): return np.array(Image.open(path).convert('L'))
def ensure_binary_gt(a): return (a>0).astype(np.uint8)

def signed_distance_map_norm(gt, cap=11.0):
    g=ensure_binary_gt(gt).astype(np.uint8)
    if g.max()==0: return np.full_like(g,dtype=np.float32,fill_value=-1.0)
    d_in  = distance_transform_edt(g)
    d_out = distance_transform_edt(1-g)
    sdt = d_in - d_out
    sdt = np.clip(sdt, -cap, cap) / (cap+1e-6)
    return sdt.astype(np.float32)

def thickness_map_pixels(gt, sampling=(1.0,1.0)):
    g=ensure_binary_gt(gt)>0
    if not g.any(): return np.zeros_like(gt,dtype=np.float32)
    skel, dist = medial_axis(g, return_distance=True)
    r = (dist*skel).astype(np.float32)
    inv = 1 - skel.astype(np.uint8)
    _, (iy, ix) = distance_transform_edt(inv, sampling=sampling, return_distances=True, return_indices=True)
    t = np.zeros_like(dist, np.float32)
    t[g] = 2.0 * r[iy[g], ix[g]]
    return t

def combine_sauna_h(gt_b, gt_t):
    fg = gt_b>0
    out = gt_b.copy()
    at = np.abs(gt_t)
    m1 = (at>0)&(gt_b>0); out[m1] = gt_b[m1] + (1.0 - gt_t[m1])
    m2 = (at>0)&(gt_b<0); out[m2] = gt_b[m2] - (1.0 - gt_t[m2])
    return np.clip(out,-1.0,1.0)

def list_images(p):
    exts=('.bmp','.png','.tif','.tiff','.jpg','.jpeg')
    return [f for f in os.listdir(p) if f.lower().endswith(exts)]

@click.command()
@click.option('--input-dir', required=True)
@click.option('--output-dir', required=True)
@click.option('--thick-pct', default=99.5, show_default=True, type=float, help='Global percentile for thickness normalization')
@click.option('--sdt-cap', default=11.0, show_default=True, type=float, help='Signed distance cap (pixels) before [-1,1] mapping')
@click.option('--spacing-y', default=1.0, show_default=True, type=float, help='Pixel size along Y (for physical EDT)')
@click.option('--spacing-x', default=1.0, show_default=True, type=float, help='Pixel size along X (for physical EDT)')
def main(input_dir, output_dir, thick_pct, sdt_cap, spacing_y, spacing_x):
    os.makedirs(output_dir, exist_ok=True)
    files = list_images(input_dir)
    vals=[]
    for f in tqdm(files, desc='Scan global stats', unit='file'):
        gt = ensure_binary_gt(load_gray(os.path.join(input_dir,f)))
        t = thickness_map_pixels(gt, sampling=(spacing_y, spacing_x))
        if t.max()>0: vals.append(t[gt>0])
    if len(vals)==0: raise RuntimeError('No vessel pixels found.')
    vals = np.concatenate(vals)
    gmax = np.percentile(vals, thick_pct)
    if gmax<=0: gmax = float(vals.max() if vals.max()>0 else 1.0)

    for f in tqdm(files, desc='Generate SAUNA', unit='file'):
        ip = os.path.join(input_dir,f); op = os.path.join(output_dir,f)
        gt = ensure_binary_gt(load_gray(ip))
        gt_b = signed_distance_map_norm(gt, cap=sdt_cap)                         # [-1,1]
        t_px = thickness_map_pixels(gt, sampling=(spacing_y, spacing_x))         # [0,∞) in phys units
        gt_t = np.zeros_like(t_px, np.float32); m=(gt>0)
        gt_t[m] = np.clip(t_px[m], 0, gmax)/(gmax+1e-6)                          # [0,1] global-norm
        sauna = combine_sauna_h(gt_b, gt_t)                                      # [-1,1]
        prob = np.clip((sauna+1.0)*0.5, 0, 1)                                    # [0,1]
        Image.fromarray((prob*255).astype(np.uint8)).save(op)

if __name__=='__main__':
    main()
