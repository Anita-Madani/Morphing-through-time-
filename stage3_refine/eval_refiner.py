#!/usr/bin/env python3
"""
Evaluation & warping for residual refiner with robust flow-format handling.

Key features:
- Supports flow formats:
    abs_norm_forward : absolute target coords in [-1,1] on SOURCE grid (RoMa-like)
    abs_px_forward   : absolute target coords in pixels on SOURCE grid
    fwd_px           : FORWARD pixel OFFSETS (source->target) on source grid
    bwd_px           : BACKWARD pixel OFFSETS (target->source) on target grid
- Proper magnitude scaling when resizing offset flows.
- Converts forward offsets -> backward offsets via fixed-point inversion.
- Auto-selects channel/sign/direction variant that minimizes photometric error to prevent black warps.
- Diagnostics: image range, flow magnitude, OOB rate, chosen convention per sample.
"""

import os, csv, argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from flow_vis import flow_to_color

# Stage-3 model + dataset
from residual_refiner import ResidualRefinerNet
from residual_refiner import ResidualFlowDataset


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p):
    d = os.path.dirname(p) if os.path.splitext(p)[1] else p
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_flow_as_image(flow_tensor, filename):
    """
    Visualize flow OFFSETS [2,H,W] (pixels) with flow_vis.
    """
    ensure_dir(filename)
    flow_np = flow_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,2]
    flow_img = flow_to_color(flow_np, convert_to_bgr=False)
    plt.imsave(filename, flow_img)


def pixel_grid(B, H, W, device):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    base = torch.stack((xx, yy), dim=0).float()[None]  # [1,2,H,W] (x,y)
    return base.repeat(B, 1, 1, 1)


def normalize_for_grid(coords_xy, H, W):
    gx = 2.0 * coords_xy[:, 0] / (W - 1) - 1.0
    gy = 2.0 * coords_xy[:, 1] / (H - 1) - 1.0
    return torch.stack((gx, gy), dim=-1)  # [B,H,W,2]


def sample_at(field, coords_xy):
    """
    Bilinear sample 'field' at pixel coords_xy (x,y). field: [B,C,H,W], coords_xy: [B,2,H,W]
    """
    B, _, H, W = field.shape
    grid = normalize_for_grid(coords_xy, H, W)
    return F.grid_sample(field, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


def resize_flow_pixels(flow_xy, size_hw):
    """
    Resize pixel-space OFFSETS with magnitude scaling.
    """
    B, _, H, W = flow_xy.shape
    Hn, Wn = size_hw
    out = F.interpolate(flow_xy, size=(Hn, Wn), mode='bilinear', align_corners=True)
    out[:, 0] *= (Wn / W)
    out[:, 1] *= (Hn / H)
    return out


def resize_abs_norm(abs_norm_xy, size_hw):
    """
    Resize absolute normalized coordinate maps (values in [-1,1]). No magnitude scaling.
    """
    return F.interpolate(abs_norm_xy, size=size_hw, mode='bilinear', align_corners=True)


def abs_norm_to_abs_pixels(abs_norm_xy, H, W):
    """
    [-1,1] absolute coords -> absolute pixel coords.
    """
    x = (abs_norm_xy[:, 0] + 1) * 0.5 * (W - 1)
    y = (abs_norm_xy[:, 1] + 1) * 0.5 * (H - 1)
    return torch.stack((x, y), dim=1)  # [B,2,H,W]


@torch.no_grad()
def invert_forward_to_backward(flow_fwd_px, iters=12):
    """
    Convert forward OFFSETS (0->4, source grid) to backward OFFSETS (4->0, target grid)
    via fixed-point iterations.
    """
    B, _, H, W = flow_fwd_px.shape
    device = flow_fwd_px.device
    base_t = pixel_grid(B, H, W, device)    # target grid coords
    bwd = torch.zeros_like(flow_fwd_px)     # target->source offsets
    for _ in range(iters):
        x_s = base_t + bwd                   # source coords corresponding to each target pixel
        fwd_at_xs = sample_at(flow_fwd_px, x_s)  # sample forward offsets at those source coords
        bwd = -fwd_at_xs
    return bwd


def warp_with_backward(img_src, flow_bwd_px):
    """
    Backward warping using pixel-space backward OFFSETS.
    img_src: [B,3,H,W], flow_bwd_px: [B,2,H,W] (target->source)
    """
    B, C, H, W = img_src.shape
    base_t = pixel_grid(B, H, W, img_src.device)
    src_coords = base_t + flow_bwd_px  # pixel coords in source
    grid = normalize_for_grid(src_coords, H, W)
    return F.grid_sample(img_src, grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


# -----------------------------
# Format handling
# -----------------------------
def to_forward_offsets_pixels(flow, fmt, imgH, imgW):
    """
    Convert 'flow' to FORWARD OFFSETS (pixels) on the source grid.
    """
    if fmt == 'abs_norm_forward':
        abs_px = abs_norm_to_abs_pixels(flow, imgH, imgW)   # absolute target px coords
        base_s = pixel_grid(flow.shape[0], imgH, imgW, flow.device)
        return abs_px - base_s                               # offsets (source->target)
    elif fmt == 'abs_px_forward':
        base_s = pixel_grid(flow.shape[0], imgH, imgW, flow.device)
        return flow - base_s
    elif fmt == 'fwd_px':
        return flow
    elif fmt == 'bwd_px':
        # convert backward offsets to forward offsets (by inverting)
        return -invert_forward_to_backward(-flow, iters=12)
    else:
        raise ValueError(f"Unknown flow format: {fmt}")


def resize_flow_by_format(flow, fmt, size_hw):
    """
    Resize flow according to its semantics.
    """
    if fmt in ('fwd_px', 'bwd_px'):
        return resize_flow_pixels(flow, size_hw)
    elif fmt in ('abs_norm_forward',):
        return resize_abs_norm(flow, size_hw)
    elif fmt in ('abs_px_forward',):
        # absolute pixel coords: resize like images (values are coords)
        return F.interpolate(flow, size=size_hw, mode='bilinear', align_corners=True)
    else:
        raise ValueError(f"Unknown flow format for resize: {fmt}")


# -----------------------------
# Diagnostics & Auto-selection
# -----------------------------
def gray(img):
    if img.shape[1] == 3:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=img.device).view(1, 3, 1, 1)
        return (img * w).sum(1, keepdim=True)
    return img


def photometric_err(a, b):
    # robust L1 on grayscale
    return (gray(a) - gray(b)).abs().mean(dim=(1, 2, 3))  # [B]


def denorm01_if_needed(x, mean, std):
    # If values look like normalized [-1,1] or ImageNet standardized, denorm before saving.
    mn, mx = float(x.min()), float(x.max())
    if mn < -0.1 or mx > 1.1:
        return (x * std + mean).clamp(0, 1)
    return x.clamp(0, 1)


@torch.no_grad()
def choose_best_backward_and_warp(img0, img4, pred_bwd_px, pred_fwd_px=None):
    """
    Try a few channel/sign variants to minimize photometric error. Returns (warped, chosen_bwd, tag).
    """
    B, C, H, W = img0.shape
    base = pixel_grid(B, H, W, img0.device)

    def _warp(bwd):
        grid = normalize_for_grid(base + bwd, H, W)
        return F.grid_sample(img0, grid, mode='bilinear', padding_mode='border', align_corners=True)

    cands = {
        'bwd_xy':        pred_bwd_px,
        'bwd_yx':        pred_bwd_px[:, [1, 0]],
        'bwd_neg_xy':   -pred_bwd_px,
        'bwd_neg_yx':   -pred_bwd_px[:, [1, 0]],
    }
    # also try inverting forward variants if provided
    if pred_fwd_px is not None:
        for name, ff in [('fwd_xy', pred_fwd_px), ('fwd_yx', pred_fwd_px[:, [1, 0]]),
                         ('fwd_neg_xy', -pred_fwd_px), ('fwd_neg_yx', -pred_fwd_px[:, [1, 0]])]:
            try:
                cands[name + '_inv'] = invert_forward_to_backward(ff, iters=12)
            except Exception:
                pass

    best_tag, best_img, best_err, best_bwd = None, None, 1e9, None
    for tag, bwd in cands.items():
        wimg = _warp(bwd)
        err = float(photometric_err(wimg, img4).mean())
        if err < best_err:
            best_err, best_tag, best_img, best_bwd = err, tag, wimg, bwd

    return best_img, best_bwd, best_tag, best_err


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, dataloader, device, args):
    model.eval()
    os.makedirs(args.vis_dir, exist_ok=True)
    results = []
    conv_log_path = os.path.join(args.vis_dir, "chosen_conventions.txt")
    with open(conv_log_path, "w") as _f:
        _f.write("sample,tag,err,oob_rate,median_mag,max_mag\n")

    mean = torch.tensor(args.img_mean, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(args.img_std,  device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for i, (img0, img4, roma_flow, gt_flow, sample_id) in enumerate(tqdm(dataloader)):
            img0, img4 = img0.to(device), img4.to(device)
            roma_flow, gt_flow = roma_flow.to(device), gt_flow.to(device)
            B, C, H, W = img0.shape

            # Resize flows to image size with correct semantics
            if roma_flow.shape[-2:] != (H, W):
                roma_flow = resize_flow_by_format(roma_flow, args.pred_format, (H, W))
            if gt_flow.shape[-2:] != (H, W):
                gt_flow = resize_flow_by_format(gt_flow, args.gt_format, (H, W))

            # Predict flow (same format as args.pred_format)
            pred_flow = model(img0, img4, roma_flow)

            # Convert to FORWARD OFFSETS (pixels) for metrics
            pred_fwd_px = to_forward_offsets_pixels(pred_flow, args.pred_format, H, W)
            gt_fwd_px   = to_forward_offsets_pixels(gt_flow,   args.gt_format,   H, W)

            # Build BACKWARD OFFSETS (pixels) for warping
            pred_bwd_px = invert_forward_to_backward(pred_fwd_px, iters=args.invert_iters)

            # ---- Diagnostics (before auto-select) ----
            base = pixel_grid(B, H, W, device)
            src_coords = base + pred_bwd_px
            oob = (
                (src_coords[:, 0] < 0) | (src_coords[:, 0] > W - 1) |
                (src_coords[:, 1] < 0) | (src_coords[:, 1] > H - 1)
            ).float().mean()
            mag = pred_bwd_px.norm(dim=1)
            med_mag, max_mag = float(mag.median()), float(mag.max())

            # ---- Auto-select best convention & warp ----
            warped_img0_to_4, chosen_bwd, tag, best_err = choose_best_backward_and_warp(
                img0, img4, pred_bwd_px, pred_fwd_px
            )

            # ---- Metrics ----
            epe_map = torch.norm(pred_fwd_px - gt_fwd_px, dim=1)  # [B,H,W]
            epe = float(epe_map.mean())
            ecc = float((1 - F.cosine_similarity(pred_fwd_px, gt_fwd_px, dim=1)).mean())

            # ---- Save visualizations ----
            if isinstance(sample_id, (list, tuple)):
                sample_name = str(sample_id[0])
            else:
                sample_name = str(sample_id)

            # Denorm if needed (for saving only)
            warped_to_save = denorm01_if_needed(warped_img0_to_4, mean, std)
            img0_to_save   = denorm01_if_needed(img0, mean, std)
            img4_to_save   = denorm01_if_needed(img4, mean, std)

            warped_path = os.path.join(args.vis_dir, f"{sample_name}_warped.png")
            img0_path   = os.path.join(args.vis_dir, f"{sample_name}_img0.png")
            img4_path   = os.path.join(args.vis_dir, f"{sample_name}_img4.png")
            ensure_dir(warped_path)
            save_image(warped_to_save.cpu(), warped_path)
            save_image(img0_to_save.cpu(), img0_path)
            save_image(img4_to_save.cpu(), img4_path)

            # Save forward-offset flow visualizations for apples-to-apples viewing
            pred_fwd_img = os.path.join(args.vis_dir, f"{sample_name}_pred_fwd.png")
            gt_fwd_img   = os.path.join(args.vis_dir, f"{sample_name}_gt_fwd.png")
            save_flow_as_image(pred_fwd_px[0], pred_fwd_img)
            save_flow_as_image(gt_fwd_px[0],   gt_fwd_img)

            # Also save the chosen backward flow used for warping
            chosen_bwd_img = os.path.join(args.vis_dir, f"{sample_name}_chosen_bwd.png")
            save_flow_as_image(chosen_bwd[0], chosen_bwd_img)

            results.append({
                "sample": sample_name,
                "epe": epe,
                "ecc": ecc,
                "warped_img": warped_path,
                "pred_flow_fwd_img": pred_fwd_img,
                "gt_flow_fwd_img": gt_fwd_img,
                "chosen_flow_bwd_img": chosen_bwd_img,
                "chosen_convention": tag,
                "oob_rate": float(oob),
                "median_flow_mag": med_mag,
                "max_flow_mag": max_mag,
            })

            # Log convention & diagnostics
            with open(conv_log_path, "a") as f:
                f.write(f"{sample_name},{tag},{best_err:.6f},{float(oob):.6f},{med_mag:.3f},{max_mag:.3f}\n")

    # Save CSV
    if args.csv_path and len(results) > 0:
        ensure_dir(args.csv_path)
        with open(args.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Saved visualizations to: {args.vis_dir}")
    if args.csv_path:
        print(f"Saved metrics to: {args.csv_path}")
    print(f"Convention log: {conv_log_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--split_path", type=str, default="splits", help="Path to split file/folder")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vis_dir", type=str, default="flow_vis", help="Directory to save images")
    parser.add_argument("--csv_path", type=str, default="eval_results.csv")

    # Flow format flags (set these to match your data/model)
    parser.add_argument("--pred_format", type=str, default="abs_norm_forward",
                        choices=["abs_norm_forward", "abs_px_forward", "fwd_px", "bwd_px"],
                        help="Format of PREDICTED flow from the model.")
    parser.add_argument("--gt_format", type=str, default="abs_norm_forward",
                        choices=["abs_norm_forward", "abs_px_forward", "fwd_px", "bwd_px"],
                        help="Format of GT flow from the dataset.")

    parser.add_argument("--invert_iters", type=int, default=12,
                        help="Iterations for forward->backward inversion")

    # For saving images correctly if tensors are normalized
    parser.add_argument("--img_mean", nargs=3, type=float, default=[0.485, 0.456, 0.406],
                        help="Per-channel mean used during preprocessing (for de-normalization on save).")
    parser.add_argument("--img_std",  nargs=3, type=float, default=[0.229, 0.224, 0.225],
                        help="Per-channel std used during preprocessing (for de-normalization on save).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    dataset = ResidualFlowDataset(args.data_dir, split=args.split, split_file=args.split_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Model & weights
    model = ResidualRefinerNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    evaluate(model, loader, device, args)


if __name__ == "__main__":
    main()
