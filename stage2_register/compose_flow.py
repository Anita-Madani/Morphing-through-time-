"""
Stage 2 — Stepwise RoMa registration + flow composition.

For each morph sequence produced by Stage 1 (a folder with frames 00.png..0N.png),
RoMa estimates the dense warp between consecutive frames and the warps are COMPOSED
into a single 00 -> 0N correspondence. The composed field is:

  * converted to a pixel-space flow and saved as <morph_dir>/<sample>/roma_flow.npy
    (shape HxWx2) so it can feed the Stage-3 residual refiner, and
  * optionally evaluated (EPE / ECE) against ground-truth flow if --gt_flow_dir is given.

RoMa must be importable (`pip install` from https://github.com/Parskatt/RoMa).

Example:
    python compose_flow.py --morph_dir ../stage1_morph/outputs/levir_morph \
                           --out_dir   warped_levir \
                           --gt_flow_dir /path/LEVIR-CD256/eval/flow_gt_npy   # optional
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from romatch import roma_outdoor


def load_image_tensor(path, H, W, device):
    img = Image.open(path).convert("RGB").resize((H, W))
    return ToTensor()(img).unsqueeze(0).to(device)


def get_warp_grid(roma, img1_path, img2_path, device):
    warp, _ = roma.match(img1_path, img2_path, device=device)
    _, full_W, _ = warp.shape
    W = full_W // 2
    return warp[:, W:, :2].contiguous().to(dtype=torch.float32)


def compose_grids(grid1, grid2):
    if grid1.dim() == 3:
        grid1 = grid1.unsqueeze(0)
    if grid2.dim() == 3:
        grid2 = grid2.unsqueeze(0)
    grid2_sample = grid2.permute(0, 3, 1, 2)
    grid2_composed = F.grid_sample(grid2_sample, grid1, align_corners=True)
    return grid2_composed.permute(0, 2, 3, 1)


def normalized_to_pixel_flow(grid, H, W):
    base_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), [1, 1, H, W], align_corners=True).to(grid.device)
    flow = (grid - base_grid) * torch.tensor([W / 2, H / 2], device=grid.device)
    return flow[0].cpu().numpy()   # [H, W, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph_dir", required=True, help="Stage-1 output root (folders of 00..0N frames).")
    ap.add_argument("--out_dir", required=True, help="Where composed warp visualizations / CSV go.")
    ap.add_argument("--gt_flow_dir", default=None, help="Optional GT flow (.npy per sample) for EPE/ECE.")
    ap.add_argument("--frames", type=int, default=5, help="Frames per morph sequence (default 5).")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    H = W = args.size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    roma = roma_outdoor(device=device)
    roma.upsample_res = (H, W)

    folders = sorted(f for f in os.listdir(args.morph_dir) if os.path.isdir(os.path.join(args.morph_dir, f)))
    results = []

    for folder in folders:
        try:
            fpath = os.path.join(args.morph_dir, folder)
            composed_warp, first_img = None, None

            for i in range(args.frames - 1):
                im1 = os.path.join(fpath, f"{i:02d}.png")
                im2 = os.path.join(fpath, f"{i + 1:02d}.png")
                if not (os.path.exists(im1) and os.path.exists(im2)):
                    raise FileNotFoundError(f"Missing frame: {im1} or {im2}")
                if i == 0:
                    first_img = load_image_tensor(im1, H, W, device)
                warp = get_warp_grid(roma, im1, im2, device)
                if warp.dim() == 3:
                    warp = warp.unsqueeze(0)
                composed_warp = warp if composed_warp is None else compose_grids(composed_warp, warp)

            # Save the composed warp visualization
            final_img = F.grid_sample(first_img, composed_warp, align_corners=True)
            save_image(final_img, os.path.join(args.out_dir, f"{folder}_composed.png"))

            # Save the composed pixel flow next to the frames (feeds Stage 3)
            pred_flow = normalized_to_pixel_flow(composed_warp, H, W)   # [H, W, 2]
            np.save(os.path.join(fpath, "roma_flow.npy"), pred_flow.astype(np.float32))

            # Optional evaluation against GT flow
            if args.gt_flow_dir:
                gt_path = os.path.join(args.gt_flow_dir, f"{folder}.npy")
                if os.path.exists(gt_path):
                    pred_t = torch.tensor(pred_flow).permute(2, 0, 1).unsqueeze(0).to(device)
                    gt_t = torch.tensor(np.load(gt_path)).unsqueeze(0).to(device)
                    gt_t = F.interpolate(gt_t, size=(H, W), mode="bilinear", align_corners=False)
                    epe = torch.norm(pred_t - gt_t, dim=1).mean().item()
                    ece = (1 - F.cosine_similarity(pred_t, gt_t, dim=1)).mean().item()
                    results.append({"folder": folder, "EPE": epe, "ECE": ece})
                    print(f"{folder}: EPE={epe:.4f} ECE={ece:.4f}")
            else:
                print(f"{folder}: composed flow saved")

        except Exception as e:
            print(f"[error] {folder}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.out_dir, "roma_eval_results.csv"), index=False)
        print(f"Mean EPE: {df['EPE'].mean():.4f} | Mean ECE: {df['ECE'].mean():.4f}")


if __name__ == "__main__":
    main()
