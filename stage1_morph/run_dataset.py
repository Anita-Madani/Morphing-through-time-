"""
Stage 1 driver — run diffusion morphing over a change-detection dataset.

For every aligned pair (A/<name>.png, B/<name>.png) it calls morph_pair.py to
synthesize a short morph sequence (00.png ... 04.png) into <out>/<name>/.

The "after" images are typically the affine-perturbed views (e.g. B_affine),
so the morph bridges the geometric + temporal gap between the two acquisitions.

Example (LEVIR-CD):
    python run_dataset.py \
        --dir_a /path/LEVIR-CD256/train/A \
        --dir_b /path/LEVIR-CD256/train/B_affine \
        --out   outputs/levir_morph \
        --prompt_0 "satellite photo of a neighborhood before new buildings were built" \
        --prompt_1 "satellite photo of the same neighborhood with newly constructed buildings"
"""
import os
import sys
import argparse
from glob import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True, help="Directory of 'before' images (A).")
    ap.add_argument("--dir_b", required=True, help="Directory of 'after' images (B / B_affine).")
    ap.add_argument("--out", required=True, help="Output root; one folder of morph frames per pair.")
    ap.add_argument("--prompt_0", default="a satellite image", help="Text prompt for image A.")
    ap.add_argument("--prompt_1", default="a satellite image", help="Text prompt for image B.")
    ap.add_argument("--num_frames", type=int, default=5, help="Frames per morph (default 5 -> 00..04).")
    ap.add_argument("--model_path", default="stabilityai/stable-diffusion-2-1-base")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    morph_pair = os.path.join(here, "morph_pair.py")

    paths_a = sorted(glob(os.path.join(args.dir_a, "*.png")))
    paths_b = sorted(glob(os.path.join(args.dir_b, "*.png")))
    if len(paths_a) != len(paths_b):
        print(f"[warn] A has {len(paths_a)} images, B has {len(paths_b)}; zipping to the shorter.")

    for img_a, img_b in zip(paths_a, paths_b):
        name = os.path.splitext(os.path.basename(img_a))[0]
        out_dir = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out_dir, f"{args.num_frames - 1:02d}.png")):
            continue  # already morphed
        cmd = (
            f'{args.python} "{morph_pair}" '
            f'--model_path "{args.model_path}" '
            f'--image_path_0 "{img_a}" --image_path_1 "{img_b}" '
            f'--prompt_0 "{args.prompt_0}" --prompt_1 "{args.prompt_1}" '
            f'--output_path "{out_dir}" --num_frames {args.num_frames} '
            f'--use_adain --use_reschedule --save_inter'
        )
        os.system(cmd)


if __name__ == "__main__":
    main()
