#!/usr/bin/env python3
# copy_npy_to_named_folders.py
import argparse, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing .npy files (searched recursively)")
    ap.add_argument("--dst", required=True, help="Destination root to create per-file folders")
    ap.add_argument("--pattern", default="*.npy", help="Glob pattern (default: *.npy)")
    ap.add_argument("--name", default="roma_flow.npy",
                    help="Output filename per sample folder (use gt_flow.npy for ground-truth flow).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite the target file if it exists")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = list(src.rglob(args.pattern))
    if not files:
        print("No .npy files found."); return

    copied, skipped = 0, 0
    for p in files:
        if not p.is_file(): 
            continue
        out_dir = dst / p.stem  # folder named after the file (without .npy)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.name

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        shutil.copy2(p, out_path)
        copied += 1

    print(f"Done. Copied: {copied}, skipped: {skipped}. Output root: {dst}")

if __name__ == "__main__":
    main()
