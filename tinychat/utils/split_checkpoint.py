"""Split a single checkpoint (.safetensors or .pt) into per-key .pt files.

This produces a folder containing files named exactly like the model's
state_dict keys with a `.pt` suffix, which matches what
`mem_efficient_load_checkpoint` expects.

Usage:
  python split_checkpoint.py --input /path/to/model.safetensors --outdir /path/to/shards/
"""

import argparse
import gc
import os
from pathlib import Path

import torch
from tqdm import tqdm


def load_checkpoint(path: Path):
    p = str(path)
    if p.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load

            sd = safe_load(p)
            return sd
        except Exception as e:
            raise RuntimeError(f"Failed to load safetensors file: {e}")
    else:
        # torch.load may return a bare state_dict or a wrapped dict
        data = torch.load(p, map_location="cpu")
        # common wrappers
        if isinstance(data, dict):
            for candidate in ("state_dict", "model_state_dict", "model", "state"):
                if candidate in data and isinstance(data[candidate], dict):
                    return data[candidate]
            # If it already looks like a state_dict (tensor values), return as-is
            return data
        raise RuntimeError("Unsupported checkpoint format")


def sanitize_filename(key: str) -> str:
    # Keys typically contain dots and alphanumerics which are valid in filenames.
    # Keep it simple: replace os.sep if present (shouldn't be) and strip leading/trailing spaces.
    return key.replace(os.sep, "_").strip()


def split_checkpoint(input_path: Path, outdir: Path, overwrite: bool = False):
    input_path = input_path.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    state_dict = load_checkpoint(input_path)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Loaded checkpoint is not a state dict mapping")

    keys = list(state_dict.keys())
    print(f"Loaded checkpoint with {len(keys)} keys. Saving to: {outdir}")

    for k in tqdm(keys, desc="Saving shards"):
        val = state_dict[k]
        # Move to CPU if it's a tensor
        try:
            if hasattr(val, "cpu"):
                val = val.cpu()
        except Exception:
            pass

        filename = sanitize_filename(k) + ".pt"
        outpath = outdir / filename
        if outpath.exists() and not overwrite:
            tqdm.write(f"Skipping existing file: {outpath}")
            continue
        # Save single tensor (or object) to its own file
        torch.save(val, str(outpath))
        # Free memory
        del val
        gc.collect()

    print("Done. You can now point --llm-checkpoint to the shard folder.")


def parse_args():
    p = argparse.ArgumentParser(description="Split a checkpoint into per-key .pt files for layerwise loading")
    p.add_argument("--input", "-i", required=True, help="Path to input checkpoint (.safetensors or .pt)")
    p.add_argument("--outdir", "-o", required=True, help="Folder where to write per-key .pt files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files if present")
    return p.parse_args()


def main():
    args = parse_args()
    split_checkpoint(Path(args.input), Path(args.outdir), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
