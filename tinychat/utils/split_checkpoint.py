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


# NOTE: We intentionally do not sanitize the keys into arbitrary filenames
# because the loader expects file paths that match the original state_dict
# keys as closely as possible. We will use the raw key to build the on-disk
# path (appending ".pt" if missing) and create parent directories when
# necessary. This preserves exact naming and avoids missing files.


def split_checkpoint(input_path: Path, outdir: Path, overwrite: bool = False):
    input_path = input_path.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    state_dict = load_checkpoint(input_path)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Loaded checkpoint is not a state dict mapping")

    # Handle common tied-weight conventions: some checkpoints store only the
    # token embedding under e.g. 'model.embed_tokens.weight' while the model
    # expects a separate 'lm_head.weight' key. If lm_head.weight is missing
    # but we have the embedding, create an alias so the split produces a
    # corresponding 'lm_head.weight.pt' file.
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        try:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
            print("Added missing key 'lm_head.weight' from 'model.embed_tokens.weight'")
        except Exception:
            # Non-fatal: continue without adding
            pass

    keys = list(state_dict.keys())
    print(f"Loaded checkpoint with {len(keys)} keys. Saving to: {outdir}")
    saved = 0
    skipped = 0
    for k in tqdm(keys, desc="Saving shards"):
        val = state_dict[k]
        # Move to CPU if it's a tensor
        try:
            if isinstance(val, torch.Tensor):
                val = val.cpu()
        except Exception:
            pass

        # Use the raw key name as filename (append .pt if missing).
        # If the key contains path separators, Path will create nested dirs.
        filename = k if k.endswith(".pt") else k + ".pt"
        outpath = outdir / Path(filename)

        # Create parent directories if needed
        outpath.parent.mkdir(parents=True, exist_ok=True)

        if outpath.exists() and not overwrite:
            tqdm.write(f"Skipping existing file: {outpath}")
            skipped += 1
            continue

        # Save single tensor (or object) to its own file
        try:
            torch.save(val, str(outpath))
        except Exception as e:
            tqdm.write(f"Failed to save {outpath}: {e}")
            # continue to next key without incrementing saved
            continue

        saved += 1
        # Free memory
        del val
        gc.collect()

    print(f"Saved {saved} files; skipped {skipped} existing files.")

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
