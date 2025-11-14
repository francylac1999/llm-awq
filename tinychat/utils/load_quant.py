import gc
import os
import re
from typing import Union, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.quantize.quantizer import real_quantize_model_weight
from awq.quantize.qmodule import WQLinear
from tqdm import tqdm

import tinychat.utils.constants

version_message = """
[Warning] The awq quantized checkpoint seems to be in v1 format. 
If the model cannot be loaded successfully, please use the latest awq library to re-quantized the model, or repack the current checkpoint with tinychat/offline-weight-repacker.py
"""


def ckpt_version_check(quant_path):
    if not quant_path.endswith("v2.pt"):
        print(version_message)



def mem_efficient_load_checkpoint(
    model: nn.Module,
    ckpts_folder: Union[str, os.PathLike],
):
    checkpoint_files = [
        ckpts_folder + "/" + f for f in os.listdir(ckpts_folder) if f.endswith(".pt")
    ]

    # Prepare model keys and a mapping of available checkpoint files
    model_keys = list(model.state_dict().keys())
    # map from key -> filepath if present
    ckpt_map = {
        os.path.splitext(f)[0]: os.path.join(ckpts_folder, f)
        for f in os.listdir(ckpts_folder)
        if f.endswith(".pt")
    }

    missing = [k for k in model_keys if k not in ckpt_map]
    extra = [k for k in ckpt_map.keys() if k not in model_keys]
    if missing:
        print(f"Warning: {len(missing)} model keys are missing in checkpoint folder. Missing keys sample: {missing[:10]}")
    if extra:
        print(f"Note: {len(extra)} extra checkpoint files were found that don't match model keys. Extra sample: {extra[:10]}")

    # Heuristic: detect AWQ / quantized shard formats (qweight/scales/scaled_zeros)
    # If present, abort with an informative error because mem_efficient_load_checkpoint
    # is intended to load per-key float tensors (one .pt per state_dict key).
    awq_indicators = (".qweight", ".scales", ".scaled_zeros", "qweight", "scales", "scaled_zeros")
    found_awq = any(any(ind in name for ind in awq_indicators) for name in extra)
    if found_awq:
        raise RuntimeError(
            "Checkpoint folder appears to contain quantized AWQ shards (qweight/scales/...).\n"
            "mem_efficient_load_checkpoint expects a folder of per-key float .pt files (one file per state_dict key).\n"
            "If you have an AWQ quantized checkpoint, load it with the AWQ loader (e.g. load_awq_model) or provide the non-quantized per-key shards.\n"
        )

    # Load files in the order of model_keys so load_state_dict updates correctly
    total_to_load = len([k for k in model_keys if k in ckpt_map])
    with tqdm(total=total_to_load) as pbar:
        pbar.set_description("Loading checkpoint shards")
        for key in model_keys:
            if key not in ckpt_map:
                continue
            checkpoint_file = ckpt_map[key]
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
            # If the shard file contains a single tensor (common when we
            # split the state_dict into per-key files), wrap it into a
            # dict mapping the expected key name -> tensor so
            # load_state_dict accepts it.
            if isinstance(checkpoint, torch.Tensor):
                checkpoint = {key: checkpoint}
            elif not isinstance(checkpoint, dict):
                # try to coerce mapping-like objects
                try:
                    checkpoint = dict(checkpoint)
                except Exception:
                    raise TypeError(
                        f"Unsupported shard type {type(checkpoint)} for {checkpoint_file}."
                    )
            model.load_state_dict(checkpoint, strict=False)
            # Force Python to clean up.
            del checkpoint
            gc.collect()
            pbar.update(1)
    if missing:
        print("Warning: model may be incomplete due to missing keys in checkpoint folder.")
    return model

def load_non_quantized_model(model, checkpoint, device):
    if hasattr(model.config, "tie_encoder_decoder"):
        model.config.tie_encoder_decoder = False
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    if tinychat.utils.constants.mem_efficient_load:
        assert os.path.isdir(
            checkpoint
        ), "You are in mem_efficient_load mode. \n Please set --load_quant the path to the folder containing all checkpoint files."
        model = mem_efficient_load_checkpoint(
            model,
            checkpoint,
        ).to(device)
    else:
        ckpt_version_check(checkpoint)
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint")
        for i in pbar:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint,
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                    "CLIPEncoderLayer",
                ],
            ).to(device)
    return model

def make_quant_linear(module, names, w_bit, groupsize, device, name=""):
    if isinstance(module, WQLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            delattr(module, attr)
            setattr(
                module,
                attr,
                WQLinear(
                    w_bit,
                    groupsize,
                    tmp.in_features,
                    tmp.out_features,
                    tmp.bias is not None,
                    device,
                ),
            )
    for name1, child in module.named_children():
        make_quant_linear(
            child,
            names,
            w_bit,
            groupsize,
            device,
            name + "." + name1 if name != "" else name1,
        )


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def load_awq_llama_fast(model, checkpoint, w_bit, group_size, device):
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant_linear(model, layers, w_bit, group_size, device)
    del layers

    if tinychat.utils.constants.mem_efficient_load:
        # TODO: mem-efficient load for llama
        assert os.path.isdir(
            checkpoint
        ), "You are in mem_efficient_load mode. \n Please set --load_quant the path to the folder containing all checkpoint files."
        model = mem_efficient_load_checkpoint(
            model,
            checkpoint,
        )
    else:
        ckpt_version_check(checkpoint)
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint")
        for i in pbar:
            if checkpoint.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load

                model.load_state_dict(safe_load(checkpoint))
            else:
                model.load_state_dict(torch.load(checkpoint))

    return model.to(device)
