import torch
from datasets import load_dataset
import json

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def get_calib_dataset_from_json(json_path, tokenizer=None, n_samples=512, block_size=512):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # raw_data dovrebbe essere una lista di dict con chiave "text"
    dataset = [
        conv["value"] for item in raw_data for conv in item.get("conversations", []) if conv["from"] == "gpt"
    ]    
    print(dataset[:5])  # Stampa i primi 5 elementi per debug
    samples = []
    n_run = 0
    for line in dataset:
        line = line.strip()
        print(line)
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    if not samples:
        raise ValueError("Nessun campione valido trovato.")

    # Concatena i token e suddividi in blocchi
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size]
        for i in range(n_split)
    ]
