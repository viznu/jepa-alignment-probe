"""Extract per-layer residual-stream trajectories at the last prompt token.

Output tensor shape: (N, L, d)
  N = number of prompts
  L = number of transformer layers
  d = hidden dim
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract(model_name: str, prompts: list[str], device: str, dtype: torch.dtype,
            output_path: Path, max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    layers = model.model.layers
    L = len(layers)
    d = model.config.hidden_size
    print(f"model={model_name}  layers={L}  hidden={d}  device={device}  dtype={dtype}")

    captured = [None] * L
    handles = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[:, -1, :].detach().to(torch.float32).cpu()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))

    N = len(prompts)
    trajectories = torch.zeros(N, L, d, dtype=torch.float32)

    with torch.inference_mode():
        for i, prompt in enumerate(tqdm(prompts, desc="extracting")):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            model(**inputs)
            for l in range(L):
                trajectories[i, l] = captured[l][0]

    for h in handles:
        h.remove()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trajectories": trajectories,
        "prompts": prompts,
        "model_name": model_name,
        "num_layers": L,
        "hidden_dim": d,
        "token_position": "last_prompt_token",
    }, output_path)
    print(f"saved shape={tuple(trajectories.shape)} -> {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument("--output", default="data/activations_iter1_smoke.pt")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    ds = load_dataset(args.dataset, split="train")
    prompts = [x["instruction"] for x in ds.shuffle(seed=args.seed).select(range(args.n_prompts))]

    extract(args.model, prompts, device, dtype, Path(args.output), args.max_len)


if __name__ == "__main__":
    main()
