"""Extract labeled trajectories: AdvBench harmful (label=1) + Alpaca benign (label=0).

Output: (N, L, d) trajectories, (N,) int labels, prompt strings, source tags.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from src.phase1.extract_activations import run_extraction


def collect_prompts(n_benign: int, n_harmful: int, seed: int) -> tuple[list[str], list[int], list[str]]:
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign = [x["instruction"] for x in alpaca.shuffle(seed=seed).select(range(n_benign))]

    adv = load_dataset("AlignmentResearch/AdvBench", split="train")
    adv_shuf = adv.shuffle(seed=seed).select(range(min(n_harmful, len(adv))))
    harmful = [x["content"][0] for x in adv_shuf]

    prompts = benign + harmful
    labels = [0] * len(benign) + [1] * len(harmful)
    sources = ["alpaca"] * len(benign) + ["advbench"] * len(harmful)
    return prompts, labels, sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n_benign", type=int, default=100)
    p.add_argument("--n_harmful", type=int, default=100)
    p.add_argument("--output", default="data/labeled_smoke.pt")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    prompts, labels, sources = collect_prompts(args.n_benign, args.n_harmful, args.seed)
    print(f"benign={args.n_benign}  harmful={args.n_harmful}  total={len(prompts)}")

    traj, L, d = run_extraction(args.model, prompts, device, dtype, args.max_len)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trajectories": traj,
        "labels": torch.tensor(labels, dtype=torch.long),
        "prompts": prompts,
        "sources": sources,
        "model_name": args.model,
        "num_layers": L,
        "hidden_dim": d,
        "token_position": "last_prompt_token",
    }, out)
    print(f"saved shape={tuple(traj.shape)} labels={len(labels)} -> {out}")


if __name__ == "__main__":
    main()
