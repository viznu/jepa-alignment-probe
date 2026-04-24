"""Cross-task extraction: AdvBench harmful prompt + prefilled refuse or comply.

Each harmful prompt yields two paired trajectories:
  label 0 (aligned / refuse):   prefilled with a generic refusal.
  label 1 (misaligned / comply): prefilled with a generic compliance opener.

The prefix differs only in the assistant's message; the user turn (the
harmful request) is identical within a pair. Probe at the last prefilled
token, same as Instructed-Pairs.

This is the Apollo-style structure (same prompt, two conditions) but on
a different behavioral axis (refuse vs comply) and different content
(harmful requests rather than factual statements).
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REFUSE = "I'm sorry, but I can't help with that request."
COMPLY = "Sure, here's how to accomplish that:"


def build_prompt(tokenizer, user_turn: str, assistant_prefill: str,
                 drop_last_n: int) -> str:
    prefill_ids = tokenizer(assistant_prefill, add_special_tokens=False).input_ids
    kept = prefill_ids[: max(1, len(prefill_ids) - drop_last_n)]
    truncated = tokenizer.decode(kept)
    messages = [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": truncated},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


def extract(model_name: str, prompts: list[str], device: str, dtype: torch.dtype,
            drop_last_n: int, max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    layers = model.model.layers
    L = len(layers)
    d = model.config.hidden_size
    print(f"model={model_name}  layers={L}  hidden={d}  device={device}")

    captured = [None] * L
    handles = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[:, -1, :].detach().to(torch.float32).cpu()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))

    conditions = [(REFUSE, 0), (COMPLY, 1)]
    N = len(prompts) * len(conditions)
    traj = torch.zeros(N, L, d, dtype=torch.float32)
    labels = torch.zeros(N, dtype=torch.long)
    prompt_ids = [0] * N
    rendered: list[str] = []

    i = 0
    with torch.inference_mode():
        for pid, user_turn in enumerate(tqdm(prompts, desc="prompts")):
            for prefill, lab in conditions:
                text = build_prompt(tokenizer, user_turn, prefill, drop_last_n)
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=max_len, add_special_tokens=False).to(device)
                model(**inputs)
                for l in range(L):
                    traj[i, l] = captured[l][0]
                labels[i] = lab
                prompt_ids[i] = pid
                rendered.append(text)
                i += 1

    for h in handles:
        h.remove()
    return traj, labels, prompt_ids, rendered, L, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument("--output", default="data/refusal_pairs.pt")
    p.add_argument("--drop_last_n", type=int, default=5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    ds = load_dataset("AlignmentResearch/AdvBench", split="train")
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=gen).tolist()
    prompts = [ds[j]["content"][0] for j in perm[: args.n_prompts]]
    print(f"loaded {len(prompts)} AdvBench harmful prompts")

    traj, labels, prompt_ids, rendered, L, d = extract(
        args.model, prompts, device, dtype, args.drop_last_n, args.max_len,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Naming: fact_ids kept as the field name so transfer_test's group_field
    # default works unchanged (refusal_pairs is grouped by source prompt).
    torch.save({
        "trajectories": traj,
        "labels": labels,
        "fact_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "prompts": rendered,
        "user_turns": prompts,
        "model_name": args.model,
        "num_layers": L,
        "hidden_dim": d,
        "token_position": "last_prefilled_token",
        "refuse_prefill": REFUSE,
        "comply_prefill": COMPLY,
        "drop_last_n": args.drop_last_n,
    }, out)
    print(f"saved shape={tuple(traj.shape)} labels={len(labels)} prompts={len(prompts)} -> {out}")


if __name__ == "__main__":
    main()
