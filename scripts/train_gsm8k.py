"""
Train a small model on GSM8K using GRPO.
=========================================
This script demonstrates the full training pipeline using GPT-2 (small)
as a proof-of-concept. For real research, replace with Qwen2.5-Math or similar.

Run:
    python scripts/train_gsm8k.py --model gpt2 --steps 200 --group_size 4

For Qwen (recommended for real results):
    python scripts/train_gsm8k.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --steps 500 --group_size 8 --batch_size 4
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from grpo.algorithm import GRPOConfig
from grpo.trainer import GRPOTrainer
from grpo.rewards import RewardComposer, exact_match_reward, format_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve the problem step by step and end with '#### <answer>'."
)

def format_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k_prompts(split="train", max_samples=1000):
    """
    Load GSM8K and format as (prompt, ground_truth) pairs.
    GSM8K answers appear after '####' in the solution field.
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts, answers = [], []
    for item in dataset:
        prompt = format_prompt(item["question"])
        # Extract numeric answer from GSM8K format
        answer_str = item["answer"].split("####")[-1].strip()
        prompts.append(prompt)
        answers.append(answer_str)

    return prompts, answers


# ---------------------------------------------------------------------------
# Reward function (closure that captures ground truths)
# ---------------------------------------------------------------------------

def make_reward_fn(answer_map: dict):
    """
    Build a reward function that looks up ground truth by prompt.

    GRPO passes (prompts, completions) to reward_fn.
    We need ground truth — store it in a dict keyed by prompt.
    """
    composer = RewardComposer([
        (lambda p, c: exact_match_reward(p, c, [answer_map[pr] for pr in p]), 1.0),
        (format_reward, 0.2),
    ])

    def reward_fn(prompts, completions):
        return composer(prompts, completions)

    return reward_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="gpt2",          help="Model name or path")
    parser.add_argument("--steps",      type=int, default=200,   help="Training steps")
    parser.add_argument("--group_size", type=int, default=4,     help="G: completions per prompt")
    parser.add_argument("--batch_size", type=int, default=2,     help="Prompts per step")
    parser.add_argument("--kl_coeff",   type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument("--max_samples",type=int, default=500,   help="Max training prompts")
    parser.add_argument("--save_dir",   default="checkpoints",   help="Where to save")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {args.model}")

    # --- Load model and tokenizer ---
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    )
    # Reference model: same weights, never updated
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    )

    # --- Load data ---
    logger.info("Loading GSM8K dataset...")
    prompts, answers = load_gsm8k_prompts(max_samples=args.max_samples)
    answer_map = dict(zip(prompts, answers))
    logger.info(f"Loaded {len(prompts)} training prompts")

    # --- Config ---
    config = GRPOConfig(
        group_size=args.group_size,
        kl_coeff=args.kl_coeff,
        max_new_tokens=128,
        temperature=0.9,
    )

    # --- Trainer ---
    reward_fn = make_reward_fn(answer_map)
    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
        device=device,
    )

    # --- Training loop ---
    logger.info(f"Starting GRPO training for {args.steps} steps...")
    all_metrics = []

    for step, metrics in trainer.train(prompts, num_steps=args.steps, batch_size=args.batch_size):
        all_metrics.append(metrics)

        if step % 50 == 0:
            avg_reward = sum(m["reward/mean"] for m in all_metrics[-50:]) / 50
            avg_loss   = sum(m["loss/total"]  for m in all_metrics[-50:]) / 50
            logger.info(
                f"[Step {step}/{args.steps}] "
                f"avg_reward={avg_reward:.4f} avg_loss={avg_loss:.4f}"
            )

        if step % 100 == 0:
            save_path = Path(args.save_dir) / f"step_{step}"
            save_path.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # --- Save metrics ---
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.save_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
