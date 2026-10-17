# GRPO From Scratch

A clean, fully documented implementation of **Group Relative Policy Optimization (GRPO)** for training language models on mathematical reasoning tasks.

> **Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (Shao et al., 2024)

---

## What is GRPO?

GRPO is a reinforcement learning algorithm for language model fine-tuning. It is a memory-efficient alternative to PPO that **eliminates the critic network** by computing advantages from group statistics.

**The core insight:**

For each prompt, generate **G completions** (a "group"). The advantage of completion *i* is:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

The model learns: *"was this output better or worse than my other attempts at the same prompt?"*

### GRPO vs PPO

| | PPO | GRPO |
|---|---|---|
| Advantage source | Learned value network | Group reward statistics |
| Memory | 2× policy size (policy + critic) | 1× policy size |
| Critic training | Required | Not needed |
| Best for | General RL | Verifiable rewards (math, code) |

---

## Repository Structure

```
grpo-from-scratch/
├── grpo/
│   ├── algorithm.py   # Core math: advantages, loss, KL penalty
│   ├── trainer.py     # Full training loop
│   ├── rewards.py     # Composable reward functions
│   └── __init__.py
├── scripts/
│   └── train_gsm8k.py # End-to-end training on GSM8K
├── experiments/       # Put your experiment configs here
├── tests/
└── README.md
```

---

## Quickstart

### Install

```bash
git clone https://github.com/YOUR_USERNAME/grpo-from-scratch
cd grpo-from-scratch
pip install -r requirements.txt
```

### Run (GPT-2, proof of concept)

```bash
python scripts/train_gsm8k.py \
  --model gpt2 \
  --steps 200 \
  --group_size 4 \
  --batch_size 2
```

### Run (Qwen2.5-Math, real research)

```bash
python scripts/train_gsm8k.py \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --steps 500 \
  --group_size 8 \
  --batch_size 4 \
  --kl_coeff 0.04
```

---

## The Algorithm, Step by Step

### Step 1 — Sample a prompt batch

```python
batch_prompts = random.sample(prompts, batch_size)
```

### Step 2 — Generate G completions per prompt (rollout)

```python
# Each prompt generates G completions → B*G total outputs
output = policy.generate(prompt_ids.repeat_interleave(G), ...)
```

### Step 3 — Score with reward function

```python
rewards = reward_fn(prompts_repeated, completions)  # [B*G]
```

For math: **1.0** if extracted answer matches ground truth, **0.0** otherwise.

### Step 4 — Compute GRPO advantages

```python
# Reshape [B*G] → [B, G], normalize within each group
advantages = (rewards_grouped - mean) / (std + eps)  # [B, G]
```

### Step 5 — Compute policy gradient loss

```python
# PPO clipped surrogate loss
ratio = exp(log_pi_new - log_pi_old)
loss  = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

# KL penalty (keeps policy near reference model)
loss += β * KL(π_current || π_ref)
```

### Step 6 — Update policy

```python
loss.backward()
optimizer.step()
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `group_size` G | 8 | Completions per prompt. Larger G → better advantage estimates |
| `clip_epsilon` | 0.2 | PPO clipping range. Smaller = more conservative updates |
| `kl_coeff` β | 0.04 | KL penalty weight. Higher = stays closer to reference model |
| `temperature` | 0.9 | Generation diversity. Lower = more greedy |
| `learning_rate` | 1e-6 | Very small — GRPO updates are sensitive |

---

## Reward Functions

The `grpo/rewards.py` module contains composable reward functions:

```python
from grpo.rewards import RewardComposer, exact_match_reward, format_reward

composer = RewardComposer([
    (exact_match_reward, 1.0),   # Correct answer: +1.0
    (format_reward,      0.2),   # Correct format: +0.2
])
```

You can add custom reward functions by matching the signature:
```python
def my_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    ...
```

---

## Results

> Experiments are logged to `checkpoints/training_metrics.json`.

| Model | Dataset | Steps | Final Reward |
|---|---|---|---|
| GPT-2 | GSM8K (demo) | 200 | ~0.05 (expected — GPT-2 cannot do math) |
| Qwen2.5-Math-1.5B | GSM8K | 500 | TBD |

*GPT-2 results are included only to verify the training loop runs correctly. Use Qwen2.5-Math for meaningful results.*

---

## Citation

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```
