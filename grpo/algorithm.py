"""
GRPO: Group Relative Policy Optimization — From Scratch
========================================================
Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
       (Shao et al., 2024) — https://arxiv.org/abs/2402.03300

What is GRPO?
-------------
GRPO is a reinforcement learning algorithm designed for language model fine-tuning.
It is a variant of PPO (Proximal Policy Optimization) that eliminates the need for
a separate critic/value network by instead computing advantages from GROUP statistics.

Key idea:
  For each prompt, generate G outputs (a "group").
  Compute a reward for each output.
  The advantage of output i = (reward_i - mean(rewards)) / std(rewards)

  This means the model learns: "was this output better or worse than my
  other attempts at the same prompt?" No critic needed.

Why is this better than PPO for LLMs?
  - PPO needs a value network (same size as policy) → 2x memory
  - GRPO replaces the critic with group statistics → much cheaper
  - Works well when reward is verifiable (math, code, etc.)

Comparison:
  PPO advantage  = r_t - V(s_t)           ← needs learned value function
  GRPO advantage = (r_i - mean_g) / std_g ← computed from group rewards

Training loop overview:
  1. Sample a batch of prompts
  2. For each prompt, generate G completions using the current policy
  3. Score each completion with a reward function
  4. Compute GRPO advantages from group statistics
  5. Compute clipped surrogate loss (same as PPO)
  6. Add KL penalty to prevent policy from drifting too far
  7. Update policy weights
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """
    All hyperparameters for a GRPO run.

    group_size (G):
        How many completions to generate per prompt.
        Larger G → better advantage estimates, but more compute.
        Typical values: 4, 8, 16.

    clip_epsilon:
        The PPO clipping range. Keeps policy updates small.
        ratio = pi_new(a|s) / pi_old(a|s)
        Loss clips at [1 - epsilon, 1 + epsilon].

    kl_coeff (beta):
        Weight on the KL divergence penalty.
        KL penalty = beta * KL(pi_new || pi_ref)
        Higher beta → policy stays closer to the reference (pre-GRPO) model.

    temperature:
        Sampling temperature for generation.
        Lower → more greedy; higher → more diverse.

    normalize_advantages:
        If True, z-score the advantages within each group.
        Almost always True — stabilizes training.

    entropy_coeff:
        Optional entropy bonus to encourage exploration.
        Usually 0 or very small (0.001).
    """
    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.04
    temperature: float = 0.9
    max_new_tokens: int = 256
    normalize_advantages: bool = True
    entropy_coeff: float = 0.0
    grad_clip: float = 1.0
    learning_rate: float = 1e-6


# ---------------------------------------------------------------------------
# Advantage Computation (the heart of GRPO)
# ---------------------------------------------------------------------------

def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute GRPO advantages from a flat tensor of rewards.

    GRPO replaces the PPO critic with group normalization:
        advantage_i = (reward_i - mean(group)) / (std(group) + eps)

    This is the ENTIRE innovation of GRPO over PPO for RL from verifiable rewards.

    Args:
        rewards:    Flat tensor of shape [batch_size * group_size].
                    Rewards for each (prompt, completion) pair.
        group_size: G — number of completions per prompt.
        normalize:  Whether to z-score within each group.
        eps:        Small constant for numerical stability.

    Returns:
        advantages: Same shape as rewards, [batch_size * group_size].

    Example:
        rewards    = [0, 1, 0, 1, 1, 0, 0, 0]   # 4 prompts, G=2
        group_size = 2
        groups     = [[0,1], [0,1], [1,0], [0,0]]
        means      = [0.5,   0.5,   0.5,   0.0 ]
        stds       = [0.5,   0.5,   0.5,   0.0 ]
        advantages = [[-1, 1], [-1, 1], [1, -1], [0, 0]]
    """
    # Reshape: [batch * G] → [batch, G]
    batch_size = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(batch_size, group_size)   # [B, G]

    if not normalize:
        # Advantage = reward - group mean (no std normalization)
        means = rewards_grouped.mean(dim=1, keepdim=True)    # [B, 1]
        advantages = rewards_grouped - means                  # [B, G]
    else:
        # Z-score within each group
        means = rewards_grouped.mean(dim=1, keepdim=True)    # [B, 1]
        stds  = rewards_grouped.std(dim=1, keepdim=True)     # [B, 1]
        advantages = (rewards_grouped - means) / (stds + eps) # [B, G]

    return advantages.view(-1)  # Flatten back to [batch * G]


# ---------------------------------------------------------------------------
# Token-level Log Probabilities
# ---------------------------------------------------------------------------

def compute_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for the response tokens only.

    In GRPO we only update the policy on the completion tokens, not the prompt.
    The response_mask marks which tokens are part of the completion.

    Args:
        logits:        [batch, seq_len, vocab_size] — model output logits
        input_ids:     [batch, seq_len]             — token ids
        response_mask: [batch, seq_len]             — 1 for response tokens, 0 for prompt

    Returns:
        log_probs: [batch, seq_len] — log p(token) for each position,
                   masked to 0 outside response tokens.

    Note:
        We shift logits/ids by 1 because at position t, the model predicts token t+1.
        logits[:, :-1] predicts input_ids[:, 1:]
    """
    # Shift: logits at position t predict the token at position t+1
    shift_logits = logits[:, :-1, :]          # [B, L-1, V]
    shift_labels = input_ids[:, 1:]           # [B, L-1]
    shift_mask   = response_mask[:, 1:]       # [B, L-1]

    # log_softmax for numerical stability over raw logits
    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]

    # Gather log probability of the actual token chosen at each position
    token_log_probs = log_probs_all.gather(
        dim=2,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                             # [B, L-1]

    # Zero out prompt tokens — we don't want to optimize those
    return token_log_probs * shift_mask       # [B, L-1]


def sequence_log_prob(token_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Sum token log probs over the response to get sequence log probability.

    log p(sequence) = sum_t log p(token_t)

    Args:
        token_log_probs: [batch, seq_len]
        mask:            [batch, seq_len] — 1 for response tokens

    Returns:
        seq_log_probs: [batch] — one scalar per sequence
    """
    shift_mask = mask[:, 1:]  # align with shifted token log probs
    return (token_log_probs * shift_mask).sum(dim=-1)


# ---------------------------------------------------------------------------
# KL Divergence Penalty
# ---------------------------------------------------------------------------

def compute_kl_penalty(
    current_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sequence KL divergence: KL(pi_current || pi_ref).

    KL penalty keeps the trained policy close to the reference (base) model.
    Without it, the model may find degenerate high-reward outputs that
    don't resemble natural language (reward hacking at the token level).

    KL(P || Q) ≈ sum_t [ log P(t) - log Q(t) ]
    (This is the "reverse KL" approximation used in PPO/GRPO)

    Args:
        current_log_probs: [batch, seq_len] — log probs from current policy
        ref_log_probs:     [batch, seq_len] — log probs from reference policy
        response_mask:     [batch, seq_len] — 1 for response tokens

    Returns:
        kl_per_sequence: [batch] — mean KL per token in response
    """
    shift_mask = response_mask[:, 1:].float()

    # Approximate KL per token
    kl_per_token = current_log_probs - ref_log_probs   # [B, L-1]

    # Mean over response tokens (not sum, to normalize for response length)
    response_lengths = shift_mask.sum(dim=-1).clamp(min=1)
    kl_per_sequence  = (kl_per_token * shift_mask).sum(dim=-1) / response_lengths

    return kl_per_sequence


# ---------------------------------------------------------------------------
# GRPO Loss
# ---------------------------------------------------------------------------

def grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the full GRPO loss for a batch.

    GRPO Loss = -E[ min(r*A, clip(r, 1-ε, 1+ε)*A) ] + β * KL(π || π_ref)

    where:
        r = π_current(a|s) / π_old(a|s)   (probability ratio)
        A = GRPO advantage (group-normalized reward)
        β = KL coefficient

    The first term is the PPO clipped surrogate objective.
    The second term penalizes drifting from the reference model.

    Args:
        current_log_probs: [B*G, L] — log probs from current policy (being updated)
        old_log_probs:     [B*G, L] — log probs from rollout policy (frozen snapshot)
        ref_log_probs:     [B*G, L] — log probs from reference policy (pre-GRPO model)
        advantages:        [B*G]    — GRPO advantages from compute_grpo_advantages()
        response_mask:     [B*G, L] — 1 for completion tokens
        config:            GRPOConfig

    Returns:
        loss:    Scalar loss tensor (call .backward() on this)
        metrics: Dict of logged values for monitoring training
    """
    shift_mask = response_mask[:, 1:].float()  # [B*G, L-1]

    # --- Sequence-level log probs ---
    seq_log_probs_cur = (current_log_probs * shift_mask).sum(dim=-1)   # [B*G]
    seq_log_probs_old = (old_log_probs    * shift_mask).sum(dim=-1)    # [B*G]

    # --- Probability ratio r = π_new / π_old ---
    # In log space: log r = log π_new - log π_old
    log_ratio = seq_log_probs_cur - seq_log_probs_old.detach()
    ratio = log_ratio.exp()  # [B*G]

    # --- PPO clipped surrogate objective ---
    # Unclipped: r * A
    policy_loss_unclipped = ratio * advantages

    # Clipped: clip(r, 1-ε, 1+ε) * A
    ratio_clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
    policy_loss_clipped = ratio_clipped * advantages

    # Take the minimum (pessimistic bound) and negate (we minimize the loss)
    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()

    # --- KL Divergence Penalty ---
    kl = compute_kl_penalty(current_log_probs, ref_log_probs, response_mask)
    kl_loss = config.kl_coeff * kl.mean()

    # --- Optional Entropy Bonus ---
    entropy_loss = torch.tensor(0.0)
    if config.entropy_coeff > 0:
        # Encourage exploration by rewarding high-entropy distributions
        entropy = -(current_log_probs * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
        entropy_loss = -config.entropy_coeff * entropy.mean()

    total_loss = policy_loss + kl_loss + entropy_loss

    # --- Metrics for logging ---
    with torch.no_grad():
        clip_fraction = ((ratio - 1).abs() > config.clip_epsilon).float().mean()
        approx_kl     = (log_ratio ** 2).mean() / 2  # Second-order approximation

    metrics = {
        "loss/total":         total_loss.item(),
        "loss/policy":        policy_loss.item(),
        "loss/kl":            kl_loss.item(),
        "loss/entropy":       entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
        "train/clip_fraction": clip_fraction.item(),
        "train/approx_kl":    approx_kl.item(),
        "train/ratio_mean":   ratio.mean().item(),
        "train/advantage_mean": advantages.mean().item(),
        "train/advantage_std":  advantages.std().item(),
    }

    return total_loss, metrics
