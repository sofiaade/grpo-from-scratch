"""
GRPOTrainer
===========
Ties together:
  - Prompt sampling
  - Group rollout generation
  - Reward scoring
  - Advantage computation
  - Policy gradient update

Usage:
    trainer = GRPOTrainer(model, ref_model, tokenizer, reward_fn, config)
    for step, metrics in trainer.train(prompts, num_steps=500):
        print(step, metrics)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Callable, Iterator, List
import logging
import random

from .algorithm import (
    GRPOConfig,
    compute_grpo_advantages,
    compute_token_log_probs,
    grpo_loss,
)

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Full GRPO training loop for a language model.

    Args:
        policy_model:  The model being trained (will have gradients).
        ref_model:     Reference model (frozen, no gradients).
                       Usually the SFT checkpoint before GRPO.
        tokenizer:     HuggingFace tokenizer.
        reward_fn:     Callable(prompts: list[str], completions: list[str]) -> list[float]
                       Returns a reward score in [0, 1] for each (prompt, completion) pair.
        config:        GRPOConfig hyperparameters.

    Architecture note:
        policy_model and ref_model are usually the same architecture.
        ref_model is frozen with torch.no_grad() throughout training.
        They can be loaded from the same checkpoint — ref_model is never updated.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        reward_fn: Callable[[List[str], List[str]], List[float]],
        config: GRPOConfig,
        device: str = "cuda",
    ):
        self.policy  = policy_model.to(device)
        self.ref     = ref_model.to(device)
        self.tok     = tokenizer
        self.reward_fn = reward_fn
        self.config  = config
        self.device  = device

        # Freeze reference model — it never receives gradient updates
        for param in self.ref.parameters():
            param.requires_grad_(False)
        self.ref.eval()

        # AdamW with a small LR — GRPO updates are fine-grained
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,   # Typically no weight decay in RLHF
        )

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    # ------------------------------------------------------------------
    # Step 1: Generate G completions per prompt (rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout(self, prompts: List[str]) -> tuple:
        """
        For each prompt, generate G completions using the current policy.

        Returns:
            input_ids:     [B*G, L] — tokenized (prompt + completion)
            response_mask: [B*G, L] — 1 for completion tokens only
            completions:   list of decoded completion strings (length B*G)
            prompts_rep:   prompts repeated G times (for reward_fn)
        """
        G = self.config.group_size
        self.policy.eval()

        # Tokenize prompts
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        prompt_ids     = enc["input_ids"]           # [B, prompt_len]
        prompt_mask    = enc["attention_mask"]       # [B, prompt_len]
        prompt_len     = prompt_ids.shape[1]
        B              = prompt_ids.shape[0]

        # Repeat each prompt G times: [B, L] → [B*G, L]
        prompt_ids_rep  = prompt_ids.repeat_interleave(G, dim=0)   # [B*G, prompt_len]
        prompt_mask_rep = prompt_mask.repeat_interleave(G, dim=0)

        # Generate completions
        output = self.policy.generate(
            input_ids      = prompt_ids_rep,
            attention_mask = prompt_mask_rep,
            max_new_tokens = self.config.max_new_tokens,
            temperature    = self.config.temperature,
            do_sample      = True,
            pad_token_id   = self.tok.pad_token_id,
            eos_token_id   = self.tok.eos_token_id,
        )  # [B*G, prompt_len + response_len]

        # Build response mask: 1 for tokens AFTER the prompt
        full_ids      = output                              # [B*G, full_len]
        full_len      = full_ids.shape[1]
        response_mask = torch.zeros_like(full_ids)
        response_mask[:, prompt_len:] = 1                  # mask the completion

        # Decode only the completion tokens for reward scoring
        completion_ids = full_ids[:, prompt_len:]           # [B*G, resp_len]
        completions    = self.tok.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        prompts_rep = [p for p in prompts for _ in range(G)]

        self.policy.train()
        return full_ids, response_mask, completions, prompts_rep

    # ------------------------------------------------------------------
    # Step 2: Score completions with reward function
    # ------------------------------------------------------------------

    def _score_rewards(self, prompts_rep, completions) -> torch.Tensor:
        """
        Call user-provided reward function and return as tensor.

        The reward function should return floats in [0, 1] (or [-1, 1]).
        Higher = better completion.
        """
        rewards_list = self.reward_fn(prompts_rep, completions)
        return torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Step 3: Compute log probs under current + reference policy
    # ------------------------------------------------------------------

    def _compute_log_probs(self, model, input_ids, response_mask, use_grad=True):
        """
        Run model forward pass and extract token log probs.

        We call this twice per update step:
          - Once for the current policy (with gradients)
          - Once for the reference policy (no gradients, already disabled)
        """
        ctx = torch.enable_grad if use_grad else torch.no_grad
        with ctx():
            outputs = model(input_ids=input_ids)
            logits  = outputs.logits                            # [B*G, L, V]
            lp = compute_token_log_probs(logits, input_ids, response_mask)
        return lp                                               # [B*G, L-1]

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        prompts: List[str],
        num_steps: int = 500,
        batch_size: int = 4,
    ) -> Iterator[tuple[int, dict]]:
        """
        Run the GRPO training loop.

        Each step:
          1. Sample batch of prompts
          2. Generate G completions per prompt (rollout)
          3. Score with reward function
          4. Compute GRPO advantages (group normalization)
          5. Forward pass through current and reference policy
          6. Compute GRPO loss and update policy

        Args:
            prompts:    Full list of training prompts to sample from.
            num_steps:  Number of gradient update steps.
            batch_size: Prompts per step (completions = batch_size * G).

        Yields:
            (step, metrics_dict) — use for logging / early stopping
        """
        for step in range(1, num_steps + 1):
            # --- Sample batch ---
            batch_prompts = random.sample(prompts, min(batch_size, len(prompts)))

            # --- Rollout: generate G completions per prompt ---
            input_ids, response_mask, completions, prompts_rep = self._rollout(batch_prompts)

            # --- Reward scoring ---
            rewards = self._score_rewards(prompts_rep, completions)  # [B*G]

            # --- Log reward stats (useful for debugging) ---
            reward_mean = rewards.mean().item()
            reward_std  = rewards.std().item()

            # --- GRPO advantage computation ---
            advantages = compute_grpo_advantages(
                rewards,
                group_size=self.config.group_size,
                normalize=self.config.normalize_advantages,
            )  # [B*G]

            # --- Reference log probs (frozen) ---
            with torch.no_grad():
                ref_lp = self._compute_log_probs(
                    self.ref, input_ids, response_mask, use_grad=False
                )

            # --- Store "old" log probs before update ---
            with torch.no_grad():
                old_lp = self._compute_log_probs(
                    self.policy, input_ids, response_mask, use_grad=False
                )

            # --- Current policy log probs (with gradients) ---
            self.optimizer.zero_grad()
            cur_lp = self._compute_log_probs(
                self.policy, input_ids, response_mask, use_grad=True
            )

            # --- Compute loss and backprop ---
            loss, metrics = grpo_loss(
                current_log_probs=cur_lp,
                old_log_probs=old_lp,
                ref_log_probs=ref_lp,
                advantages=advantages,
                response_mask=response_mask,
                config=self.config,
            )

            loss.backward()

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Add reward metrics to output
            metrics["reward/mean"] = reward_mean
            metrics["reward/std"]  = reward_std
            metrics["step"]        = step

            if step % 10 == 0:
                logger.info(
                    f"Step {step:4d} | loss={metrics['loss/total']:.4f} "
                    f"| reward={reward_mean:.3f} | kl={metrics['loss/kl']:.4f}"
                )

            yield step, metrics
