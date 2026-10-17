"""
Reward Functions for Mathematical Reasoning
============================================
GRPO needs a reward function that scores (prompt, completion) pairs.

For math tasks (GSM8K, MATH), the most common reward signals are:
  1. Exact match — did the model produce the correct final answer?
  2. Format reward — did the model follow the expected output format?
  3. Length penalty — penalize overly long outputs
  4. Process reward — is each reasoning step correct? (harder, needs PRM)

This file implements composable reward functions for math problems.

Design philosophy:
  - Each reward function is stateless and pure
  - Rewards are normalized to [0, 1]
  - They compose via RewardComposer with configurable weights
"""

import re
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Answer Extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    r"""
    Extract answer from LaTeX \boxed{} notation.
    This is the standard format for MATH dataset answers.

    Example:
        text = "...therefore the answer is \boxed{42}."
        → "42"
    """
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from GSM8K-style output.
    GSM8K answers appear after '####'.

    Example:
        text = "She had 10 apples. She ate 3. #### 7"
        → "7"
    """
    # Pattern 1: #### answer (standard GSM8K format)
    match = re.search(r"####\s*([+-]?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()

    # Pattern 2: "The answer is X" fallback
    match = re.search(r"[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()

    # Pattern 3: Last number in the text (last resort)
    numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
    return numbers[-1] if numbers else None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    Removes commas, leading zeros, and extra whitespace.

    Example: "1,000" → "1000", "007" → "7"
    """
    if answer is None:
        return ""
    answer = answer.replace(",", "").strip()
    try:
        # Normalize numeric representations: 1.0 == 1
        return str(float(answer)) if "." in answer else str(int(answer))
    except ValueError:
        return answer.lower()


# ---------------------------------------------------------------------------
# Individual Reward Functions
# ---------------------------------------------------------------------------

def exact_match_reward(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[str],
    extractor: Callable = extract_gsm8k_answer,
) -> list[float]:
    """
    Binary reward: 1.0 if extracted answer matches ground truth, else 0.0.

    This is the primary reward signal for verifiable math problems.
    It's sparse (most samples get 0) but clean and unambiguous.

    Args:
        prompts:       Batch of input prompts (used for context, not scoring here)
        completions:   Batch of model-generated completions
        ground_truths: Correct answers (one per prompt, repeated G times externally)
        extractor:     Function to extract answer from completion string

    Returns:
        rewards: List of 0.0 or 1.0, one per completion
    """
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        extracted = extractor(completion)
        reward = 1.0 if (
            extracted is not None and
            normalize_answer(extracted) == normalize_answer(truth)
        ) else 0.0
        rewards.append(reward)
    return rewards


def format_reward(
    prompts: list[str],
    completions: list[str],
    required_pattern: str = r"####\s*\d+",
) -> list[float]:
    """
    Reward completions that follow the expected output format.

    Teaches the model to produce structured, parseable outputs.
    Often used alongside exact_match_reward.

    Returns 0.5 for correct format (partial credit), 0.0 otherwise.
    The idea: format alone is necessary but not sufficient.
    """
    rewards = []
    for completion in completions:
        has_format = bool(re.search(required_pattern, completion))
        rewards.append(0.5 if has_format else 0.0)
    return rewards


def length_penalty_reward(
    prompts: list[str],
    completions: list[str],
    max_tokens: int = 256,
    penalty_per_extra_token: float = 0.001,
) -> list[float]:
    """
    Penalize outputs that are longer than necessary.

    Motivation: without this, models learn to pad outputs with irrelevant
    text to avoid accidentally cutting off the answer. This is a mild form
    of reward hacking — long outputs "feel" safer to the model.

    Returns a small penalty in [-1, 0] based on excess length.
    """
    rewards = []
    for completion in completions:
        length = len(completion.split())
        penalty = min(0.0, -(max(0, length - max_tokens)) * penalty_per_extra_token)
        rewards.append(penalty)
    return rewards


def chain_of_thought_reward(
    prompts: list[str],
    completions: list[str],
    min_steps: int = 2,
) -> list[float]:
    """
    Small reward for showing reasoning steps before the answer.

    Motivation: we want the model to reason, not just output numbers.
    This encourages multi-step intermediate work.

    Heuristic: count lines that look like reasoning steps.
    """
    rewards = []
    for completion in completions:
        # Count lines with mathematical operations or reasoning words
        step_patterns = [
            r"\d+\s*[\+\-\*\/]\s*\d+",       # arithmetic
            r"(first|then|next|so|therefore|thus|since)",  # connectives
            r"=\s*\d+",                         # assignments/results
        ]
        step_count = sum(
            len(re.findall(p, completion, re.IGNORECASE))
            for p in step_patterns
        )
        reward = 0.2 if step_count >= min_steps else 0.0
        rewards.append(reward)
    return rewards


# ---------------------------------------------------------------------------
# Reward Composer
# ---------------------------------------------------------------------------

class RewardComposer:
    """
    Combines multiple reward signals with configurable weights.

    Example:
        composer = RewardComposer([
            (exact_match_reward,      1.0),
            (format_reward,           0.3),
            (length_penalty_reward,   0.1),
            (chain_of_thought_reward, 0.2),
        ])
        rewards = composer(prompts, completions, ground_truths=truths)

    The final reward is a weighted sum, clipped to [-1, 1].
    """

    def __init__(self, reward_fns_and_weights: list[tuple]):
        """
        Args:
            reward_fns_and_weights: List of (reward_fn, weight) tuples.
                Each reward_fn: (prompts, completions, **kwargs) → list[float]
        """
        self.fns_weights = reward_fns_and_weights

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        n = len(completions)
        total = [0.0] * n

        for fn, weight in self.fns_weights:
            try:
                scores = fn(prompts, completions, **kwargs)
            except TypeError:
                # Some reward fns don't use kwargs
                scores = fn(prompts, completions)

            for i, score in enumerate(scores):
                total[i] += weight * score

        # Clip to [-1, 1] range
        return [max(-1.0, min(1.0, r)) for r in total]
