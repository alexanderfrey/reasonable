from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import re
from typing import List, Dict, Any
from collections import Counter
from sentence_transformers import SentenceTransformer
import language_tool_python  # Correct import
from spellchecker import SpellChecker


sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
tool = language_tool_python.LanguageTool("en-US")


def strict_format_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    pattern = r"^<thinking>\n.*?\n</thinking>\n<answer>\n.*?\n</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]


def soft_format_reward(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def structural_reward(text: str) -> float:
    score = 0.0

    # Core tag presence (0.6)
    if "<thinking>" in text and "</thinking>" in text:
        score += 0.3
    if "<answer>" in text and "</answer>" in text:
        score += 0.3

    # Proper ordering (0.4)
    if (
        text.find("<thinking>")
        < text.find("</thinking>")
        < text.find("<answer>")
        < text.find("</answer>")
    ):
        score += 0.4

    return score


def reward_answer_length(
    completions,
    target_length=50,
    reward_scale=1.0,
    penalty_scale=0.1,
    length_threshold_low=10,
    length_threshold_high=150,
    **kwargs
):
    """
    Rewards completions based on their length, encouraging answers around a target length.

    Args:
        completions (List[str]): Model-generated answers.
        target_length (int): Ideal answer length (in words or characters).
        reward_scale (float): Reward for being close to target length.
        penalty_scale (float): Penalty for being too short or too long.
        length_threshold_low (int): Lower bound for acceptable length.
        length_threshold_high (int): Upper bound for acceptable length.
        **kwargs: Other arguments.

    Returns:
        List[float]: Rewards.
    """
    rewards = []
    for completion in completions:
        length = len(completion.split())  # Or len(completion) for character count
        if length >= length_threshold_low and length <= length_threshold_high:
            reward = reward_scale * max(
                0, 1 - abs(length - target_length) / target_length
            )  # Reward decreases as length deviates from target
        elif length < length_threshold_low:
            reward = -penalty_scale * (
                length_threshold_low - length
            )  # Penalty for being too short
        else:  # length > length_threshold_high
            reward = -penalty_scale * (
                length - length_threshold_high
            )  # Penalty for being too long
        rewards.append(reward)
    return rewards


def reward_answer_similarity(
    completions, reference_answers, reward_scale=1.0, similarity_threshold=0.7, **kwargs
):
    """
    Rewards completions based on semantic similarity to provided reference answers.

    Args:
        completions (List[str]): Model-generated answers.
        reference_answers (List[str]): Corresponding correct/reference answers for each question.
        reward_scale (float): Scale factor for similarity score.
        similarity_threshold (float): Minimum similarity to get positive reward.
        **kwargs: Other arguments.

    Returns:
        List[float]: Rewards.
    """
    rewards = []
    for completion, ref_answer in zip(
        completions, reference_answers
    ):  # Assuming completions and ref_answers are aligned
        ref_embedding = sentence_model.encode([ref_answer], convert_to_tensor=True)
        completion_embedding = sentence_model.encode(
            [completion], convert_to_tensor=True
        )
        similarity_score = F.cosine_similarity(
            ref_embedding, completion_embedding
        ).item()

        if similarity_score >= similarity_threshold:
            rewards.append(similarity_score * reward_scale)
        else:
            rewards.append(
                0.0
            )  # Or negative reward if you want to penalize dissimilarity
    return rewards


def reward_question_relevance(
    completions, questions, reward_scale=1.0, similarity_threshold=0.6, **kwargs
):
    """
    Rewards completions based on semantic similarity to the questions.

    Args:
        completions (List[str]): Model-generated answers.
        questions (List[str]): Corresponding questions for each answer.
        reward_scale (float): Scale factor for similarity score.
        similarity_threshold (float): Minimum similarity to get positive reward.
        **kwargs: Other arguments.

    Returns:
        List[float]: Rewards.
    """
    rewards = []
    for completion, question in zip(completions, questions):
        question_embedding = sentence_model.encode([question], convert_to_tensor=True)
        completion_embedding = sentence_model.encode(
            [completion], convert_to_tensor=True
        )
        similarity_score = F.cosine_similarity(
            question_embedding, completion_embedding
        ).item()

        if similarity_score >= similarity_threshold:
            rewards.append(similarity_score * reward_scale)
        else:
            rewards.append(0.0)  # Or negative reward
    return rewards


def reward_repetition_penalty(completions, ngram_size=3, penalty_scale=0.2, **kwargs):
    """
    Penalizes completions with high n-gram repetition.

    Args:
        completions (List[str]): Model-generated answers.
        ngram_size (int): Size of n-grams to check for repetition.
        penalty_scale (float): Scale of the penalty for repetition.
        **kwargs: Other arguments.

    Returns:
        List[float]: Rewards (penalties are negative rewards).
    """
    rewards = []
    for completion in completions:
        words = completion.lower().split()
        if len(words) < ngram_size:
            rewards.append(0.0)  # Too short to check for n-gram repetition
            continue

        ngrams = Counter(
            tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
        )
        max_repetition_count = max(ngrams.values()) if ngrams else 0
        penalty = penalty_scale * max(
            0, max_repetition_count - 1
        )  # Penalize if n-gram repeats more than once
        rewards.append(-penalty)  # Negative reward (penalty)
    return rewards


def reward_grammar_spelling(
    completions, grammar_penalty_scale=0.1, spelling_penalty_scale=0.05, **kwargs
):
    """
    Rewards completions based on grammatical correctness and absence of spelling errors using language_tool_python.

    Args:
        completions (List[str]): Model-generated answers.
        grammar_penalty_scale (float): Penalty per grammatical error.
        spelling_penalty_scale (float): Penalty per spelling error.
        **kwargs: Other arguments.

    Returns:
        List[float]: Rewards.
    """
    rewards = []
    tool = language_tool_python.LanguageTool(
        "en-US"
    )  # Initialize grammar checker using language_tool_python
    spell = SpellChecker()  # Initialize spell checker

    for completion in completions:
        grammar_errors = tool.check(completion)
        spelling_errors = spell.unknown(completion.split())

        grammar_penalty = len(grammar_errors) * grammar_penalty_scale
        spelling_penalty = len(spelling_errors) * spelling_penalty_scale

        # Simple reward: base reward (0) minus penalties
        reward = 0.0 - grammar_penalty - spelling_penalty

        rewards.append(reward)

    return rewards


class RewardFunctions:
    def __init__(self, tokenizer_name="Qwen/Qwen2-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def length_reward(
        self, completions: List[str], target_length: int = 100, **kwargs
    ) -> List[float]:
        """Reward based on completion length proximity to target"""
        rewards = [-abs(target_length - len(completion)) for completion in completions]
        return rewards

    def diversity_reward(self, completions: List[str], **kwargs) -> List[float]:
        """Reward based on lexical diversity"""

        def calculate_diversity(text):
            words = set(text.split())
            total_words = len(text.split())
            return len(words) / total_words if total_words > 0 else 0

        return [calculate_diversity(completion) * 10 for completion in completions]

    def relevance_reward(
        self, completions: List[str], prompt: str, **kwargs
    ) -> List[float]:
        """Reward based on semantic similarity with prompt"""

        def simple_keyword_overlap(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2))

        return [
            simple_keyword_overlap(prompt, completion) * 10
            for completion in completions
        ]

    def combined_reward(
        self,
        completions: List[str],
        prompt: str,
        weights: dict = {"length": 0.3, "diversity": 0.3, "relevance": 0.4},
        **kwargs
    ) -> List[float]:
        """Combines multiple rewards with specified weights"""
        length_scores = self.length_reward(completions)
        diversity_scores = self.diversity_reward(completions)
        relevance_scores = self.relevance_reward(completions, prompt)

        combined_scores = []
        for i in range(len(completions)):
            score = (
                weights["length"] * length_scores[i]
                + weights["diversity"] * diversity_scores[i]
                + weights["relevance"] * relevance_scores[i]
            )
            combined_scores.append(score)

        return combined_scores
