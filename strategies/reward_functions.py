from typing import List
import torch
from transformers import AutoTokenizer
import numpy as np

class RewardFunctions:
    def __init__(self, tokenizer_name="Qwen/Qwen2-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def length_reward(self, completions: List[str], target_length: int = 100, **kwargs) -> List[float]:
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
    
    def relevance_reward(self, completions: List[str], prompt: str, **kwargs) -> List[float]:
        """Reward based on semantic similarity with prompt"""
        def simple_keyword_overlap(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2))
            
        return [simple_keyword_overlap(prompt, completion) * 10 for completion in completions]
    
    def combined_reward(self, completions: List[str], prompt: str, 
                       weights: dict = {"length": 0.3, "diversity": 0.3, "relevance": 0.4},
                       **kwargs) -> List[float]:
        """Combines multiple rewards with specified weights"""
        length_scores = self.length_reward(completions)
        diversity_scores = self.diversity_reward(completions)
        relevance_scores = self.relevance_reward(completions, prompt)
        
        combined_scores = []
        for i in range(len(completions)):
            score = (weights["length"] * length_scores[i] +
                    weights["diversity"] * diversity_scores[i] +
                    weights["relevance"] * relevance_scores[i])
            combined_scores.append(score)
            
        return combined_scores