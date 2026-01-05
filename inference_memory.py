"""
Inference script for Memory-Augmented GPT

Run the model with persistent episodic memory that accumulates knowledge
across documents and can be queried.

Usage:
    # Interactive mode - type text, see memory accumulate
    python inference_memory.py --interactive

    # Process a file and accumulate memories
    python inference_memory.py --input_file document.txt

    # Process multiple files, building shared memory
    python inference_memory.py --input_dir ./documents/

    # Load existing memory state and continue
    python inference_memory.py --load_memory memory_state.pt --interactive
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F

from experiential import MemoryAugmentedGPT, EpisodicMemory
from model import GPT, GPTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryInference:
    """
    Inference wrapper for Memory-Augmented GPT.

    Supports:
    - Text generation with memory retrieval
    - Memory accumulation across documents
    - Memory inspection and querying
    - Saving/loading memory state
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        memory_capacity: int = 1000,
        crystallization_threshold: float = 0.2,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model, self.tokenizer = self._load_model(
            checkpoint_path,
            memory_capacity,
            crystallization_threshold
        )
        self.model.eval()

        # Track processing stats
        self.stats = {
            'documents_processed': 0,
            'chunks_processed': 0,
            'tokens_processed': 0,
            'memories_crystallized': 0,
        }

    def _load_model(
        self,
        checkpoint_path: str,
        memory_capacity: int,
        crystallization_threshold: float
    ):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Get config
        saved_config = checkpoint.get('config', {})
        saved_args = checkpoint.get('args', {})

        # Handle both dict and namespace args
        if hasattr(saved_args, '__dict__'):
            saved_args = vars(saved_args)

        # Merge config sources
        for key in ['vocab_size', 'd_model', 'n_head', 'n_layer', 'max_seq_len', 'n_kv_head', 'd_ff']:
            if key in saved_args and key not in saved_config:
                saved_config[key] = saved_args[key]

        # Infer d_ff if needed
        d_ff = saved_config.get('d_ff')
        if d_ff is None:
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('memory_gpt_state_dict', {}))
            for key, tensor in state_dict.items():
                if 'ffn.gate_up_proj.weight' in key:
                    d_ff = tensor.shape[0] // 2
                    break

        config = GPTConfig(
            vocab_size=saved_config.get('vocab_size', 128256),
            d_model=saved_config.get('d_model', 768),
            n_head=saved_config.get('n_head', 12),
            n_layer=saved_config.get('n_layer', 12),
            max_seq_len=saved_config.get('max_seq_len', 1024),
            n_kv_head=saved_config.get('n_kv_head'),
            dropout=0.0,
            d_ff=d_ff,
        )

        logger.info(f"Model config: d_model={config.d_model}, n_layer={config.n_layer}")

        # Build model
        gpt = GPT(config)

        # Try to load base GPT weights first
        base_checkpoint_path = saved_args.get('checkpoint', 'tiny_pretrain_output/model_best_eval.pt')
        if os.path.exists(base_checkpoint_path):
            base_ckpt = torch.load(base_checkpoint_path, map_location='cpu', weights_only=False)
            base_state = base_ckpt.get('model_state_dict', {})
            # Strip prefixes
            base_state = {k.replace('module.', '').replace('_orig_mod.', ''): v
                         for k, v in base_state.items()}
            gpt.load_state_dict(base_state, strict=False)
            logger.info(f"Loaded base GPT from {base_checkpoint_path}")

        # Wrap with memory
        memory_gpt = MemoryAugmentedGPT(
            gpt,
            memory_capacity=memory_capacity,
            crystallization_threshold=crystallization_threshold,
            memory_integration=saved_args.get('integration', 'gated'),
            use_experiential=True,
        ).to(self.device)

        # Load memory-augmented weights
        if 'memory_gpt_state_dict' in checkpoint:
            memory_gpt.load_state_dict(checkpoint['memory_gpt_state_dict'], strict=False)
            logger.info("Loaded memory-augmented weights")

        # Load episodic memory if saved
        if 'episodic_memory' in checkpoint:
            memory_gpt.memory.restore(checkpoint['episodic_memory'], device=self.device)
            logger.info(f"Restored {memory_gpt.memory.size} episodic memories")

        # Load tokenizer
        tokenizer = None
        tokenizer_name = saved_args.get('tokenizer_name') or saved_args.get('tokenizer')
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                memory_gpt.set_tokenizer(tokenizer)
                logger.info(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")

        if tokenizer is None:
            # Try default
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
                memory_gpt.set_tokenizer(tokenizer)
                logger.info("Loaded default Llama-3 tokenizer")
            except Exception as e:
                logger.warning(f"Could not load default tokenizer: {e}")

        return memory_gpt, tokenizer

    def process_text(
        self,
        text: str,
        crystallize: bool = True,
        use_memory: bool = True,
        chunk_size: Optional[int] = None,
        overlap: int = 64,
        reset_state_between_chunks: bool = False,
    ) -> Dict[str, Any]:
        """
        Process text through the model, optionally crystallizing memories.

        Args:
            text: Input text to process
            crystallize: Whether to store surprising moments as memories
            use_memory: Whether to retrieve from existing memories
            chunk_size: Max tokens per chunk (default: model's max_seq_len - 1)
            overlap: Token overlap between chunks for continuity
            reset_state_between_chunks: Reset hidden state between chunks

        Returns:
            Dict with processing results and memory info
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded - cannot process text")

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if chunk_size is None:
            chunk_size = self.model.gpt.config.max_seq_len - 1

        results = {
            'chunks_processed': 0,
            'tokens_processed': len(tokens),
            'memories_before': self.model.memory.size,
            'memories_after': 0,
            'avg_surprise': 0.0,
            'crystallized_count': 0,
            'retrieved_memories': [],
            # Experiential stats
            'avg_meta_surprise': 0.0,
            'avg_confidence': 0.0,
            'avg_valence': 0.0,
            'avg_arousal': 0.0,
            'avg_salience': 0.0,
            # EMA baseline stats
            'ema_mu': None,
            'ema_sigma': None,
            # Per-chunk details
            'chunk_details': [],
        }

        # Accumulators
        total_surprise = 0.0
        total_meta_surprise = 0.0
        total_confidence = 0.0
        total_valence = 0.0
        total_arousal = 0.0
        total_salience = 0.0
        prev_memory_query = None

        # Process in chunks
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Need at least 2 tokens for next-token prediction
            if len(chunk_tokens) < 2:
                break

            input_ids = torch.tensor([chunk_tokens], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits, hidden, mem_out = self.model(
                    input_ids,
                    crystallize=crystallize,
                    use_memory=use_memory,
                    prev_memory_query=prev_memory_query,
                )

                # Track for next chunk
                if mem_out.get('next_memory_query') is not None:
                    prev_memory_query = mem_out['next_memory_query']

                # Collect chunk details
                chunk_detail = {
                    'chunk_idx': results['chunks_processed'],
                    'tokens': len(chunk_tokens),
                }

                # Experiential stats
                if mem_out.get('chunk_surprise') is not None:
                    chunk_detail['surprise'] = mem_out['chunk_surprise'].item()
                    total_surprise += chunk_detail['surprise']

                if mem_out.get('surprise') is not None:
                    chunk_detail['raw_surprise'] = mem_out['surprise'].mean().item()

                if mem_out.get('meta_surprise') is not None:
                    chunk_detail['meta_surprise'] = mem_out['meta_surprise'].mean().item()
                    total_meta_surprise += chunk_detail['meta_surprise']

                if mem_out.get('confidence_gate') is not None:
                    chunk_detail['confidence'] = mem_out['confidence_gate'].mean().item()
                    total_confidence += chunk_detail['confidence']

                if mem_out.get('valence') is not None:
                    chunk_detail['valence'] = mem_out['valence'].mean().item()
                    total_valence += chunk_detail['valence']

                if mem_out.get('arousal') is not None:
                    chunk_detail['arousal'] = mem_out['arousal'].mean().item()
                    total_arousal += chunk_detail['arousal']

                if mem_out.get('salience') is not None:
                    chunk_detail['salience'] = mem_out['salience'].mean().item()
                    total_salience += chunk_detail['salience']

                # EMA baseline stats
                if mem_out.get('ema_mu') is not None:
                    results['ema_mu'] = mem_out['ema_mu']
                    results['ema_sigma'] = mem_out.get('ema_sigma')

                # Crystallization
                if mem_out.get('crystallized', False):
                    results['crystallized_count'] += 1
                    chunk_detail['crystallized'] = True

                # Track retrieved memories
                if mem_out.get('episodic_weights') is not None:
                    weights = mem_out['episodic_weights']
                    if weights.numel() > 0:
                        top_weight, top_idx = weights.max(dim=-1)
                        chunk_detail['top_retrieval_weight'] = top_weight.item()
                        if top_weight.item() > 0.1:
                            results['retrieved_memories'].append({
                                'chunk': results['chunks_processed'],
                                'weight': top_weight.item(),
                                'memory_idx': top_idx.item(),
                            })

                results['chunk_details'].append(chunk_detail)

            results['chunks_processed'] += 1

            if reset_state_between_chunks:
                self.model.reset_hidden_state()

            # Move to next chunk with overlap
            start = end - overlap if end < len(tokens) else end

        results['memories_after'] = self.model.memory.size

        # Compute averages
        n_chunks = max(1, results['chunks_processed'])
        results['avg_surprise'] = total_surprise / n_chunks
        results['avg_meta_surprise'] = total_meta_surprise / n_chunks
        results['avg_confidence'] = total_confidence / n_chunks
        results['avg_valence'] = total_valence / n_chunks
        results['avg_arousal'] = total_arousal / n_chunks
        results['avg_salience'] = total_salience / n_chunks

        # Update global stats
        self.stats['chunks_processed'] += results['chunks_processed']
        self.stats['tokens_processed'] += results['tokens_processed']
        self.stats['memories_crystallized'] += results['crystallized_count']

        return results

    def process_file(
        self,
        file_path: str,
        crystallize: bool = True,
        use_memory: bool = True,
    ) -> Dict[str, Any]:
        """Process a text file."""
        logger.info(f"Processing file: {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Reset hidden state for new document but keep memories
        self.model.reset_hidden_state()

        results = self.process_text(text, crystallize=crystallize, use_memory=use_memory)
        results['file'] = file_path

        self.stats['documents_processed'] += 1

        logger.info(
            f"  Processed {results['tokens_processed']} tokens in {results['chunks_processed']} chunks. "
            f"Crystallized {results['crystallized_count']} memories. "
            f"Memory size: {results['memories_after']}"
        )

        return results

    def process_directory(
        self,
        dir_path: str,
        extensions: List[str] = ['.txt', '.md', '.py', '.json'],
        crystallize: bool = True,
        use_memory: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process all matching files in a directory."""
        results = []
        dir_path = Path(dir_path)

        files = []
        for ext in extensions:
            files.extend(dir_path.glob(f'**/*{ext}'))

        files = sorted(files)
        logger.info(f"Found {len(files)} files to process")

        for file_path in files:
            try:
                result = self.process_file(str(file_path), crystallize, use_memory)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")

        return results

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_memory: bool = True,
    ) -> str:
        """
        Generate text continuation using memory-augmented model.

        Args:
            prompt: Starting text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            use_memory: Whether to use retrieved memories

        Returns:
            Generated text (prompt + continuation)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        prev_memory_query = None
        generated = list(tokens)

        for _ in range(max_new_tokens):
            # Truncate to max length if needed
            max_len = self.model.gpt.config.max_seq_len
            if input_ids.size(1) > max_len:
                input_ids = input_ids[:, -max_len:]

            with torch.no_grad():
                logits, hidden, mem_out = self.model(
                    input_ids,
                    crystallize=False,  # Don't crystallize during generation
                    use_memory=use_memory,
                    prev_memory_query=prev_memory_query,
                )

                prev_memory_query = mem_out.get('next_memory_query')

                # Get next token logits
                next_logits = logits[0, -1, :] / temperature

                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated)

    def get_experiential_state(self) -> Dict[str, Any]:
        """Get current state of the experiential stream."""
        exp = self.model.experiential
        if exp is None:
            return {'enabled': False}

        state = {
            'enabled': True,
            'has_persistent_state': exp.get_state() is not None,
            'd_model': exp.d_model,
            'use_affect': exp.use_affect,
            'use_meta_surprise': exp.use_meta_surprise,
            'use_persistent_state': exp.use_persistent_state,
        }

        # EMA statistics (baseline for surprise)
        state['ema_mu'] = exp.ema_mu.item()
        state['ema_sigma'] = exp.ema_sigma.item()
        state['ema_initialized'] = exp.ema_initialized.item()

        # Raw surprise EMA
        state['ema_raw_mu'] = exp.ema_raw_mu.item()
        state['ema_raw_sigma'] = exp.ema_raw_sigma.item()

        # Persistent state info
        if exp.get_state() is not None:
            ps = exp.get_state()
            state['persistent_state_shape'] = list(ps.shape)
            state['persistent_state_norm'] = ps.norm().item()
            state['persistent_state_mean'] = ps.mean().item()
            state['persistent_state_std'] = ps.std().item()

        return state

    def get_experiential_summary(self) -> str:
        """Get human-readable summary of experiential state."""
        state = self.get_experiential_state()

        if not state.get('enabled'):
            return "Experiential stream not enabled."

        lines = [
            "Experiential Stream State",
            "=" * 50,
            f"Model dimension: {state['d_model']}",
            f"Features: affect={state['use_affect']}, meta_surprise={state['use_meta_surprise']}, persistent_state={state['use_persistent_state']}",
            "",
            "EMA Baseline Statistics:",
            f"  CE baseline (mu): {state['ema_mu']:.4f}",
            f"  CE spread (sigma): {state['ema_sigma']:.4f}",
            f"  Initialized: {state['ema_initialized']}",
            "",
            "Raw Surprise EMA:",
            f"  Mean: {state['ema_raw_mu']:.4f}",
            f"  Sigma: {state['ema_raw_sigma']:.4f}",
        ]

        if state.get('persistent_state_shape'):
            lines.extend([
                "",
                "Persistent State:",
                f"  Shape: {state['persistent_state_shape']}",
                f"  Norm: {state['persistent_state_norm']:.4f}",
                f"  Mean: {state['persistent_state_mean']:.4f}",
                f"  Std: {state['persistent_state_std']:.4f}",
            ])
        else:
            lines.append("\nPersistent State: None (not initialized)")

        return "\n".join(lines)

    def get_memory_summary(self, top_k: int = 10) -> str:
        """Get a human-readable summary of current memory state."""
        memory = self.model.memory

        if memory.size == 0:
            return "Memory is empty."

        lines = [
            f"Episodic Memory Summary",
            f"=" * 50,
            f"Total memories: {memory.size}",
            f"",
        ]

        # Top by salience
        episodes = sorted(memory.episodes, key=lambda e: e.salience, reverse=True)[:top_k]
        lines.append(f"Top {len(episodes)} by salience:")
        lines.append("-" * 50)

        for i, ep in enumerate(episodes, 1):
            text = ep.text or "(no text)"
            text = ' '.join(text.split())[:100] + "..." if len(text) > 100 else text
            lines.append(f"{i:2d}. [sal={ep.salience:.3f}, retr={ep.retrieval_count}] {text}")

        lines.append("")

        # Top by retrieval count
        episodes = sorted(memory.episodes, key=lambda e: e.retrieval_count, reverse=True)[:top_k]
        lines.append(f"Top {len(episodes)} by retrieval count:")
        lines.append("-" * 50)

        for i, ep in enumerate(episodes, 1):
            text = ep.text or "(no text)"
            text = ' '.join(text.split())[:100] + "..." if len(text) > 100 else text
            lines.append(f"{i:2d}. [retr={ep.retrieval_count}, sal={ep.salience:.3f}] {text}")

        return "\n".join(lines)

    def query_memory(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query memory with a text string.

        Returns top-k most similar memories to the query.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        if self.model.memory.size == 0:
            return []

        # Encode query and get hidden state
        tokens = self.tokenizer.encode(query_text, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, hidden, mem_out = self.model(
                input_ids,
                crystallize=False,
                use_memory=False,  # Just get embedding, don't retrieve
            )

            # Use the memory query embedding
            query_emb = mem_out.get('next_memory_query')
            if query_emb is None:
                query_emb = hidden[:, -1, :]  # Use last hidden state

            # Query memory
            retrieved, weights = self.model.memory.retrieve_soft(
                query_emb, temperature=0.1
            )

            # Get top-k
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)

            top_weights, top_indices = weights[0].topk(min(top_k, len(weights[0])))

        results = []
        for weight, idx in zip(top_weights.tolist(), top_indices.tolist()):
            ep = self.model.memory.episodes[idx]
            results.append({
                'similarity': weight,
                'salience': ep.salience,
                'retrieval_count': ep.retrieval_count,
                'text': ep.text or "(no text stored)",
            })

        return results

    def get_episode_details(self, episode_idx: int, top_k_tokens: int = 10) -> Dict[str, Any]:
        """Get detailed information about a specific episode including surprising tokens.

        Args:
            episode_idx: Index of the episode (1-based for user friendliness)
            top_k_tokens: Number of top surprising tokens to show

        Returns:
            Dict with episode details
        """
        if episode_idx < 1 or episode_idx > self.model.memory.size:
            return {'error': f'Episode index must be between 1 and {self.model.memory.size}'}

        ep = self.model.memory.episodes[episode_idx - 1]

        details = {
            'index': episode_idx,
            'timestamp': ep.timestamp,
            'salience': ep.salience,
            'valence': ep.valence,
            'arousal': ep.arousal,
            'retrieval_count': ep.retrieval_count,
            'text': ep.text,
            'has_token_surprises': ep.token_surprises is not None,
        }

        # Get surprising tokens if available
        if ep.token_surprises is not None and ep.token_ids is not None:
            surprising = ep.get_surprising_tokens(self.tokenizer, top_k=top_k_tokens)
            details['top_surprising_tokens'] = surprising

            # Also provide summary stats
            details['token_count'] = len(ep.token_ids)
            details['avg_surprise'] = sum(ep.token_surprises) / len(ep.token_surprises)
            details['max_surprise'] = max(ep.token_surprises)
            details['min_surprise'] = min(ep.token_surprises)

        return details

    def colorize_episode_text(self, episode_idx: int) -> str:
        """Get episode text colorized by per-token surprise using ANSI codes.

        Color scale (by percentile):
        - Gray/dim: bottom 25% (expected tokens)
        - White: 25-50%
        - Yellow: 50-75%
        - Red: 75-90%
        - Bright red + bold: top 10% (most surprising)
        """
        if episode_idx < 1 or episode_idx > self.model.memory.size:
            return f'Episode index must be between 1 and {self.model.memory.size}'

        ep = self.model.memory.episodes[episode_idx - 1]

        if ep.token_surprises is None or ep.token_ids is None:
            return "(No per-token surprise data - cannot colorize)"

        if self.tokenizer is None:
            return "(No tokenizer loaded - cannot colorize)"

        # ANSI color codes
        RESET = '\033[0m'
        DIM = '\033[2m'        # dim/gray for low surprise
        NORMAL = '\033[0m'     # normal for medium-low
        YELLOW = '\033[33m'    # yellow for medium-high
        RED = '\033[31m'       # red for high
        BOLD_RED = '\033[1;91m'  # bold bright red for very high

        # Calculate percentile thresholds
        surprises = ep.token_surprises
        sorted_surp = sorted(surprises)
        n = len(sorted_surp)

        p25 = sorted_surp[int(n * 0.25)] if n > 0 else 0
        p50 = sorted_surp[int(n * 0.50)] if n > 0 else 0
        p75 = sorted_surp[int(n * 0.75)] if n > 0 else 0
        p90 = sorted_surp[int(n * 0.90)] if n > 0 else 0

        # Build colorized string
        result = []
        for i, (token_id, surprise) in enumerate(zip(ep.token_ids, surprises)):
            try:
                token_text = self.tokenizer.decode([token_id])
            except:
                token_text = f"[{token_id}]"

            # Select color based on surprise percentile
            if surprise >= p90:
                color = BOLD_RED
            elif surprise >= p75:
                color = RED
            elif surprise >= p50:
                color = YELLOW
            elif surprise >= p25:
                color = NORMAL
            else:
                color = DIM

            result.append(f"{color}{token_text}{RESET}")

        return ''.join(result)

    def format_episode_details(self, episode_idx: int, colorize: bool = True) -> str:
        """Get formatted string of episode details."""
        details = self.get_episode_details(episode_idx)

        if 'error' in details:
            return details['error']

        lines = [
            f"Episode #{details['index']}",
            "=" * 50,
            f"Salience: {details['salience']:.4f}",
            f"Valence: {details['valence']:.4f} | Arousal: {details['arousal']:.4f}",
            f"Retrieval count: {details['retrieval_count']}",
            f"Timestamp: {details['timestamp']}",
        ]

        # Show colorized text if available
        if colorize and details.get('has_token_surprises'):
            lines.extend([
                "",
                "Colorized Text (gray=expected, yellow=notable, red=surprising):",
                "-" * 50,
                self.colorize_episode_text(episode_idx),
                "-" * 50,
            ])
        elif details.get('text'):
            text = details['text'][:300] + "..." if len(details['text']) > 300 else details['text']
            lines.extend(["", "Text:", text])

        if details.get('top_surprising_tokens'):
            lines.extend([
                "",
                f"Token Statistics:",
                f"  Count: {details['token_count']}",
                f"  Avg surprise: {details['avg_surprise']:.4f}",
                f"  Max surprise: {details['max_surprise']:.4f}",
                "",
                "Most Surprising Tokens:",
            ])
            for tok in details['top_surprising_tokens']:
                token_text = tok['token_text'] or f"[id={tok['token_id']}]"
                # Clean up token text for display
                token_text = repr(token_text)[1:-1]  # Show escape chars
                lines.append(f"  [{tok['index']:3d}] surp={tok['surprise']:.4f}: {token_text}")
        else:
            lines.append("\n(No per-token surprise data stored)")

        return "\n".join(lines)

    def save_memory(self, path: str):
        """Save current memory state to disk."""
        state = {
            'episodic_memory': self.model.memory.snapshot(),
            'stats': self.stats,
        }
        torch.save(state, path)
        logger.info(f"Saved memory state to {path} ({self.model.memory.size} memories)")

    def load_memory(self, path: str):
        """Load memory state from disk."""
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.model.memory.restore(state['episodic_memory'], device=self.device)
        self.stats = state.get('stats', self.stats)
        logger.info(f"Loaded memory state from {path} ({self.model.memory.size} memories)")

    def clear_memory(self):
        """Clear all episodic memories."""
        self.model.memory.clear()
        self.model.reset_hidden_state()
        logger.info("Cleared all memories")


def interactive_mode(inference: MemoryInference):
    """Run interactive REPL for memory-augmented inference."""
    print("\n" + "=" * 60)
    print("Memory-Augmented GPT - Interactive Mode")
    print("=" * 60)
    print(f"\nModel: d_model={inference.model.gpt.config.d_model}, "
          f"Memory: {inference.model.memory.size} episodes")
    print("\nCommands:")
    print("  /process <text>  - Process text and accumulate memories")
    print("  /generate <prompt> - Generate continuation")
    print("  /query <text>    - Query memory for similar content")
    print("  /memory          - Show memory summary")
    print("  /experiential    - Show experiential stream state")
    print("  /stats           - Show processing statistics")
    print("  /detail          - Show detailed last result")
    print("  /inspect <n>     - Inspect episode #n with surprising tokens")
    print("  /save <path>     - Save memory state")
    print("  /load <path>     - Load memory state")
    print("  /clear           - Clear all memories")
    print("  /help            - Show this help")
    print("  /quit            - Exit")
    print("\nOr just type text to process it.")
    print("Stats shown: Surprise | Confidence | Salience | Crystallized | Memory size\n")

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == '/quit' or cmd == '/exit':
                    print("Goodbye!")
                    break

                elif cmd == '/help':
                    print("\nCommands:")
                    print("  /process <text>  - Process text and accumulate memories")
                    print("  /generate <prompt> - Generate continuation")
                    print("  /query <text>    - Query memory for similar content")
                    print("  /memory          - Show memory summary")
                    print("  /experiential    - Show experiential stream state")
                    print("  /stats           - Show processing statistics")
                    print("  /detail          - Show last processing details")
                    print("  /inspect <n>     - Inspect episode #n with surprising tokens")
                    print("  /save <path>     - Save memory state")
                    print("  /load <path>     - Load memory state")
                    print("  /clear           - Clear all memories")

                elif cmd == '/memory':
                    print("\n" + inference.get_memory_summary())

                elif cmd == '/experiential':
                    print("\n" + inference.get_experiential_summary())

                elif cmd == '/stats':
                    print(f"\nProcessing Statistics:")
                    print(f"  Documents processed: {inference.stats['documents_processed']}")
                    print(f"  Chunks processed: {inference.stats['chunks_processed']}")
                    print(f"  Tokens processed: {inference.stats['tokens_processed']}")
                    print(f"  Memories crystallized: {inference.stats['memories_crystallized']}")
                    print(f"  Current memory size: {inference.model.memory.size}")
                    # Show experiential EMA stats
                    exp_state = inference.get_experiential_state()
                    if exp_state.get('enabled'):
                        print(f"\nExperiential Baseline:")
                        print(f"  EMA mu (CE baseline): {exp_state['ema_mu']:.4f}")
                        print(f"  EMA sigma (CE spread): {exp_state['ema_sigma']:.4f}")

                elif cmd == '/clear':
                    inference.clear_memory()
                    print("Memory cleared.")

                elif cmd == '/save':
                    if not arg:
                        arg = "memory_state.pt"
                    inference.save_memory(arg)
                    print(f"Saved to {arg}")

                elif cmd == '/load':
                    if not arg:
                        arg = "memory_state.pt"
                    inference.load_memory(arg)
                    print(f"Loaded from {arg}")

                elif cmd == '/detail':
                    if not hasattr(inference, '_last_result') or inference._last_result is None:
                        print("No processing results yet. Run /process <text> first.")
                        continue
                    result = inference._last_result
                    print(f"\nLast Processing Details:")
                    print(f"=" * 50)
                    print(f"Tokens: {result['tokens_processed']} in {result['chunks_processed']} chunks")
                    print(f"\nExperiential Metrics (averages):")
                    print(f"  Surprise: {result['avg_surprise']:.4f}")
                    print(f"  Meta-surprise: {result['avg_meta_surprise']:.4f}")
                    print(f"  Confidence: {result['avg_confidence']:.4f}")
                    print(f"  Valence: {result['avg_valence']:.4f}")
                    print(f"  Arousal: {result['avg_arousal']:.4f}")
                    print(f"  Salience: {result['avg_salience']:.4f}")
                    if result.get('ema_mu') is not None:
                        print(f"\nEMA Baseline:")
                        print(f"  CE mean: {result['ema_mu']:.4f}")
                        print(f"  CE sigma: {result['ema_sigma']:.4f}")
                    print(f"\nMemory:")
                    print(f"  Before: {result['memories_before']}, After: {result['memories_after']}")
                    print(f"  Crystallized: {result['crystallized_count']}")
                    if result['retrieved_memories']:
                        print(f"  Retrievals: {len(result['retrieved_memories'])}")
                    if result['chunk_details']:
                        print(f"\nPer-chunk details (last 5):")
                        for chunk in result['chunk_details'][-5:]:
                            flags = []
                            if chunk.get('crystallized'):
                                flags.append("CRYST")
                            flags_str = f" [{','.join(flags)}]" if flags else ""
                            print(f"  Chunk {chunk['chunk_idx']}: surp={chunk.get('surprise', 0):.3f}, "
                                  f"conf={chunk.get('confidence', 0):.3f}, "
                                  f"sal={chunk.get('salience', 0):.3f}{flags_str}")

                elif cmd == '/process':
                    if not arg:
                        print("Usage: /process <text>")
                        continue
                    result = inference.process_text(arg)
                    inference._last_result = result  # Store for /detail
                    print(f"\nProcessed {result['tokens_processed']} tokens in {result['chunks_processed']} chunks")
                    print(f"  Surprise: {result['avg_surprise']:.3f} | Meta-surprise: {result['avg_meta_surprise']:.3f}")
                    print(f"  Confidence: {result['avg_confidence']:.3f} | Salience: {result['avg_salience']:.3f}")
                    print(f"  Valence: {result['avg_valence']:.3f} | Arousal: {result['avg_arousal']:.3f}")
                    print(f"  Crystallized: {result['crystallized_count']} | Memory: {result['memories_after']}")
                    print("  (Use /detail for more info)")

                elif cmd == '/generate':
                    if not arg:
                        print("Usage: /generate <prompt>")
                        continue
                    print("\nGenerating...")
                    output = inference.generate(arg, max_new_tokens=100)
                    print(f"\n{output}")

                elif cmd == '/query':
                    if not arg:
                        print("Usage: /query <text>")
                        continue
                    results = inference.query_memory(arg, top_k=5)
                    if not results:
                        print("No memories found.")
                    else:
                        print(f"\nTop {len(results)} similar memories:")
                        for i, r in enumerate(results, 1):
                            text = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
                            print(f"\n{i}. [sim={r['similarity']:.3f}] {text}")

                elif cmd == '/file':
                    if not arg:
                        print("Usage: /file <path>")
                        continue
                    inference.process_file(arg)

                elif cmd == '/inspect':
                    if not arg:
                        print(f"Usage: /inspect <n>  (1 to {inference.model.memory.size})")
                        continue
                    try:
                        idx = int(arg)
                        print("\n" + inference.format_episode_details(idx))
                    except ValueError:
                        print(f"Invalid episode number: {arg}")

                else:
                    print(f"Unknown command: {cmd}")

            else:
                # Default: process text
                result = inference.process_text(user_input)
                inference._last_result = result  # Store for /detail
                print(f"\nProcessed {result['tokens_processed']} tokens | "
                      f"Surprise: {result['avg_surprise']:.3f} | "
                      f"Conf: {result['avg_confidence']:.3f} | "
                      f"Sal: {result['avg_salience']:.3f} | "
                      f"Cryst: {result['crystallized_count']} | "
                      f"Mem: {result['memories_after']}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Memory-Augmented GPT Inference")

    # Model
    parser.add_argument("--checkpoint", default="memory_augmented_output/memory_gpt_epoch_1.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--memory_capacity", type=int, default=1000)
    parser.add_argument("--crystallization_threshold", type=float, default=0.2)

    # Input modes
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Process a single file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Process all files in directory")
    parser.add_argument("--extensions", type=str, default=".txt,.md",
                        help="File extensions to process (comma-separated)")

    # Memory management
    parser.add_argument("--load_memory", type=str, default=None,
                        help="Load existing memory state")
    parser.add_argument("--save_memory", type=str, default=None,
                        help="Save memory state after processing")

    # Generation
    parser.add_argument("--generate", type=str, default=None,
                        help="Generate continuation from prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Options
    parser.add_argument("--no_crystallize", action="store_true",
                        help="Don't crystallize new memories")
    parser.add_argument("--no_memory", action="store_true",
                        help="Don't use memory retrieval")

    args = parser.parse_args()

    # Initialize
    inference = MemoryInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        memory_capacity=args.memory_capacity,
        crystallization_threshold=args.crystallization_threshold,
    )

    # Load existing memory if specified
    if args.load_memory:
        inference.load_memory(args.load_memory)

    crystallize = not args.no_crystallize
    use_memory = not args.no_memory

    # Process inputs
    if args.input_file:
        inference.process_file(args.input_file, crystallize=crystallize, use_memory=use_memory)

    if args.input_dir:
        extensions = args.extensions.split(',')
        inference.process_directory(args.input_dir, extensions=extensions,
                                    crystallize=crystallize, use_memory=use_memory)

    # Generate if requested
    if args.generate:
        print("\nGenerating from prompt...")
        output = inference.generate(
            args.generate,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_memory=use_memory,
        )
        print(f"\n{output}")

    # Save memory if requested
    if args.save_memory:
        inference.save_memory(args.save_memory)

    # Interactive mode
    if args.interactive:
        interactive_mode(inference)

    # Print final summary if we processed anything
    if not args.interactive and (args.input_file or args.input_dir):
        print("\n" + "=" * 50)
        print("Processing Complete")
        print("=" * 50)
        print(f"Documents: {inference.stats['documents_processed']}")
        print(f"Tokens: {inference.stats['tokens_processed']}")
        print(f"Memories: {inference.model.memory.size}")
        print("\nMemory Summary:")
        print(inference.get_memory_summary(top_k=5))


if __name__ == "__main__":
    main()
