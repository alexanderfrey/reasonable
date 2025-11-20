import inspect
import torch
import torch.distributed as dist
from torch import amp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
# Import AutoTokenizer from transformers
from transformers import AutoTokenizer
import argparse
from argparse import Namespace # For type hinting args
import os, json, glob, csv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from tqdm import tqdm
import math
import numpy as np
import time
import logging
from typing import Optional
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_PIN_MEMORY_DEVICE_SUPPORTED = "pin_memory_device" in inspect.signature(DataLoader.__init__).parameters


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped by DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def init_distributed_mode(args: Namespace) -> bool:
    """
    Initialize torch.distributed if --distributed is enabled.
    Returns True when distributed training is active.
    """
    if not hasattr(args, "distributed"):
        args.distributed = False

    args.rank = 0
    args.world_size = 1
    args.is_main_process = True
    args.local_rank = getattr(args, "local_rank", -1)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1:
        args.distributed = True

    if not args.distributed:
        if args.local_rank in (-1, None):
            args.local_rank = 0
        return False

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable but --distributed was requested.")

    if "RANK" not in os.environ:
        raise RuntimeError("--distributed expects RANK env variable (use torchrun to launch).")
    rank = int(os.environ.get("RANK", 0))
    world_size = env_world_size
    local_rank_env = os.environ.get("LOCAL_RANK")

    if world_size <= 1:
        raise RuntimeError("--distributed requires WORLD_SIZE>1 (launch with torchrun).")

    if local_rank_env is not None:
        args.local_rank = int(local_rank_env)
    elif args.local_rank in (-1, None):
        raise RuntimeError("--distributed expects LOCAL_RANK env or --local_rank argument.")

    args.rank = rank
    args.world_size = world_size

    torch.cuda.set_device(args.local_rank)
    dist_backend = getattr(args, "dist_backend", "nccl")
    dist.init_process_group(
        backend=dist_backend,
        rank=args.rank,
        world_size=args.world_size,
        device_id=torch.device(f"cuda:{args.local_rank}"),
    )
    dist.barrier()
    args.is_main_process = args.rank == 0
    return True


def compute_default_n_kv_head(n_head: int) -> int:
    """
    Prefer a 4:1 query-to-KV head ratio (GQA). If that ratio does not divide n_head,
    fall back to the largest divisor ≤ target. Guarantees at least 1 head.
    """
    if n_head <= 0:
        return n_head
    target = max(1, n_head // 4)
    while target > 1 and n_head % target != 0:
        target -= 1
    return max(1, target)


def _build_optimizer(trainable_params, lr: float, weight_decay: float, *, logger_prefix: str = ""):
    """
    Prefer bitsandbytes AdamW8bit when available to reduce optimizer memory foot print.
    """
    try:
        import bitsandbytes as bnb  # type: ignore

        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
        note = f"{logger_prefix}Using bitsandbytes AdamW8bit optimizer."
        logger.info(note.strip())
        return optimizer
    except Exception as exc:
        note = f"{logger_prefix}Falling back to torch AdamW (8-bit optimizer unavailable: {exc})"
        logger.warning(note.strip())
        return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


# Import the model class from model.py
try:
    from model import GPT, upgrade_state_dict_for_block_norms
    from identity_block import IdentityBlock
except ImportError:
    print("Error: model.py not found. Please ensure it's in the same directory.")
    exit(1)

# --- Tokenizer Setup ---

# --- Pretokenization Function ---
def pretokenize_corpus(
    corpus_file,
    tokenizer,
    max_seq_len,
    stride,
    output_token_file,
    data_type="Training",
    batch_lines: int = 1024,  # safer default (reduce if still memory pressure)
):
    """
    Memory-safe pretokenization:
    - Tokenizes `batch_lines` lines at a time.
    - Appends EOS per line (GPT-3-style doc separator).
    - Streams tokens to a raw .bin file (no gigantic list of arrays).
    - Returns (final_token_file, num_examples), where final_token_file will end with .bin
      to indicate raw token storage.

    NOTE: We keep the dataset & metadata format compatible by adding dtype info later.
    """
    print(f"Starting pre-tokenization of {data_type} data: {corpus_file}...")
    print(f"Using max_seq_len: {max_seq_len}, stride: {stride}")

    if not isinstance(stride, int) or stride < 1:
        raise SystemExit(f"Error: Stride must be a positive integer. Received: {stride}")
    if stride > max_seq_len:
        print(f"Warning: Stride ({stride}) > max_seq_len ({max_seq_len}). Sequences will be non-overlapping.")

    # --- choose on-disk dtype from vocab size (matches your previous logic) ---
    vocab_size = len(tokenizer)
    if vocab_size < 2**16:
        dtype = np.uint16
    elif vocab_size < 2**32:
        dtype = np.uint32
    else:
        dtype = np.int64
    dtype_name = np.dtype(dtype).name  # e.g., 'uint16'

    # We'll write to a .bin file (raw 1D array). Keep the same base name.
    # e.g., .../training_gpt2_tokens.bin
    if output_token_file.endswith(".npy"):
        final_token_file = output_token_file[:-4] + ".bin"
    else:
        final_token_file = output_token_file
    tmp_bin = final_token_file + ".tmp"

    out_dir = os.path.dirname(final_token_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    eos_id = tokenizer.eos_token_id
    add_spec = False  # don't add extra specials automatically
    trunc = False     # no truncation here

    token_count = 0
    line_count = 0

    # Stream to a binary file by appending batches
    try:
        with open(corpus_file, "r", encoding="utf-8") as f_in, open(tmp_bin, "wb") as f_out:
            pbar = tqdm(desc=f"Tokenizing {data_type}", unit="line", leave=True, dynamic_ncols=True)
            while True:
                # Read up to batch_lines non-empty lines
                lines = []
                for _ in range(batch_lines):
                    line = f_in.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        lines.append(line)

                if not lines:
                    break

                # Fast batch tokenize
                enc = tokenizer(
                    lines,
                    add_special_tokens=add_spec,
                    padding=False,
                    truncation=trunc,
                    return_attention_mask=False,
                    return_length=True,
                )
                ids_list = enc["input_ids"]  # List[List[int]]

                # Build a flat array for this batch (append EOS if present)
                if eos_id is not None:
                    flat_iter = (tok for ids in ids_list for tok in (ids + [eos_id]))
                    added_tokens = sum(int(L) for L in enc["length"]) + len(ids_list)
                else:
                    flat_iter = (tok for ids in ids_list for tok in ids)
                    added_tokens = sum(int(L) for L in enc["length"])

                # Convert generator to a NumPy array (just this batch), then write to disk
                batch_arr = np.fromiter(flat_iter, dtype=dtype)
                batch_arr.tofile(f_out)  # append raw bytes

                token_count += int(added_tokens)
                line_count += len(lines)
                pbar.update(len(lines))
            pbar.close()

    except FileNotFoundError:
        raise SystemExit(f"\nError: {data_type} corpus file not found at {corpus_file}")
    except Exception as e:
        raise SystemExit(f"\nError during tokenization of {corpus_file}: {e}")

    # Finalize the .bin file
    try:
        os.replace(tmp_bin, final_token_file)
    except Exception as e:
        # cleanup tmp on failure
        try:
            os.remove(tmp_bin)
        except Exception:
            pass
        raise SystemExit(f"Failed to finalize token file '{final_token_file}': {e}")

    if token_count == 0:
        raise SystemExit(f"Error: No tokens found in the {data_type} corpus file.")

    print(f"Finished tokenizing {line_count:,} lines → {token_count:,} tokens ({data_type}).")
    print(f"{data_type} token stream saved to {final_token_file}")

    # Calculate examples based on total token_count (no need to load the file)
    chunk = max_seq_len + 1
    if token_count < chunk:
        num_examples = 0
        print(
            f"Warning: Total {data_type} tokens ({token_count}) < {chunk}; no full sequences."
        )
    else:
        num_examples = (token_count - chunk) // stride + 1

    print(f"Calculated {num_examples:,} {data_type} examples for max_seq_len={max_seq_len} and stride={stride}")

    # Return the actual on-disk token file (.bin) and example count.
    # We'll store dtype/format in metadata (see patch below).
    return final_token_file, num_examples, dtype_name


# --- Dataset Class ---
class PretokenizedDataset(Dataset):
    def __init__(self, token_file_path, num_examples, max_seq_len, stride, data_type="Data"):
        self.token_file_path = token_file_path
        self.num_examples = num_examples  # MUST be calculated using the stride
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.data_type = data_type

        # Try loading .npy mmap first
        self.tokens = None
        npy_load_err = None
        try:
            if self.token_file_path.endswith(".npy"):
                self.tokens = np.load(self.token_file_path, mmap_mode="r")
        except FileNotFoundError:
            print(f"Error: {self.data_type} token file not found at {self.token_file_path}")
            exit(1)
        except Exception as e:
            npy_load_err = e  # fall back to .bin path

        # Fallback: load raw .bin via np.memmap using dtype from metadata
        if self.tokens is None:
            # --- Find the correct metadata path reliably ---
            meta_path = None
            if self.token_file_path.endswith("_tokens.npy"):
                meta_path = self.token_file_path.replace("_tokens.npy", "_metadata.json")
            elif self.token_file_path.endswith("_tokens.bin"):
                meta_path = self.token_file_path.replace("_tokens.bin", "_metadata.json")
            else:
                base, ext = os.path.splitext(self.token_file_path)
                # last-resort heuristic: try inserting underscore before 'metadata'
                meta_candidate = base.replace("_tokens", "_metadata") + ".json"
                meta_path = meta_candidate

            # Default dtype if metadata missing
            dtype_name = "uint32"

            # Read dtype from metadata if present
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    dtype_name = meta.get("dtype", dtype_name)
            except Exception:
                print(f"Warning: Could not read metadata at {meta_path}; will attempt to guess dtype from file size.")

                # --- Guess dtype from file size as a robust fallback ---
                try:
                    fsize = os.path.getsize(self.token_file_path)
                    for cand in ("uint16", "uint32", "int64"):
                        if fsize % np.dtype(cand).itemsize == 0:
                            dtype_name = cand
                            break
                except Exception:
                    pass

            # Finally memmap with the (meta or guessed) dtype
            try:
                self.tokens = np.memmap(self.token_file_path, dtype=np.dtype(dtype_name), mode="r")
            except FileNotFoundError:
                print(f"Error: {self.data_type} token file not found at {self.token_file_path}")
                exit(1)
            except Exception as e:
                print(f"Error loading token file {self.token_file_path}: {e}")
                if npy_load_err:
                    print(f"(np.load error earlier: {npy_load_err})")
                exit(1)

        print(
            f"{self.data_type} Dataset initialized with {self.num_examples:,} examples using memory-mapped file: {self.token_file_path}"
        )
        print(f"Total {self.data_type} tokens in mapped file: {len(self.tokens):,}")

        # Minimum tokens needed for at least one example
        min_tokens_for_one_example = self.max_seq_len + 1
        if len(self.tokens) < min_tokens_for_one_example:
            print(
                f"Error: Total {self.data_type} tokens ({len(self.tokens)}) is less than required for one sequence ({min_tokens_for_one_example})."
            )
            if self.num_examples > 0:
                exit(1)
            else:
                print(f"Warning: Number of calculated {self.data_type} examples is {self.num_examples}. Dataset is likely unusable.")

        # Sanity check num_examples against theoretical maximum
        if len(self.tokens) >= min_tokens_for_one_example:
            theoretical_max_examples = (len(self.tokens) - (self.max_seq_len + 1)) // self.stride + 1
            if self.num_examples > theoretical_max_examples:
                print(
                    f"Warning: Passed num_examples ({self.num_examples}) is greater than theoretically possible "
                    f"({theoretical_max_examples}) with stride {self.stride}. Clamping to theoretical max."
                )
                self.num_examples = theoretical_max_examples
            elif self.num_examples <= 0 and theoretical_max_examples > 0:
                print(
                    f"Error: Calculated number of {self.data_type} examples is {self.num_examples}, "
                    f"but {theoretical_max_examples} seem available with stride {self.stride}. Check metadata or num_examples calculation."
                )
                exit(1)
        elif self.num_examples > 0:
            print(f"Error: Not enough tokens for any example, but num_examples is {self.num_examples}. Check metadata.")
            exit(1)

        if self.num_examples <= 0:
            print(f"Warning: {self.data_type} dataset initialized with {self.num_examples} examples. This dataset will be empty.")


    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples:
            raise IndexError(
                f"Index {idx} out of bounds for {self.num_examples} {self.data_type} examples"
            )

        # Calculate start and end position using the stride
        start_idx = idx * self.stride # <--- Key change here
        end_idx = start_idx + self.max_seq_len + 1

        # This check is still important, especially if num_examples wasn't perfectly calculated
        # or if there's an off-by-one somewhere.
        if end_idx > len(self.tokens):
             # This can happen if num_examples was calculated slightly too optimistically or if len(tokens) is small
             # For the very last few indices, it's possible that start_idx is valid, but start_idx + max_seq_len + 1 isn't.
             # This scenario should ideally be caught by the num_examples calculation.
             actual_len = len(self.tokens) - start_idx
             if actual_len < self.max_seq_len + 1 and self.num_examples == 1 : # only one example possible and it's too short
                 raise IndexError(f"Not enough tokens for even one full sequence with start_idx {start_idx}. Required {self.max_seq_len +1}, available {actual_len}")

             # This error means num_examples was likely calculated incorrectly.
             raise IndexError(
                 f"Calculated end index {end_idx} (start_idx={start_idx}, idx={idx}) "
                 f"exceeds token array length {len(self.tokens)}. "
                 f"Num examples: {self.num_examples}, max_seq_len: {self.max_seq_len}, stride: {self.stride}."
             )

        token_chunk = self.tokens[start_idx:end_idx]

        input_ids = torch.tensor(token_chunk[:-1], dtype=torch.long)
        labels = torch.tensor(token_chunk[1:], dtype=torch.long)

        if input_ids.shape[0] != self.max_seq_len or labels.shape[0] != self.max_seq_len:
            # This should ideally not happen if end_idx and num_examples logic is correct
            raise ValueError(
                f"Data loading error: Unexpected sequence length at index {idx}. "
                f"Input: {input_ids.shape[0]}, Label: {labels.shape[0]}, Expected: {self.max_seq_len}. "
                f"Chunk len: {len(token_chunk)}, start_idx: {start_idx}, end_idx: {end_idx}"
            )

        return {"input_ids": input_ids, "labels": labels}


# --- Evaluation Function ---
@torch.no_grad()  # Ensure no gradients are computed during evaluation
def evaluate(model, eval_dataloader, criterion, device, pad_token_id, use_amp, args: Namespace):
    """Performs evaluation on the evaluation dataset."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_tokens = 0 # Count non-pad tokens for accurate loss calculation

    show_progress = (not getattr(args, "distributed", False)) or getattr(args, "is_main_process", True)
    eval_iterator = eval_dataloader
    if show_progress:
        eval_iterator = tqdm(
            eval_dataloader, desc="Evaluating", leave=False
        )  # Inner progress bar

    for batch in eval_iterator:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass


        from torch.amp import autocast
        with autocast("cuda", enabled=use_amp):
            try:
                logits, _ = model(input_ids)
                # Important: Calculate loss correctly, ignoring padding index
                # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize), (Batch * SeqLen)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # Calculate loss per token, considering padding
                # Create mask for non-padding tokens in labels
                # mask = (labels != pad_token_id).view(-1)
                # loss_per_token = loss[mask] # Select losses only for non-pad tokens
                # total_loss += loss_per_token.sum().item()
                # total_tokens += mask.sum().item()

                # Simpler approach using criterion's built-in ignore_index:
                # The loss returned by `criterion(..., ignore_index=pad_token_id)`
                # is already the mean loss over non-ignored tokens.
                # We need to multiply by the number of non-ignored tokens in the batch
                # to get the total batch loss, then divide by total tokens at the end.
                mask = (labels != pad_token_id)
                num_non_pad_tokens_in_batch = mask.sum().item()

                if num_non_pad_tokens_in_batch > 0:
                    # loss is the mean loss per non-pad token in the batch
                    batch_total_loss = loss.item() * num_non_pad_tokens_in_batch
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += batch_total_loss
                        total_tokens += num_non_pad_tokens_in_batch
                    else:
                        print(f"\nWarning: NaN/Inf loss encountered during evaluation.")
                # else: batch contained only padding, ignore.

            except Exception as e:
                print(f"\nError during evaluation forward/loss: {e}")
                # Decide if you want to skip this batch or raise the error
                # continue

    if show_progress and hasattr(eval_iterator, "close"):
        eval_iterator.close()

    model.train()  # Set model back to training mode

    if getattr(args, "distributed", False) and dist.is_initialized():
        totals = torch.tensor([total_loss, float(total_tokens)], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = totals[0].item()
        total_tokens = int(totals[1].item())

    if total_tokens == 0:
        print("\nWarning: No valid (non-pad) tokens processed during evaluation.")
        # If eval dataset was empty or all padding
        if len(eval_dataloader.dataset) > 0:
             print("This might indicate an issue with the eval data or padding logic.")
        return float("inf"), float("inf") # Return infinite loss/perplexity

    # Calculate average loss over all non-pad tokens
    avg_loss = total_loss / total_tokens
    try:
        perplexity = math.exp(avg_loss)
    except (OverflowError, ValueError):
        perplexity = float("inf") # Handle potential overflow for very high losses

    return avg_loss, perplexity


def setup_environment(args: Namespace):
    """Sets up the device, AMP, and output directory."""
    if not torch.cuda.is_available():
        raise RuntimeError("This model requires CUDA (FlashAttention).")
    local_rank = max(0, getattr(args, "local_rank", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")

    use_amp = args.use_amp and (device.type == "cuda")
    if args.use_amp and not use_amp:
        logger.warning("AMP requested (--use_amp) but CUDA not available. Disabling AMP.")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    return device, use_amp

def load_and_prepare_tokenizer(tokenizer_name: str):
    """
    Load a Hugging Face tokenizer configured for GPT-3–style pretraining.

    Principles:
    - No BOS token (GPT-3 uses <|endoftext|> as a document separator; no explicit BOS).
    - Use EOS as PAD to avoid growing vocab and to keep ignore_index functional.
    - Do NOT grow the vocab unless absolutely necessary (i.e., if EOS is missing).

    Returns:
        tokenizer, vocab_size, pad_token_id, eos_token_id, bos_token_id (None), num_added_toks
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logger.info(f"Tokenizer {tokenizer_name} loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer '{tokenizer_name}': {e}")
        logger.error("Please check the tokenizer name and ensure you have internet connectivity.")
        exit(1)
        
    # make sure we use the fast backend & stop max-length warnings:
    if not getattr(tokenizer, "is_fast", False):
        logger.warning("Loaded a slow Python tokenizer; consider a *fast* tokenizer variant.")
    # disable truncation and any max_len checks:
    try:
        tokenizer.model_max_length = int(1e12)  # effectively "no limit"
    except Exception:
        pass

    num_added_toks = 0

    # Ensure EOS exists. Most GPT-2-family tokenizers already have <|endoftext|>.
    if tokenizer.eos_token is None:
        logger.warning(f"Tokenizer '{tokenizer_name}' lacks an eos token. Adding '[EOS]'.")
        try:
            num_added_toks += tokenizer.add_special_tokens({"eos_token": "[EOS]"})
            logger.info("Added eos_token '[EOS]'.")
        except Exception as e:
            logger.critical(f"Failed to add eos_token: {e}")
            exit(1)

    # Use EOS as PAD (GPT-3 style). This does NOT add a new token.
    if tokenizer.pad_token is None:
        logger.info(f"Using eos_token ('{tokenizer.eos_token}') as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # GPT-3 style: do NOT use a BOS token. Make sure we don't add one.
    # If the tokenizer ships with a BOS, we simply don't use it.
    # (HF allows setting to None; some tokenizers may keep an internal value—it's fine as long as we don't rely on it.)
    try:
        tokenizer.bos_token = None
    except Exception:
        # Some tokenizers may not allow unsetting; safe to ignore since we won't use it.
        pass

    # Final IDs
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id  # expected to be None for GPT-3 style

    if pad_token_id is None or eos_token_id is None:
        logger.critical(
            f"CRITICAL: pad/eos token IDs unresolved (pad={pad_token_id}, eos={eos_token_id})."
        )
        exit(1)

    vocab_size = len(tokenizer)

    logger.info(
        f"Special tokens -> pad: {pad_token_id} ('{tokenizer.pad_token}'), "
        f"eos: {eos_token_id} ('{tokenizer.eos_token}'), bos: {bos_token_id} ({tokenizer.bos_token})"
    )
    logger.info(f"Final tokenizer vocab size: {vocab_size} (added {num_added_toks} token(s))")

    return tokenizer, vocab_size, pad_token_id, eos_token_id, bos_token_id, num_added_toks

def _generate_file_paths(output_dir: str, data_type: str, tokenizer_name: str):
    """Generates consistent file paths for tokenized data and metadata."""
    safe_tokenizer_name = tokenizer_name.replace('/', '_')
    token_file = os.path.join(output_dir, f"{data_type.lower()}_{safe_tokenizer_name}_tokens.npy")
    metadata_file = os.path.join(output_dir, f"{data_type.lower()}_{safe_tokenizer_name}_metadata.json")
    return token_file, metadata_file

def _load_or_tokenize_data(
    corpus_file: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    stride: int,
    output_dir: str,
    data_type: str,  # "Training" or "Evaluation"
    force_retokenize: bool,
    current_vocab_size: int,
    current_tokenizer_name: str,
    current_pad_token_id: int,
    current_eos_token_id: int,
):
    """
    Loads pre-tokenized data or tokenizes if needed, with robust recompute of num_examples
    for the current (max_seq_len, stride). Retokenizes only when tokenizer identity/IDs changed.
    Compatible with both .npy and streamed .bin token files.
    """
    if not corpus_file or not os.path.exists(corpus_file):
        logger.warning(f"{data_type} corpus file not found or not specified: {corpus_file}. Skipping {data_type} data setup.")
        return None, 0

    # Base paths (metadata is always here; token file path may change to .bin)
    token_file_base, metadata_file = _generate_file_paths(output_dir, data_type, current_tokenizer_name)

    num_examples = 0
    needs_retokenize = bool(force_retokenize)

    # --- Try to reuse existing tokens/metadata ---
    if os.path.exists(metadata_file) and not needs_retokenize:
        logger.info(f"Loading pre-tokenized {data_type} data info from {metadata_file}")
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Use the actual token file path recorded in metadata (could be .bin)
            meta_token_file = metadata.get("token_file", token_file_base)
            dtype_name = metadata.get("dtype", "uint32")

            # If recorded token file is missing, we must retokenize
            if not os.path.exists(meta_token_file):
                logger.warning(f"Recorded token file missing: {meta_token_file}. Will retokenize.")
                needs_retokenize = True
            else:
                # Decide if we must retokenize due to tokenizer/vocab/eos changes
                retok_reasons = []
                if metadata.get("tokenizer_name") != current_tokenizer_name:
                    retok_reasons.append(
                        f"tokenizer name (saved='{metadata.get('tokenizer_name')}', current='{current_tokenizer_name}')"
                    )
                if metadata.get("vocab_size") != current_vocab_size:
                    retok_reasons.append(
                        f"vocab size (saved={metadata.get('vocab_size')}, current={current_vocab_size})"
                    )
                if metadata.get("eos_token_id") != current_eos_token_id:
                    retok_reasons.append(
                        f"eos_token_id (saved={metadata.get('eos_token_id')}, current={current_eos_token_id})"
                    )

                if retok_reasons:
                    logger.warning(
                        f"{data_type} token data requires retokenization due to: " + "; ".join(retok_reasons)
                    )
                    needs_retokenize = True
                else:
                    # Recompute num_examples using file length (no loading the whole thing)
                    try:
                        if meta_token_file.endswith(".npy"):
                            tokens_mmap = np.load(meta_token_file, mmap_mode="r")
                            num_tokens_in_array = int(tokens_mmap.shape[0])
                        else:
                            itemsize = np.dtype(dtype_name).itemsize
                            num_tokens_in_array = os.path.getsize(meta_token_file) // itemsize

                        chunk = max_seq_len + 1
                        num_examples = 0 if num_tokens_in_array < chunk else (num_tokens_in_array - chunk) // stride + 1
                        logger.info(
                            f"Recomputed {data_type} num_examples={num_examples} for max_seq_len={max_seq_len}, stride={stride}"
                        )
                        # Hand back the real token file to the caller (dataset)
                        return meta_token_file, num_examples
                    except Exception as e:
                        logger.warning(
                            f"Failed to inspect existing {data_type} tokens for recompute: {e}. Forcing re-tokenization."
                        )
                        needs_retokenize = True

        except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error loading/validating {data_type} metadata {metadata_file}: {e}. Forcing re-tokenization.")
            needs_retokenize = True
            num_examples = 0

    # --- Retokenize when needed or when metadata is missing ---
    if not os.path.exists(metadata_file) or needs_retokenize:
        logger.info(
            f"Preparing {data_type} tokens (retokenize={needs_retokenize}, metadata_exists={os.path.exists(metadata_file)})"
        )

        # Best effort: remove any old token files at the base paths
        try:
            if os.path.exists(token_file_base):
                os.remove(token_file_base)
        except OSError:
            pass
        try:
            # If a .bin counterpart exists from a previous run, remove it too
            bin_candidate = token_file_base[:-4] + ".bin" if token_file_base.endswith(".npy") else token_file_base + ".bin"
            if os.path.exists(bin_candidate):
                os.remove(bin_candidate)
        except OSError:
            pass
        try:
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
        except OSError:
            pass

        # Run pretokenization (streams to .bin and returns actual path + dtype)
        token_file, num_examples, dtype_name = pretokenize_corpus(
            corpus_file,
            tokenizer,
            max_seq_len,
            stride,
            token_file_base,   # base name; function will switch to .bin as needed
            data_type=data_type,
        )
        if num_examples == 0:
            logger.error(
                f"Pre-tokenization resulted in 0 {data_type} examples from {corpus_file}. Check corpus and tokenizer."
            )
            if data_type == "Training":
                exit(1)
            else:
                return None, 0

        # Compute total number of tokens without loading the file
        try:
            if token_file.endswith(".npy"):
                tokens_mmap = np.load(token_file, mmap_mode="r")
                num_tokens = int(tokens_mmap.shape[0])
            else:
                num_tokens = os.path.getsize(token_file) // np.dtype(dtype_name).itemsize
        except Exception as e:
            logger.warning(f"Could not determine {data_type} num_tokens: {e}")
            num_tokens = None

        # Persist metadata with real token file path and dtype
        metadata = {
            "num_examples": num_examples,
            "num_tokens": num_tokens,
            "max_seq_len": max_seq_len,
            "stride": stride,
            "vocab_size": current_vocab_size,
            "tokenizer_name": current_tokenizer_name,
            "pad_token_id": current_pad_token_id,
            "eos_token_id": current_eos_token_id,
            "dtype": dtype_name,
            "format": "bin" if token_file.endswith(".bin") else "npy",
            "token_file": token_file,
        }
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"{data_type} metadata saved to {metadata_file}")
        except IOError as e:
            logger.error(f"Failed to save {data_type} metadata to {metadata_file}: {e}")

        return token_file, num_examples

    # Shouldn’t get here, but in case:
    return None, 0


def prepare_dataloaders(args: Namespace, tokenizer: AutoTokenizer, vocab_size: int, pad_token_id: int, eos_token_id: int):
    """Prepares training and optional evaluation dataloaders."""

    def _resolve_stride(requested: int | None, fallback: int, label: str) -> int:
        stride_val = requested if requested is not None else fallback
        if stride_val <= 0:
            raise SystemExit(f"{label} must be > 0 (received {stride_val}).")
        return stride_val

    train_stride = _resolve_stride(getattr(args, "train_stride", None), args.max_seq_len, "train_stride")
    eval_stride = _resolve_stride(getattr(args, "eval_stride", None), train_stride, "eval_stride")

    world_size = max(1, getattr(args, "world_size", 1))
    rank = max(0, getattr(args, "rank", 0))

    def _build_dataloader(dataset, *, batch_size: int, shuffle: bool, drop_last: bool, is_eval: bool, sampler=None):
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "drop_last": drop_last,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
        else:
            loader_kwargs["shuffle"] = shuffle
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            prefetch = getattr(args, "eval_prefetch_factor", None) if is_eval else getattr(args, "prefetch_factor", None)
            if prefetch is None:
                prefetch = getattr(args, "prefetch_factor", 4)
            loader_kwargs["prefetch_factor"] = max(2, int(prefetch))
        pin_device = getattr(args, "pin_memory_device", None)
        if pin_device and torch.cuda.is_available() and _PIN_MEMORY_DEVICE_SUPPORTED:
            loader_kwargs["pin_memory_device"] = pin_device
        return DataLoader(dataset, **loader_kwargs)

    # --- Training Data ---
    train_sampler = None
    train_token_file, train_num_examples = _load_or_tokenize_data(
        corpus_file=args.corpus_file,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        stride=train_stride,
        output_dir=args.output_dir,
        data_type="Training",
        force_retokenize=args.force_retokenize,
        current_vocab_size=vocab_size,
        current_tokenizer_name=args.tokenizer_name,
        current_pad_token_id = pad_token_id,
        current_eos_token_id = eos_token_id,
    )
    if not train_token_file or train_num_examples == 0:
         logger.critical("Failed to prepare training data. Exiting.")
         exit(1)

    logger.info("Initializing training dataset...")
    try:
        train_dataset = PretokenizedDataset(
            train_token_file, train_num_examples, args.max_seq_len, train_stride, data_type="Train"
        )
        if len(train_dataset) == 0:
             logger.critical("Training dataset has length 0 after initialization. Exiting.")
             exit(1)

        if getattr(args, "distributed", False):
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

        train_dataloader = _build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            is_eval=False,
            sampler=train_sampler,
        )
        logger.info(f"Training dataloader created with {len(train_dataloader)} batches per epoch.")
    except Exception as e:
         logger.critical(f"Failed to initialize training dataset/dataloader: {e}", exc_info=True)
         exit(1)

    # --- Evaluation Data ---
    eval_dataloader = None
    eval_sampler = None
    eval_num_examples = 0
    if args.eval_corpus_file:
        eval_token_file, eval_num_examples = _load_or_tokenize_data(
            corpus_file=args.eval_corpus_file,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            stride=eval_stride,
            output_dir=args.output_dir,
            data_type="Evaluation",
            force_retokenize=args.force_retokenize,
            current_vocab_size=vocab_size,
            current_tokenizer_name=args.tokenizer_name,
            current_pad_token_id = pad_token_id,
            current_eos_token_id = eos_token_id,
        )

        if eval_token_file and eval_num_examples > 0:
            logger.info("Initializing evaluation dataset...")
            try:
                eval_dataset = PretokenizedDataset(
                    eval_token_file, eval_num_examples, args.max_seq_len, eval_stride, data_type="Eval"
                )
                if len(eval_dataset) > 0:
                    if getattr(args, "distributed", False):
                        eval_sampler = DistributedSampler(
                            eval_dataset,
                            num_replicas=world_size,
                            rank=rank,
                            shuffle=False,
                            drop_last=False,
                        )
                    eval_dataloader = _build_dataloader(
                        eval_dataset,
                        batch_size=(args.eval_batch_size if args.eval_batch_size else args.batch_size),
                        shuffle=False,
                        drop_last=False,
                        is_eval=True,
                        sampler=eval_sampler,
                    )
                    logger.info(f"Evaluation dataloader created with {len(eval_dataloader)} batches.")
                else:
                    logger.warning("Evaluation dataset length is 0 after initialization. Skipping evaluation.")
            except Exception as e:
                 logger.error(f"Error initializing evaluation dataset/dataloader: {e}. Skipping evaluation.", exc_info=True)
                 eval_dataloader = None
        elif args.eval_corpus_file: # Log if file was specified but resulted in no data/examples
             logger.warning("Evaluation data could not be prepared (0 examples or file issue). Skipping evaluation.")

    return train_dataloader, eval_dataloader, train_sampler, eval_sampler

def resize_token_embeddings_(model, new_vocab_size: int, pad_idx: int | None):
    """
    Resize token embeddings (and lm_head) to new_vocab_size.
    - Preserves existing weights where possible.
    - Keeps device/dtype.
    - Zeros the PAD row if provided.
    - Re-ties lm_head.weight to embedding weight when applicable.
    """
    old_emb = model.token_embedding
    device, dtype = old_emb.weight.device, old_emb.weight.dtype
    d_model = old_emb.embedding_dim

    # --- new embedding ---
    new_emb = nn.Embedding(new_vocab_size, d_model, padding_idx=pad_idx).to(device=device, dtype=dtype)
    with torch.no_grad():
        n = min(old_emb.num_embeddings, new_vocab_size)
        new_emb.weight[:n].copy_(old_emb.weight[:n])
        if pad_idx is not None and 0 <= pad_idx < new_vocab_size:
            new_emb.weight[pad_idx].zero_()
    model.token_embedding = new_emb

    # --- new lm_head (weight tying safe) ---
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        old_head = model.lm_head
        if old_head.out_features != new_vocab_size:
            new_head = nn.Linear(d_model, new_vocab_size, bias=(old_head.bias is not None)).to(device=device, dtype=dtype)
            with torch.no_grad():
                m = min(old_head.out_features, new_vocab_size)
                new_head.weight[:m].copy_(old_head.weight[:m])
                if old_head.bias is not None:
                    b = min(old_head.bias.shape[0], new_vocab_size)
                    new_head.bias[:b].copy_(old_head.bias[:b])
            model.lm_head = new_head

        # (re)tie weights if head is compatible
        try:
            model.lm_head.weight = model.token_embedding.weight
        except Exception:
            # If your head cannot be exactly tied, you can ignore this.
            pass

def initialize_model(args: Namespace, vocab_size: int, pad_token_id: int, eos_token_id: int,
                     num_added_toks: int, device: torch.device):
    """Initializes the GPT model, handles embedding resizing and optional compilation."""
    logger.info("Initializing model...")
    
    n_kv_head = getattr(args, "n_kv_head", None)
    if n_kv_head is None:
        n_kv_head = compute_default_n_kv_head(args.n_head)
        logger.info(f"Auto-selected n_kv_head={n_kv_head} (n_head={args.n_head}).")
    elif args.n_head % n_kv_head != 0:
        logger.warning(
            f"n_head ({args.n_head}) is not divisible by requested n_kv_head ({n_kv_head}); "
            "adjusting to nearest divisor."
        )
        adjusted = compute_default_n_kv_head(args.n_head)
        logger.warning(f"Setting n_kv_head={adjusted} for compatibility.")
        n_kv_head = adjusted

    mla_config_dict = {
        "num_head": args.n_head,       # d_model / 64 (common head dim)
        "d_c": 64,            # KV compression dimension
        "d_c1": 64,           # Query compression dimension
        "d_rotate": 32,       # Rotary embedding dimension (e.g., d_k / 2)
        # 'max_seq_len' will be added from the main gpt_config inside GPT __init__
    }

    # Configuration for MixtureOfExperts
    moe_config_dict = {
        "d_expert": args.d_model, # Hidden dimension of each expert FFN
        "K": 1,                 # Top-K experts per token
        "N_s": 0,               # Number of shared experts
        "N_r": 2,               # Number of routed experts
        "alpha1": 0.01,         # Expert balance factor
        "alpha2": 0.01,         # Device balance factor
        "alpha3": 0.01,         # Communication balance factor
        "D": max(1, getattr(args, "world_size", 1)),                 # Number of devices (matches distributed world size)
        "M": 1                  # Device limit for routing (can be N_r for D=1)
    }
    
    model_config = {
        "vocab_size": vocab_size,      
        "d_model": args.d_model,
        "n_head": args.n_head,
        "n_kv_head": n_kv_head,
        "n_layer": args.n_layer,
        "max_seq_len": args.max_seq_len,
        "max_kv_len": getattr(args, "max_kv_len", None),
        # "mla_config": mla_config_dict,
        # "moe_config": moe_config_dict,
        "dropout": args.dropout,
        "pad_idx": pad_token_id,    
        "d_ff": args.d_ff,
        "use_gradient_checkpointing": getattr(args, "grad_checkpoint", False),
        "identity_dim": getattr(args, "identity_dim", None),
        "identity_hidden_dim": getattr(args, "identity_hidden_dim", None),
        "identity_dropout": getattr(args, "identity_dropout", 0.1),
        "identity_type": getattr(args, "identity_type", "mlp"),
        "identity_diffusion_steps": getattr(args, "identity_diffusion_steps", 4),
        "identity_sigma_min": getattr(args, "identity_sigma_min", 0.02),
        "identity_sigma_max": getattr(args, "identity_sigma_max", 1.0),
        "identity_t_embed_dim": getattr(args, "identity_t_embed_dim", 128),
    }
    

    if model_config["max_kv_len"]:
        logger.info(f"Using sliding KV cache window of {model_config['max_kv_len']} tokens.")
    if model_config["use_gradient_checkpointing"]:
        logger.info("Gradient checkpointing enabled for Transformer blocks.")

    model = GPT(**model_config).to(device)
    pad_eq_eos = (pad_token_id is not None) and (eos_token_id is not None) and (pad_token_id == eos_token_id)
    with torch.no_grad():
        if (
            pad_token_id is not None
            and 0 <= pad_token_id < model.token_embedding.num_embeddings
            and not pad_eq_eos
        ):
            model.token_embedding.weight[pad_token_id].zero_()
    # If tying is expected:
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        try:
            model.lm_head.weight = model.token_embedding.weight
        except Exception:
            pass
    
    
    logger.info(f"Embeddings: {model.token_embedding.num_embeddings} | tokenizer vocab: {vocab_size}")

    # No need to resize, we already passed vocab_size to GPT
    if num_added_toks > 0:
        logger.info(f"Tokenizer added {num_added_toks} tokens; "
                    f"model initialized with vocab_size={vocab_size}.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {num_params:,} trainable parameters.")
    logger.debug(f"[debug] Trainable parameter count: {num_params:,}")

    return model, model_config



def initialize_training_components(args: Namespace, model: nn.Module, use_amp: bool, pad_token_id: int):
    """Initializes optimizer, loss criterion, and GradScaler."""
    print("learning_rate:", args.learning_rate)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        logger_prefix="[pretrain] ",
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP) with GradScaler.")
    return optimizer, criterion, scaler


def build_identity_optimizer(args: Namespace, model: nn.Module) -> Optional[optim.Optimizer]:
    """
    Create a slower optimizer dedicated to the Identity Block so introspection
    updates do not disturb the main optimizer state.
    """
    base_model = unwrap_model(model)
    identity_block = getattr(base_model, "identity_block", None)
    if identity_block is None:
        return None

    identity_params = [p for p in identity_block.parameters() if p.requires_grad]
    if not identity_params:
        return None

    lr = getattr(args, "identity_lr", None)
    if lr is None:
        lr = args.learning_rate * 0.1

    logger.info(
        "Initializing identity optimizer with %d parameters at lr=%.2e",
        sum(p.numel() for p in identity_params),
        lr,
    )
    return optim.AdamW(identity_params, lr=lr, weight_decay=0.0)


def run_identity_introspection(
    args: Namespace,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    identity_optimizer: Optional[optim.Optimizer],
    pad_token_id: Optional[int],
    eos_token_id: Optional[int],
) -> Optional[dict]:
    """
    Runs the introspection prompt + generation loop and nudges the Identity Block
    toward the generated summary using an auxiliary loss.
    """
    if identity_optimizer is None:
        return None

    prompt = getattr(args, "identity_prompt", "")
    if not prompt or getattr(args, "identity_update_interval", 0) <= 0:
        return None

    base_model = unwrap_model(model)
    identity_block = getattr(base_model, "identity_block", None)
    if identity_block is None:
        return None

    max_new_tokens = max(0, getattr(args, "identity_max_new_tokens", 0))
    if max_new_tokens <= 0:
        return None

    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)
    prompt_ids = prompt_ids[:, -base_model.max_seq_len :]

    pad_id = pad_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    top_k = getattr(args, "identity_top_k", 50)
    if top_k is not None and top_k <= 0:
        top_k = None

    try:
        generated = base_model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=getattr(args, "identity_temperature", 0.7),
            top_k=top_k,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    except Exception as e:
        logger.error(f"Identity introspection generation failed: {e}")
        return None

    if not generated:
        return None

    generated_tensor = torch.tensor(generated[0], device=device).unsqueeze(0)
    attn_mask = torch.ones_like(generated_tensor, device=device)

    with torch.no_grad():
        token_embeddings = base_model.token_embedding(generated_tensor)
        context = IdentityBlock.summarize_token_context(token_embeddings, attn_mask)

    context = context.detach()

    identity_optimizer.zero_grad(set_to_none=True)
    predicted = identity_block(context)
    reconstruction_loss = F.mse_loss(predicted, context)
    scaled_loss = reconstruction_loss * getattr(args, "identity_loss_scale", 1.0)
    scaled_loss.backward()
    identity_optimizer.step()
    identity_optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(reconstruction_loss.item()),
        "text": tokenizer.decode(generated[0]),
    }

def _find_latest_checkpoint(output_dir: str):
    """Finds the latest checkpoint file based on modification time."""
    candidate_patterns = [
        os.path.join(output_dir, "model_step_*.pt"),
        os.path.join(output_dir, "model_best_eval.pt"),
        os.path.join(output_dir, "model_final.pt")
    ]
    latest_checkpoint = None
    latest_mtime = 0

    for pattern in candidate_patterns:
         files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
         if files:
             if os.path.getmtime(files[0]) > latest_mtime:
                 latest_checkpoint = files[0]
                 latest_mtime = os.path.getmtime(files[0])
    return latest_checkpoint


def load_checkpoint(args: Namespace, model: nn.Module, optimizer: optim.Optimizer, scaler: GradScaler | None,
                    device: torch.device, use_amp: bool, current_vocab_size: int, current_pad_token_id: int,
                    current_eos_token_id: int,
                    current_model_config: dict):
    """Loads a checkpoint if specified, handling potential mismatches."""
    start_epoch = 0
    global_step = 0
    best_eval_loss = float("inf")
    checkpoint_path = None
    base_model = unwrap_model(model)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            logger.info("Attempting to find the latest checkpoint...")
            checkpoint_path = _find_latest_checkpoint(args.output_dir)
            if checkpoint_path:
                logger.info(f"Found latest checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"No checkpoints found in {args.output_dir} when searching for 'latest'. Starting fresh.")
        elif os.path.isfile(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
            logger.info(f"Attempting to resume from specified checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"Specified checkpoint path not found: {args.resume_from_checkpoint}. Trying 'latest'.")
            checkpoint_path = _find_latest_checkpoint(args.output_dir)
            if checkpoint_path:
                logger.info(f"Found latest checkpoint instead: {checkpoint_path}")
            else:
                logger.warning(f"No checkpoints found in {args.output_dir}. Starting fresh.")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            logger.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # Load to CPU first

            saved_args = checkpoint.get("args")
            saved_config = checkpoint.get("config") or {}
            mismatched_items = []

            # --- Collect mismatches (no interactive input) ---
            critical_keys = ["d_model", "n_head", "n_layer", "d_ff", "max_seq_len"]
            for key in critical_keys:
                saved_val = saved_config.get(key, getattr(saved_args, key, None) if saved_args else None)
                current_val = current_model_config.get(key, getattr(args, key, None))
                if saved_val is not None and saved_val != current_val:
                    mismatched_items.append(f"{key} (saved={saved_val}, current={current_val})")

            saved_vocab_size = saved_config.get("vocab_size", None)
            if saved_vocab_size is not None and saved_vocab_size != current_vocab_size:
                mismatched_items.append(f"vocab_size (saved={saved_vocab_size}, current={current_vocab_size})")

            saved_tokenizer_name = getattr(saved_args, "tokenizer_name", None) if saved_args else None
            if saved_tokenizer_name is not None and saved_tokenizer_name != args.tokenizer_name:
                mismatched_items.append(f"tokenizer_name (saved='{saved_tokenizer_name}', current='{args.tokenizer_name}')")

            saved_pad_idx = saved_config.get("pad_idx", None)
            if saved_pad_idx is not None and saved_pad_idx != current_pad_token_id:
                mismatched_items.append(f"pad_idx (saved={saved_pad_idx}, current={current_pad_token_id})")

            saved_use_amp = getattr(saved_args, "use_amp", None) if saved_args else None
            if saved_use_amp is not None and use_amp != saved_use_amp:
                mismatched_items.append(f"use_amp (saved={saved_use_amp}, current={use_amp})")

            if mismatched_items:
                logger.warning("\n!!!!!!!!!!!!!!!!!!! CONFIGURATION MISMATCH DETECTED !!!!!!!!!!!!!!!!!!!")
                for item in mismatched_items:
                    logger.warning(f" - {item}")
                logger.warning("Resuming may lead to errors or unexpected behavior.\n")

            # --- Handle vocab mismatch by resizing BEFORE load_state_dict ---
            if saved_vocab_size is not None and saved_vocab_size != current_vocab_size:
                logger.warning(
                    f"Vocab mismatch: saved={saved_vocab_size}, current={current_vocab_size}. "
                    f"Resizing embeddings to current vocab before loading state_dict (strict=False)."
                )
                resize_token_embeddings_(base_model, current_vocab_size, current_pad_token_id)

            # --- Load model weights (strict=False for tied head differences) ---
            state_dict = checkpoint['model_state_dict']
            def _strip_prefix(prefix: str, name: str) -> str:
                return name[len(prefix):] if name.startswith(prefix) else name
            if any(key.startswith("module.") or key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {
                    _strip_prefix("_orig_mod.", _strip_prefix("module.", key)): value
                    for key, value in state_dict.items()
                }
            missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
            if missing_keys: logger.warning(f"Missing keys in model state_dict: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys in model state_dict: {unexpected_keys}")
            logger.info("Model state loaded.")
            model.to(device)  # move to target device AFTER loading

            # --- Optimizer state: only load if vocab sizes match ---
            same_vocab = (saved_vocab_size is None) or (saved_vocab_size == current_vocab_size)
            if same_vocab and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded.")
                    # Reset LR to CLI value (avoid inheriting old LR)
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.learning_rate
                except ValueError as e:
                    logger.warning(f"Could not load optimizer state (likely shape change): {e}. Re-initializing optimizer.")
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    optimizer = _build_optimizer(
                        trainable_params,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        logger_prefix="[pretrain] ",
                    )
            else:
                if not same_vocab:
                    logger.info("Skipping optimizer state load due to vocab mismatch. Re-initializing optimizer.")
                else:
                    logger.info("Optimizer state not found. Re-initializing optimizer.")
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = _build_optimizer(
                    trainable_params,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    logger_prefix="[pretrain] ",
                )

            # --- Load training progress ---
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))

            # --- GradScaler ---
            if use_amp and scaler:
                if 'scaler_state_dict' in checkpoint:
                    try:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        logger.info("GradScaler state loaded.")
                    except Exception as e:
                        logger.warning(f"Could not load GradScaler state: {e}. Re-initializing.")
                        scaler = GradScaler()
                else:
                    logger.warning("No scaler state found. Initializing a new GradScaler.")
                    scaler = GradScaler()

            logger.info(f"Resuming from Epoch {start_epoch + 1}, Global Step {global_step}")
            if 'loss' in checkpoint: logger.info(f"Loss at checkpoint: {checkpoint['loss']:.4f}")
            if best_eval_loss != float('inf'): logger.info(f"Best eval loss so far: {best_eval_loss:.4f}")

        except FileNotFoundError:
            logger.error(f"Checkpoint file {checkpoint_path} disappeared unexpectedly. Starting fresh.")
            start_epoch, global_step, best_eval_loss = 0, 0, float('inf')
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}. Starting fresh.", exc_info=True)
            start_epoch, global_step, best_eval_loss = 0, 0, float('inf')
            model, _ = initialize_model(
                args, current_vocab_size, current_pad_token_id, current_eos_token_id, 0, device
            )
            optimizer, _, scaler = initialize_training_components(args, model, use_amp, current_pad_token_id)
    else:
        logger.info("No checkpoint specified or found. Starting training from scratch.")

    return start_epoch, global_step, best_eval_loss, model, optimizer, scaler

def save_checkpoint(args: Namespace, epoch: int, global_step: int, model: nn.Module, optimizer: optim.Optimizer,
                      scaler: Optional[GradScaler], current_loss: float, best_eval_loss: float, model_config: dict,
                      wandb_run_id: Optional[str] = None, # W&B run ID for traceability
                      is_best: bool = False, is_final: bool = False):
    """
    Saves a training checkpoint and logs it to Weights & Biases as an artifact if enabled.
    """
    model_to_save = unwrap_model(model)
    save_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": current_loss,
        "best_eval_loss": best_eval_loss,
        "args": vars(args) if isinstance(args, Namespace) else args, # Save args as dict
        "config": model_config,
        "wandb_run_id": wandb_run_id, # Store W&B run ID for resuming W&B run
    }
    if scaler is not None: # Ensure scaler is not None before accessing state_dict
        save_dict["scaler_state_dict"] = scaler.state_dict()

    checkpoint_base_name = ""
    # Aliases will be dynamically built
    current_aliases = ["latest", f"step_{global_step}"]

    if is_final:
        checkpoint_base_name = "model_final.pt"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_base_name)
        logger.info(f"Saving final model state to {checkpoint_path}")
        current_aliases.append("final")
    elif is_best:
        checkpoint_base_name = "model_best_eval.pt"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_base_name)
        logger.info(f"Saving best evaluation model to {checkpoint_path}")
        current_aliases.append("best_eval")
    else: # Regular step checkpoint
        checkpoint_base_name = f"model_step_{global_step}.pt"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_base_name)
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        # No extra specific alias for regular step other than step_X and latest

    try:
        os.makedirs(args.output_dir, exist_ok=True) # Ensure directory exists
        torch.save(save_dict, checkpoint_path)
        logger.info(f"Checkpoint successfully saved to {checkpoint_path}")

        # Clean old step checkpoints only for regular saves (not best, not final)
        if not is_best and not is_final and args.keep_last_n_checkpoints > 0:
            # Get all step checkpoints, sort by step number (descending, so newest are first)
            all_step_files = sorted(
                glob.glob(os.path.join(args.output_dir, "model_step_*.pt")),
                key=lambda f: int(f.split('_')[-1].split('.')[0]) if f.split('_')[-1].split('.')[0].isdigit() else -1, # Extract step number
                reverse=True # Newest (highest step number) first
            )

            if len(all_step_files) > args.keep_last_n_checkpoints:
                checkpoints_to_delete = all_step_files[args.keep_last_n_checkpoints:] # Files to delete are the oldest ones
                for old_ckpt in checkpoints_to_delete:
                    try:
                        os.remove(old_ckpt)
                        logger.info(f"Removed old step checkpoint: {old_ckpt}")
                    except OSError as e:
                        logger.warning(f"Error removing old step checkpoint {old_ckpt}: {e}")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {checkpoint_path}: {e}", exc_info=True)


def run_debug_generation(args: Namespace, model: nn.Module, tokenizer: AutoTokenizer, device: torch.device,
                         bos_token_id: int | None, pad_token_id: int, use_amp: bool, global_step: int,
                         epoch: int | None = None, step_in_epoch: int | None = None):
    """Runs a debug generation example."""
    base_model = unwrap_model(model)
    if not (hasattr(base_model, 'generate') and callable(getattr(base_model, 'generate'))):
        logger.warning("Model does not have a 'generate' method. Skipping debug generation.")
        return

    if epoch is not None:
        logger.info(f"\n--- Running Debug Generation @ Epoch {epoch}, Step {global_step} ---")
    else:
        logger.info(f"\n--- Running Debug Generation @ Step {global_step} ---")
    start_gen_time = time.time()
    was_training = model.training
    model.eval()

    try:
        prompt = args.debug_generate_prompt
        # GPT-3 style: no BOS; and be explicit about no special tokens
        input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
        
        if not input_ids_list:
             logger.warning("Debug prompt encoded to an empty sequence. Skipping generation.")
             return

        input_tensor = torch.tensor([input_ids_list], dtype=torch.long).to(device)

        gen_kwargs = {
            "max_new_tokens": args.debug_max_new_tokens,
            "temperature": args.debug_temperature,
            "top_k": args.debug_top_k if args.debug_top_k > 0 else None,
            "pad_token_id": pad_token_id,
            # Assuming model.generate handles sampling logic based on temp/top_k
        }

        with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
            generated_ids_tensor = base_model.generate(input_tensor, **gen_kwargs) # Shape: [1, TotalSeqLen]

        # Decode the generated part
        generated_ids = generated_ids_tensor[0]
        # Handle potential case where generate only returns new tokens vs full sequence
        if generated_ids[:len(input_ids_list)] == input_ids_list:
             newly_generated_ids = generated_ids[len(input_ids_list):]
        else:
             # Assume it returned the whole sequence or only new tokens if it doesn't start with prompt
             # This part might need adjustment based on specific model.generate behavior
             logger.debug("Assuming generate output contains full sequence or doesn't match prompt start.")
             newly_generated_ids = generated_ids # Adjust if needed

        generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)

        logger.info(f'Prompt: "{prompt}"') # Show original prompt
        logger.info(f"Generated Text:\n{generated_text}")

        # Persist prompt/completion pair for reporting
        try:
            output_dir = getattr(args, "output_dir", None)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                csv_path = os.path.join(output_dir, "debug_generations.csv")
                append_header = not os.path.exists(csv_path)
                with open(csv_path, "a", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    if append_header:
                        writer.writerow(["timestamp", "epoch", "step_in_epoch", "global_step", "prompt", "completion"])
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        epoch if epoch is not None else "",
                        step_in_epoch if step_in_epoch is not None else "",
                        global_step,
                        prompt,
                        generated_text,
                    ])
                logger.info(f"Logged prompt/completion pair to {csv_path}")
            else:
                logger.warning("No output_dir provided; skipping CSV logging for debug generation.")
        except Exception as csv_err:
            logger.error(f"Failed to write debug generation CSV entry: {csv_err}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during debug generation: {e}", exc_info=True)
    finally:
        if was_training: model.train() # Restore mode
        gen_time = time.time() - start_gen_time
        logger.info(f"--- Debug Generation Finished ({gen_time:.2f}s) ---")


class CUDAGraphTrainStep:
    """
    Wraps the micro-step (forward + loss + backward) inside a CUDA graph so we can
    replay the captured work each iteration. Only safe when GradScaler is disabled.
    """

    def __init__(self, model, criterion, device, vocab_size, gradient_accumulation_steps: int, use_amp: bool):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA.")
        self.model = model
        self.criterion = criterion
        self.device = device
        self.vocab_size = vocab_size
        self.grad_accum_steps = max(1, int(gradient_accumulation_steps))
        self.use_amp = use_amp

        self.graph = None
        self.static_input = None
        self.static_labels = None
        self.loss_tensor = torch.zeros(1, device=device)
        self.captured = False

    def __call__(self, batch):
        try:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
        except KeyError as e:
            logger.error(f"Missing key in batch for CUDA graph replay: {e}.")
            return None, None

        if not self.captured:
            return self._capture(input_ids, labels)
        return self._replay(input_ids, labels)

    def _ensure_static_buffers(self, input_ids: torch.Tensor, labels: torch.Tensor):
        if self.static_input is None or self.static_input.shape != input_ids.shape:
            self.static_input = torch.empty_like(input_ids, device=self.device)
        if self.static_labels is None or self.static_labels.shape != labels.shape:
            self.static_labels = torch.empty_like(labels, device=self.device)

    def _capture(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self._ensure_static_buffers(input_ids, labels)
        self.static_input.copy_(input_ids)
        self.static_labels.copy_(labels)
        self.loss_tensor.zero_()

        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.model.zero_grad(set_to_none=True)

        with torch.cuda.graph(self.graph):
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass
            with autocast("cuda", enabled=self.use_amp):
                logits, _ = self.model(self.static_input)
                loss = self.criterion(logits.view(-1, self.vocab_size), self.static_labels.view(-1))
            loss_scaled = loss / self.grad_accum_steps
            loss_scaled.backward()
            self.loss_tensor.copy_(loss.detach())

        self.captured = True
        logger.info("Captured CUDA graph of the train micro-step; replay enabled.")
        return float(self.loss_tensor.item()), None

    def _replay(self, input_ids: torch.Tensor, labels: torch.Tensor):
        if input_ids.shape != self.static_input.shape or labels.shape != self.static_labels.shape:
            logger.error("Batch shape mismatch detected during CUDA graph replay.")
            return None, None
        self.static_input.copy_(input_ids)
        self.static_labels.copy_(labels)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        self.graph.replay()
        return float(self.loss_tensor.item()), None


def train_step(model, batch, criterion, scaler, device, use_amp, vocab_size,
               gradient_accumulation_steps: int, cudagraph_runner: CUDAGraphTrainStep | None = None):
    """
    Performs a single forward pass, calculates loss, scales it for accumulation,
    and performs backward pass. Does NOT step optimizer or zero grads.
    """
    if cudagraph_runner is not None:
        return cudagraph_runner(batch)

    try:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
    except KeyError as e:
         logger.error(f"Missing key in batch: {e}. Check dataset __getitem__.")
         return None, None # Indicate failure
    except Exception as e:
         logger.error(f"Error moving batch to device: {e}")
         return None, None # Indicate failure

    if input_ids.shape[0] == 0:
         logger.warning("Encountered empty batch. Skipping.")
         return None, None # Indicate skippable step

    try:
        torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        # Older PyTorch versions may not have this; safe to ignore
        pass

    try:
        from torch.amp import autocast
        with autocast("cuda", enabled=use_amp):
            logits, _ = model(input_ids) 
            # Calculate raw loss (unscaled)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss detected during forward. Skipping backward for this batch.")
            return None, loss.item()

        loss_scaled = loss / gradient_accumulation_steps

    except Exception as e:
        logger.error(f"Error during forward/loss calculation: {e}", exc_info=True)
        return None, None

    try:
        if scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        return loss.item(), None

    except Exception as e:
         logger.error(f"Error during backward pass: {e}", exc_info=True)
         return None, loss.item()

def train(args: Namespace):
    """Main function to orchestrate the training process with gradient accumulation."""

    init_distributed_mode(args)
    if args.distributed and not args.is_main_process:
        args.disable_wandb = True

    # --- Weights & Biases Setup ---
    if args.is_main_process and not getattr(args, 'disable_wandb', False): # Check if disable_wandb arg exists
        try:
            # Try to get wandb_run_id from args if resuming a specific run
            wandb_run_id = getattr(args, 'wandb_run_id', None)
            if wandb.run is None: # Ensure init is called only once if already in a wandb context
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    config=vars(args),
                    resume="allow",
                    id=wandb_run_id
                )
            # If resuming, and wandb_run_id was part of the loaded checkpoint, args.wandb_run_id should be updated.
            # For simplicity, we assume if wandb_run_id is set in args, it's the one to use.
            if wandb.run: # Check if init was successful
                 logger.info(f"Weights & Biases initialized. Run URL: {wandb.run.get_url()}")
                 # Save the main script to W&B
                 # Ensure __file__ is defined; this might not work correctly if train() is called from an interactive session directly
                 if '__file__' in globals():
                    wandb.save(os.path.abspath(__file__))
                 else:
                    logger.warning("Could not save script to W&B: __file__ not defined. This can happen in interactive environments.")

        except Exception as e:
            logger.error(f"Could not initialize Weights & Biases: {e}", exc_info=True)
            args.disable_wandb = True # Disable W&B if initialization fails
    elif args.is_main_process:
        logger.info("Weights & Biases logging is disabled.")


    # --- Setup ---
    device, use_amp = setup_environment(args)
    # logger = logging.getLogger(__name__) # Already defined globally or passed

    # --- Tokenizer ---
    tokenizer, vocab_size, pad_token_id, eos_token_id, bos_token_id, num_added_toks = load_and_prepare_tokenizer(args.tokenizer_name)

    # --- Data ---
    train_dataloader, eval_dataloader, train_sampler, eval_sampler = prepare_dataloaders(args, tokenizer, vocab_size, pad_token_id, eos_token_id)

    # --- Model ---
    model, model_config = initialize_model(args, vocab_size, pad_token_id, eos_token_id, num_added_toks, device)

    # --- Optimizer, Loss, Scaler ---
    optimizer, criterion, scaler = initialize_training_components(args, model, use_amp, pad_token_id)

    # --- Weights & Biases Watch Model (after model and criterion init) ---
    # if not getattr(args, 'disable_wandb', False) and wandb.run:
    #     watch_log_freq = args.log_interval * args.gradient_accumulation_steps if args.log_interval > 0 else 1000
    #     wandb.watch(model, criterion=criterion, log=None, log_freq=watch_log_freq)


    # --- Checkpoint Loading ---
    # If load_checkpoint could return a wandb_run_id, you'd handle it before wandb.init or pass it to wandb.init
    start_epoch, global_step, best_eval_loss, model, optimizer, scaler = load_checkpoint(
        args, model, optimizer, scaler, device, use_amp, vocab_size, pad_token_id, eos_token_id, model_config
    )
    identity_optimizer = build_identity_optimizer(args, model)
    # If load_checkpoint returned a wandb_run_id, and W&B init happened after, you might re-init or ensure args.wandb_run_id was set prior.
    # For this example, we assume args.wandb_run_id is set externally if specific run resumption is needed.

    # Surface the effective model size right before training kicks off.
    current_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Starting training with %s trainable parameters (%.3fB).",
        f"{current_trainable_params:,}",
        current_trainable_params / 1e9,
    )

    if args.compile_model and not args.debug:
        compile_kwargs = {"mode": "max-autotune", "fullgraph": True}
        try:
            logger.info("Compiling model with torch.compile (mode=max-autotune, fullgraph=True)...")
            model = torch.compile(model, **compile_kwargs)
            logger.info("Model compiled successfully with tuned settings.")
        except Exception as tuned_exc:
            logger.warning(f"torch.compile with tuned settings failed: {tuned_exc}. Retrying with defaults.")
            try:
                model = torch.compile(model)
                logger.info("Model compiled successfully with default torch.compile settings.")
            except Exception as e:
                logger.warning(f"torch.compile failed entirely: {e}. Continuing without compilation.")

    if getattr(args, "distributed", False):
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    # --- Gradient Accumulation Setup ---
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 32)
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
        logger.warning(f"Invalid gradient_accumulation_steps: {gradient_accumulation_steps}. Setting to 1.")
        gradient_accumulation_steps = 1
    args.gradient_accumulation_steps = gradient_accumulation_steps # Ensure args has the final value
    if gradient_accumulation_steps > 1:
         logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps.")
         logger.info(f"Effective batch size: {args.batch_size * gradient_accumulation_steps}")

    cudagraph_runner = None
    if getattr(args, "use_cuda_graphs", False):
        if scaler is not None:
            raise RuntimeError("--use_cuda_graphs requires AMP to be disabled (GradScaler must be None).")
        if getattr(args, "distributed", False):
            raise RuntimeError("--use_cuda_graphs is not supported together with --distributed.")
        cudagraph_runner = CUDAGraphTrainStep(
            model=model,
            criterion=criterion,
            device=device,
            vocab_size=vocab_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
        )


    # --- Training Loop ---
    logger.info(f"\nStarting training from epoch {start_epoch + 1}...")
    model.train()
    total_loss_accum_for_log = 0.0
    micro_steps_in_log_period = 0
    can_toggle_sync = hasattr(model, "require_backward_grad_sync")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.is_main_process:
            logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        progress_iter = enumerate(train_dataloader)
        if args.is_main_process:
            progress_bar = tqdm(progress_iter, total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=True, dynamic_ncols=True)
        else:
            progress_bar = progress_iter
        model.train()
        steps_in_epoch = 0

        for batch_idx, batch in progress_bar:
            is_final_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
            is_last_batch_in_epoch = (batch_idx + 1) == len(train_dataloader)

            do_sync = is_final_accumulation_step or is_last_batch_in_epoch
            if can_toggle_sync:
                model.require_backward_grad_sync = do_sync

            step_loss_unscaled, error_info = train_step(
                model, batch, criterion, scaler, device, use_amp, vocab_size,
                gradient_accumulation_steps, cudagraph_runner=cudagraph_runner
            )

            if step_loss_unscaled is not None:
                total_loss_accum_for_log += step_loss_unscaled
                micro_steps_in_log_period += 1
            elif error_info is not None:
                 total_loss_accum_for_log += error_info
                 micro_steps_in_log_period += 1
                 logger.warning(f"Step {batch_idx+1}: Handled non-fatal error in train_step. Loss: {error_info:.4f}")

            if (is_final_accumulation_step or is_last_batch_in_epoch) and step_loss_unscaled is not None :
                current_micro_batch_loss = step_loss_unscaled

                try:
                    if scaler:
                        scaler.unscale_(optimizer)
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        scale_after = scaler.get_scale()
                        if scale_before > scale_after:
                            logger.warning(f"Gradient overflow detected at step {global_step + 1}. Scale reduced from {scale_before:.1f} to {scale_after:.1f}.")
                            if args.is_main_process and not getattr(args, 'disable_wandb', False) and wandb.run:
                                wandb.log({"train/grad_overflow": 1, "train/amp_scale_old": scale_before, "train/amp_scale_new": scale_after}, step=global_step)
                    else:
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    steps_in_epoch += 1

                    identity_log = None
                    if (
                        identity_optimizer is not None
                        and getattr(args, "identity_update_interval", 0) > 0
                        and global_step % args.identity_update_interval == 0
                    ):
                        identity_log = run_identity_introspection(
                            args,
                            model,
                            tokenizer,
                            device,
                            identity_optimizer,
                            pad_token_id,
                            eos_token_id,
                        )
                        if identity_log:
                            if args.is_main_process:
                                msg = (
                                    f"Identity introspection ({global_step}): "
                                    f"loss={identity_log['loss']:.4f} | text='{identity_log['text'][:120]}'"
                                )
                                logger.info(msg)
                            if (
                                args.is_main_process
                                and not getattr(args, "disable_wandb", False)
                                and wandb.run
                            ):
                                wandb.log(
                                    {
                                        "identity/loss": identity_log["loss"],
                                        "identity/text": identity_log["text"],
                                    },
                                    step=global_step,
                                )

                    # --- Logging ---
                    if args.log_interval > 0 and global_step > 0 and global_step % args.log_interval == 0:
                        avg_loss_log_period = float('nan')
                        perplexity = float('nan')
                        if micro_steps_in_log_period > 0:
                            avg_loss_log_period = total_loss_accum_for_log / micro_steps_in_log_period
                            try:
                                perplexity = math.exp(avg_loss_log_period) if avg_loss_log_period < 700 else float("inf")
                            except (OverflowError, ValueError):
                                perplexity = float("inf")

                            log_postfix = {
                                "Loss": f"{avg_loss_log_period:.4f}", "PPL": f"{perplexity:.2f}",
                                "Step": global_step, "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                            }
                            if scaler: log_postfix["Scale"] = f"{scaler.get_scale():.1f}"
                            if args.is_main_process and hasattr(progress_bar, "set_postfix"):
                                progress_bar.set_postfix(log_postfix)

                            # --- Weights & Biases Logging ---
                            if args.is_main_process and not getattr(args, 'disable_wandb', False) and wandb.run:
                                log_data = {
                                    "train/loss": avg_loss_log_period,
                                    "train/perplexity": perplexity,
                                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                                    "epoch": epoch + 1,
                                    "processed_samples": global_step * args.batch_size * gradient_accumulation_steps
                                }
                                if scaler:
                                    log_data["train/amp_scale"] = scaler.get_scale()
                                wandb.log(log_data, step=global_step)

                            total_loss_accum_for_log = 0.0
                            micro_steps_in_log_period = 0
                        else: # Log interval hit, but no successful micro-steps in period
                            log_postfix = {"Step": global_step, "LR": f"{optimizer.param_groups[0]['lr']:.2e}"}
                            if scaler: log_postfix["Scale"] = f"{scaler.get_scale():.1f}"
                            if 'current_micro_batch_loss' in locals() and current_micro_batch_loss is not None:
                                log_postfix["LastLoss"] = f"{current_micro_batch_loss:.4f}"
                            if args.is_main_process and hasattr(progress_bar, "set_postfix"):
                                progress_bar.set_postfix(log_postfix)

                            if args.is_main_process and not getattr(args, 'disable_wandb', False) and wandb.run:
                                log_data = {
                                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                                    "epoch": epoch + 1,
                                     "processed_samples": global_step * args.batch_size * gradient_accumulation_steps
                                }
                                if scaler: log_data["train/amp_scale"] = scaler.get_scale()
                                if 'current_micro_batch_loss' in locals() and current_micro_batch_loss is not None:
                                     log_data["train/last_micro_batch_loss_at_log"] = current_micro_batch_loss
                                wandb.log(log_data, step=global_step)


                    # --- Evaluation Step ---
                    if eval_dataloader and args.eval_interval > 0 and global_step % args.eval_interval == 0 and global_step > 0:
                        start_eval_time = time.time()
                        try:
                            if eval_sampler is not None:
                                eval_sampler.set_epoch(global_step)
                            eval_loss, eval_perplexity = evaluate(model, eval_dataloader, criterion, device, pad_token_id, use_amp, args)
                            eval_time_taken = time.time() - start_eval_time
                            if args.is_main_process:
                                logger.info(f"\n--- Evaluation @ Step {global_step} Finished ({eval_time_taken:.2f}s) ---")
                                logger.info(f"  Eval Loss: {eval_loss:.4f} | Eval Perplexity: {eval_perplexity:.2f}")

                            # --- Weights & Biases Logging ---
                            if args.is_main_process and not getattr(args, 'disable_wandb', False) and wandb.run:
                                wandb.log({
                                    "eval/loss": eval_loss,
                                    "eval/perplexity": eval_perplexity,
                                    "eval/time_seconds": eval_time_taken
                                }, step=global_step)

                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                if args.is_main_process:
                                    logger.info(f"  New best evaluation loss: {best_eval_loss:.4f}")
                                    if not getattr(args, 'disable_wandb', False) and wandb.run:
                                        wandb.summary["best_eval_loss"] = best_eval_loss
                                        wandb.summary["best_eval_perplexity"] = eval_perplexity
                                        wandb.summary["best_eval_step"] = global_step

                                    current_wandb_run_id = wandb.run.id if not getattr(args, 'disable_wandb', False) and wandb.run else None
                                    save_loss_for_ckpt = avg_loss_log_period if micro_steps_in_log_period == 0 and 'avg_loss_log_period' in locals() else current_micro_batch_loss # Check if avg_loss_log_period was just reset
                                    if math.isnan(save_loss_for_ckpt) and 'current_micro_batch_loss' in locals(): save_loss_for_ckpt = current_micro_batch_loss # fallback
                                    save_checkpoint(args, epoch, global_step, model, optimizer, scaler,
                                                    save_loss_for_ckpt, best_eval_loss, model_config, is_best=True, wandb_run_id=current_wandb_run_id)
                        except Exception as e:
                             logger.error(f"Error during evaluation run at step {global_step}: {e}", exc_info=True)
                        finally:
                            model.train()

                    # --- Checkpoint Saving (Regular Interval) ---
                    if args.save_interval > 0 and global_step % args.save_interval == 0 and global_step > 0 and args.is_main_process:
                        current_wandb_run_id = wandb.run.id if not getattr(args, 'disable_wandb', False) and wandb.run else None
                        save_loss_for_ckpt = avg_loss_log_period if micro_steps_in_log_period == 0 and 'avg_loss_log_period' in locals() else current_micro_batch_loss
                        if math.isnan(save_loss_for_ckpt) and 'current_micro_batch_loss' in locals(): save_loss_for_ckpt = current_micro_batch_loss
                        save_checkpoint(args, epoch, global_step, model, optimizer, scaler,
                                        save_loss_for_ckpt, best_eval_loss, model_config, is_best=False, wandb_run_id=current_wandb_run_id)

                    # --- Periodic Debug Generation ---
                    if args.is_main_process and args.debug_generate_interval > 0 and global_step % args.debug_generate_interval == 0 and global_step > 0:
                         run_debug_generation(args, model, tokenizer, device, bos_token_id, pad_token_id, use_amp, global_step, epoch + 1, steps_in_epoch)
                         model.train()

                except Exception as e:
                    logger.error(f"Error during optimizer step/logging/eval/saving at micro-batch {batch_idx+1}, global step {global_step}: {e}", exc_info=True)
                    optimizer.zero_grad(set_to_none=True)

            elif step_loss_unscaled is None:
                 if error_info is not None:
                     logger.warning(f"Zeroing gradients after failed step {batch_idx+1} due to error.")
                     optimizer.zero_grad(set_to_none=True)
                 continue

        if args.is_main_process:
            logger.info(f"Epoch {epoch+1} completed. Global Step: {global_step}")

    # --- Final Actions ---
    if args.is_main_process:
        logger.info("\nTraining finished.")

    final_model_state_dict = None
    if args.is_main_process:
        final_eval_loss_to_save = best_eval_loss
        best_model_path = os.path.join(args.output_dir, "model_best_eval.pt")
        map_location = 'cpu'

        if os.path.exists(best_model_path):
            try:
                logger.info(f"Loading best evaluation model state from {best_model_path} to save as final model.")
                best_checkpoint = torch.load(best_model_path, map_location=map_location)
                final_model_state_dict = best_checkpoint['model_state_dict']
                final_eval_loss_to_save = best_checkpoint.get('eval_loss', best_eval_loss)
                logger.info(f"(Best evaluation loss achieved: {final_eval_loss_to_save:.4f})")
            except Exception as e:
                logger.warning(f"Could not load best evaluation model state for final save: {e}. Saving current model state instead.")
                model_to_save = unwrap_model(model)
                model_to_save.cpu()
                final_model_state_dict = model_to_save.state_dict()
                model_to_save.to(device)
        else:
            logger.info("Best evaluation checkpoint not found. Saving current model state as final model.")
            model_to_save = unwrap_model(model)
            model_to_save.cpu()
            final_model_state_dict = model_to_save.state_dict()
            model_to_save.to(device)

        if final_model_state_dict:
            final_save_content = {
                "model_state_dict": final_model_state_dict, "config": model_config,
                "args": { "tokenizer_name": args.tokenizer_name, "d_model": args.d_model, "n_head": args.n_head, "n_layer": args.n_layer, "d_ff": args.d_ff, "max_seq_len": args.max_seq_len, },
                "tokenizer_info": { "name": args.tokenizer_name, "vocab_size": vocab_size, "pad_token_id": pad_token_id, "eos_token_id": eos_token_id, "bos_token_id": bos_token_id, "num_added_toks": num_added_toks, },
                "best_eval_loss": final_eval_loss_to_save, "final_global_step": global_step
            }
            final_model_path = os.path.join(args.output_dir, "model_final_slim.pt")
            try:
                 torch.save(final_save_content, final_model_path)
                 logger.info(f"Final model state dict and config saved to {final_model_path}")
            except Exception as e:
                 logger.error(f"Error saving final slim model: {e}", exc_info=True)

        if args.generate_example and final_model_state_dict: # ensure final_model_state_dict exists
            logger.info("Running final example generation...")
            model.load_state_dict(final_model_state_dict)
            model.to(device)
            model.eval()
            final_gen_args = Namespace(
                output_dir = args.output_dir,
                debug_generate_prompt = getattr(args, 'debug_generate_prompt', "[SOS]" if bos_token_id is not None else "The"),
                debug_max_new_tokens = getattr(args, 'debug_max_new_tokens', 50),
                debug_temperature = getattr(args, 'debug_temperature', 0.7),
                debug_top_k = getattr(args, 'debug_top_k', 50)
            )
            try:
                final_epoch_number = (epoch + 1) if 'epoch' in locals() else None
                final_step_in_epoch = locals().get("steps_in_epoch", None)
                run_debug_generation(final_gen_args, model, tokenizer, device, bos_token_id, pad_token_id, use_amp, global_step, final_epoch_number, final_step_in_epoch)
            except Exception as e:
                logger.error(f"Error during final debug generation: {e}", exc_info=True)

    if args.is_main_process:
        logger.info("Training script finished.")

    # --- Weights & Biases Finish ---
    if args.is_main_process and not getattr(args, 'disable_wandb', False) and wandb.run:
        wandb.summary["completed_epochs"] = epoch + 1 # Total epochs trained
        wandb.finish()

    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a GPT model using Hugging Face Tokenizer")

    # --- Data/Tokenizer Arguments ---
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--eval_corpus_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./gpt_pretrain_output_hf")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--force_retokenize", action="store_true")
    parser.add_argument("--train_stride", type=int, default=None,
                        help="Stride between successive training examples (defaults to max_seq_len).")
    parser.add_argument("--eval_stride", type=int, default=None,
                        help="Stride between evaluation examples (defaults to train stride).")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="Prefetch factor for the training DataLoader when num_workers > 0.")
    parser.add_argument("--eval_prefetch_factor", type=int, default=None,
                        help="Prefetch factor for the evaluation DataLoader when num_workers > 0. Defaults to training value.")
    parser.add_argument("--pin_memory_device", type=str, default="cuda",
                        help="Device string to pin DataLoader memory to (requires torch>=2.0).")

    # --- Model Architecture Arguments ---
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_head", type=int, default=None,
                        help="Number of KV heads for GQA (defaults to ~n_head/4, adjusted to divide n_head).")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_kv_len", type=int, default=None,
                        help="Limit the KV cache/attention window length (enables sliding-window attention).")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Enable torch.utils.checkpoint across Transformer blocks to save activation memory.")
    parser.add_argument("--identity_dim", type=int, default=None,
                        help="Enable the Identity Block by providing its latent state dimension (None disables it).")
    parser.add_argument("--identity_hidden_dim", type=int, default=None,
                        help="Hidden size of the Identity Block controller MLP (defaults to 2x state dim).")
    parser.add_argument("--identity_dropout", type=float, default=0.1,
                        help="Dropout probability inside the Identity Block controller.")
    parser.add_argument("--identity_type", type=str, default="mlp", choices=["mlp", "diffusion"],
                        help="Choose between the classic MLP identity block and the diffusion identity block.")
    parser.add_argument("--identity_diffusion_steps", type=int, default=4,
                        help="Number of denoising steps for the diffusion identity block.")
    parser.add_argument("--identity_sigma_min", type=float, default=0.02,
                        help="Minimum noise level (sigma) for the diffusion identity block schedule.")
    parser.add_argument("--identity_sigma_max", type=float, default=1.0,
                        help="Maximum noise level (sigma) for the diffusion identity block schedule.")
    parser.add_argument("--identity_t_embed_dim", type=int, default=128,
                        help="Timestep embedding dimension for the diffusion identity block.")

    # --- Training Arguments ---
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Number of steps to accumulate gradients before an optimizer step.')
    parser.add_argument("--use_cuda_graphs", action="store_true",
                        help="Capture the train micro-step with CUDA graphs for lower launch overhead (requires --use_amp disabled).")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable multi-GPU training via torch.distributed (auto-enabled if WORLD_SIZE>1).")
    parser.add_argument("--dist_backend", type=str, default="nccl",
                        help="Backend to use for torch.distributed.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank provided by torchrun (set automatically).")

    # --- Identity Loop Arguments ---
    parser.add_argument("--identity_update_interval", type=int, default=0,
                        help="Run an introspection update every N optimizer steps (0 disables the loop).")
    parser.add_argument("--identity_prompt", type=str,
                        default="Who am I? Where do I want to go?",
                        help="Prompt used to trigger the introspection generation pass.")
    parser.add_argument("--identity_max_new_tokens", type=int, default=64,
                        help="Number of tokens to sample during introspection generation.")
    parser.add_argument("--identity_temperature", type=float, default=0.7,
                        help="Sampling temperature for introspection generation.")
    parser.add_argument("--identity_top_k", type=int, default=50,
                        help="Top-k sampling filter for introspection generation (<=0 disables filtering).")
    parser.add_argument("--identity_lr", type=float, default=None,
                        help="Learning rate for the Identity Block optimizer (defaults to 0.1x main LR).")
    parser.add_argument("--identity_loss_scale", type=float, default=1.0,
                        help="Multiplier applied to the introspection reconstruction loss.")

    # --- Logging, Checkpointing, Evaluation Arguments ---
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, nargs="?", const="latest", default=None)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=2)

    # --- Generation Arguments ---
    parser.add_argument("--generate_example", action="store_true")
    parser.add_argument("--debug_generate_interval", type=int, default=100)
    parser.add_argument("--debug_generate_prompt", type=str, default="The meaning of life is")
    parser.add_argument("--debug_max_new_tokens", type=int, default=256)
    parser.add_argument("--debug_temperature", type=float, default=0.7)
    parser.add_argument("--debug_top_k", type=int, default=50)

    # --- Weights & Biases Arguments ---
    parser.add_argument('--wandb_project', type=str, default="gpt_pretrain_hf", help="Weights & Biases project name.")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Weights & Biases entity (username or team).")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Weights & Biases run name (auto-generated if None).")
    parser.add_argument('--wandb_run_id', type=str, default=None, help="Weights & Biases specific run ID to resume.")
    parser.add_argument('--disable_wandb', action='store_true', help="Disable Weights & Biases logging.")


    args = parser.parse_args()

    if args.d_ff is None: args.d_ff = args.d_model * 4
    assert args.d_model % args.n_head == 0, f"d_model ({args.d_model}) must be divisible by n_head ({args.n_head})"
    if args.eval_interval > 0 and args.eval_corpus_file is None:
        print("Warning: --eval_interval > 0 but --eval_corpus_file not provided. Disabling step-based eval.")
        args.eval_interval = 0
    if args.eval_batch_size is None: args.eval_batch_size = args.batch_size
    if args.max_kv_len is not None and args.max_kv_len <= 0:
        raise SystemExit("--max_kv_len must be greater than 0 when provided.")
    if args.train_stride is not None and args.train_stride <= 0:
        raise SystemExit("--train_stride must be greater than 0 when provided.")
    if args.eval_stride is not None and args.eval_stride <= 0:
        raise SystemExit("--eval_stride must be greater than 0 when provided.")
    if args.use_cuda_graphs and args.use_amp:
        raise SystemExit("--use_cuda_graphs cannot be combined with --use_amp (GradScaler is unsupported in CUDA graphs).")

    # Setup basic logging (if not already configured by a larger script framework)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



    train(args)
