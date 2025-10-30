# preprocess_dirs_to_corpus_fast.py
import os, re, glob, random, argparse, unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterable, Optional
from tqdm import tqdm

# pip install blake3 transformers
from blake3 import blake3
from transformers import AutoTokenizer


# -------------------------
# Binary detection helper
# -------------------------
def is_binary_file(path: str, blocksize: int = 1024) -> bool:
    """
    Heuristically detect binary files by scanning the first 'blocksize' bytes.
    - Skip if NUL byte found or if >30% bytes are non-texty.
    - On error, treat as binary (skip) to stay safe.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(blocksize)
            if not chunk:
                return False  # empty file -> not binary
            if b"\x00" in chunk:
                return True
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
            nontext = sum(b not in text_chars for b in chunk)
            return (nontext / len(chunk)) > 0.30
    except Exception as e:
        print(f"[SKIP] Could not probe {path}: {e}")
        return True


# -------------------------
# Text helpers
# -------------------------
def normalize(text: str) -> str:
    # NFC normalize + strip CR + collapse whitespace
    text = unicodedata.normalize("NFC", text.replace("\r", ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iter_txt_files(dirs: List[str]) -> List[str]:
    """
    Recursively find *.txt under dirs, returning each underlying file once.
    De-duplicates using (st_dev, st_ino), robust to symlinks/hardlinks.
    """
    seen_inodes = set()
    out = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
            try:
                st = os.stat(p, follow_symlinks=True)
                key = (st.st_dev, st.st_ino)
                if key in seen_inodes:
                    continue
                seen_inodes.add(key)
                out.append(p)
            except Exception as e:
                print(f"[SKIP] Could not stat {p}: {e}")
                continue
    return out


def _open_text(path: str):
    """Open a text file with UTF-8, fallback to latin-1 (ignore undecodable)."""
    try:
        return open(path, "r", encoding="utf-8")
    except UnicodeDecodeError:
        return open(path, "r", encoding="latin-1", errors="ignore")


# -------------------------
# Read & chunk (silent; logging is in task)
# -------------------------
def read_file_chunks(
    path: str,
    min_chars: int,
    large_file_threshold_bytes: int,
    chunk_bytes: int
) -> List[str]:
    """
    Read a file and return a list of normalized document chunks.
    No logging here; binary check is performed again defensively (cheap).
    """
    # Defensive binary check (fast)
    if is_binary_file(path):
        return []

    try:
        size = os.path.getsize(path)
    except Exception:
        return []

    docs: List[str] = []
    if size <= large_file_threshold_bytes:
        try:
            with _open_text(path) as f:
                raw = f.read()
        except Exception:
            return []
        doc = normalize(raw)
        doc = re.sub(r"OceanofPDF\.com", "", doc, flags=re.IGNORECASE).strip()
        if len(doc) >= min_chars:
            docs.append(doc)
        return docs

    # Large file: chunked processing
    approx_chunk_chars = max(1, chunk_bytes)  # text-mode .read() takes "chars"
    try:
        with _open_text(path) as f:
            while True:
                block = f.read(approx_chunk_chars)
                if not block:
                    break
                tail = f.readline()  # finish current line
                chunk_text = block + tail
                doc = normalize(chunk_text)
                doc = re.sub(r"OceanofPDF\.com", "", doc, flags=re.IGNORECASE).strip()
                if len(doc) >= min_chars:
                    docs.append(doc)
    except Exception:
        return []
    return docs


# -------------------------
# Dedup & token length
# -------------------------
def hash_doc(doc: str) -> str:
    return blake3(doc.encode("utf-8")).hexdigest()


def batch_token_lengths(tokenizer, docs: List[str]) -> int:
    """
    Fast token counting using the Rust 'fast' tokenizer with return_length=True.
    """
    enc = tokenizer(docs, add_special_tokens=False, return_length=True,
                    padding=False, truncation=False)
    if hasattr(enc, "length") and enc.length is not None:
        return sum(int(l) for l in enc.length)
    return sum(len(ids) for ids in enc["input_ids"])


# -------------------------
# Main (true streaming)
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Streaming corpus prep from multiple directories of .txt files (binary-safe, deduped, low-memory)")
    ap.add_argument("dirs", nargs="+", help="Directories with .txt files")
    ap.add_argument("--train_out", default="train_corpus.txt")
    ap.add_argument("--eval_out", default="eval_corpus.txt")
    ap.add_argument("--min_chars", type=int, default=2000)
    ap.add_argument("--eval_fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8,
                    help="Thread workers for parallel I/O/normalization")
    ap.add_argument("--tokenizer_name", default="gpt2")
    ap.add_argument("--batch_tokenize_size", type=int, default=2048,
                    help="Token-counting batch size (docs per batch)")
    ap.add_argument("--streaming", action="store_true",
                    help="(Ignored; streaming is always on in this version)")
    ap.add_argument("--no_tokenizer", action="store_true",
                    help="Skip tokenizer and token counting")
    ap.add_argument("--large_file_threshold_mb", type=int, default=20,
                    help="Files larger than this are processed in chunks and printed")
    ap.add_argument("--chunk_size_mb", type=int, default=4,
                    help="Approximate chunk size (MB) when processing large files")
    args = ap.parse_args()

    # Derived sizes
    threshold_bytes = args.large_file_threshold_mb * 1024 * 1024
    chunk_bytes     = args.chunk_size_mb * 1024 * 1024

    # List files (unique)
    files = iter_txt_files(args.dirs)
    if not files:
        raise SystemExit("No .txt files found.")
    print(f"Found {len(files):,} unique files.")

    # Tokenizer (optional)
    tokenizer = None
    eos_id = None
    if not args.no_tokenizer:
        print(f"Loading tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=True, trust_remote_code=True
        )
        eos_id = tokenizer.eos_token_id

    # Counters & stats
    counters = {
        "skipped_binary": 0,
        "skipped_errors": 0,
        "large_files": 0,
        "processed_files": 0,
        "train_docs": 0,
        "eval_docs": 0,
        "train_chars": 0,
        "eval_chars": 0,
        "train_tokens": 0,
        "eval_tokens": 0,
    }

    # Dedup by content hash (exact)
    seen_hashes = set()

    # Prepare output files
    train_dir = os.path.dirname(args.train_out)
    eval_dir = os.path.dirname(args.eval_out)
    if train_dir:
        os.makedirs(train_dir, exist_ok=True)
    if eval_dir:
        os.makedirs(eval_dir, exist_ok=True)
    train_f = open(args.train_out, "w", encoding="utf-8")
    eval_f  = open(args.eval_out,  "w", encoding="utf-8")

    # Small buffers for batch token-counting (keeps memory tiny)
    train_tokbuf: List[str] = []
    eval_tokbuf:  List[str] = []

    def flush_token_buffers():
        nonlocal train_tokbuf, eval_tokbuf
        if tokenizer is None:
            train_tokbuf.clear()
            eval_tokbuf.clear()
            return
        if train_tokbuf:
            counters["train_tokens"] += batch_token_lengths(tokenizer, train_tokbuf)
            train_tokbuf.clear()
        if eval_tokbuf:
            counters["eval_tokens"]  += batch_token_lengths(tokenizer, eval_tokbuf)
            eval_tokbuf.clear()

    # Single-source logging + bounded parallelism via executor.map
    def task(p: str) -> List[str]:
        # Binary?
        if is_binary_file(p):
            counters["skipped_binary"] += 1
            print(f"[SKIP-BINARY] {p}")
            return []
        try:
            size = os.path.getsize(p)
        except Exception as e:
            print(f"[SKIP] Could not stat {p}: {e}")
            counters["skipped_errors"] += 1
            return []
        if size > threshold_bytes:
            counters["large_files"] += 1
            print(f"[LARGE] {p} — {size / (1024*1024):.1f} MB")
        try:
            pieces = read_file_chunks(p, args.min_chars, threshold_bytes, chunk_bytes)
            counters["processed_files"] += 1
            return pieces
        except Exception as e:
            print(f"[SKIP] Unexpected error on {p}: {e}")
            counters["skipped_errors"] += 1
            return []

    # Process files streaming, writing lines immediately
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pieces in tqdm(ex.map(task, files, chunksize=64),
                               total=len(files), desc="Files", unit="file"):
                if not pieces:
                    continue
                for doc in pieces:
                    h = hash_doc(doc)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    # Split by deterministic hash bucket
                    bucket = (int(h[:8], 16) / 2**32)  # 0..1
                    if bucket < args.eval_fraction:
                        eval_f.write(doc + "\n")
                        counters["eval_docs"] += 1
                        counters["eval_chars"] += len(doc)
                        if tokenizer is not None:
                            eval_tokbuf.append(doc)
                            if len(eval_tokbuf) >= args.batch_tokenize_size:
                                counters["eval_tokens"] += batch_token_lengths(tokenizer, eval_tokbuf)
                                eval_tokbuf.clear()
                    else:
                        train_f.write(doc + "\n")
                        counters["train_docs"] += 1
                        counters["train_chars"] += len(doc)
                        if tokenizer is not None:
                            train_tokbuf.append(doc)
                            if len(train_tokbuf) >= args.batch_tokenize_size:
                                counters["train_tokens"] += batch_token_lengths(tokenizer, train_tokbuf)
                                train_tokbuf.clear()
    finally:
        # Flush token buffers and close files
        flush_token_buffers()
        train_f.close()
        eval_f.close()

    # Account for +1 EOS per doc (if tokenizer has eos)
    if tokenizer is not None and eos_id is not None:
        counters["train_tokens"] += counters["train_docs"]
        counters["eval_tokens"]  += counters["eval_docs"]

    # Summary
    def avg_len(total_chars: int, n_docs: int) -> float:
        return (total_chars / max(1, n_docs)) if n_docs else 0.0

    print("\n=== Summary (Streaming) ===")
    print(f"Train docs : {counters['train_docs']:,} → {args.train_out}")
    print(f"Eval docs  : {counters['eval_docs']:,} → {args.eval_out}")
    print(f"Avg chars  : train={avg_len(counters['train_chars'], counters['train_docs']):.1f} | "
          f"eval={avg_len(counters['eval_chars'], counters['eval_docs']):.1f}")
    if tokenizer is not None:
        print("Token counts (incl. +1 EOS per doc if tokenizer provides EOS):")
        print(f"  TRAIN: {counters['train_tokens']:,}")
        print(f"  EVAL : {counters['eval_tokens']:,}")
    else:
        print("Token counts: skipped (--no_tokenizer)")

    print("\n=== File stats ===")
    print(f"Processed files   : {counters['processed_files']:,}")
    print(f"Large files       : {counters['large_files']:,}")
    print(f"Skipped (binary)  : {counters['skipped_binary']:,}")
    print(f"Skipped (errors)  : {counters['skipped_errors']:,}")

if __name__ == "__main__":
    main()
