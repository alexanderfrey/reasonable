# preprocess_dirs_to_corpus_fast.py
import os, re, glob, random, argparse, unicodedata, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Iterable, Optional
from tqdm import tqdm

# pip install blake3 transformers
from blake3 import blake3
from transformers import AutoTokenizer

def normalize(text: str) -> str:
    # NFC normalize + strip CR + collapse whitespace
    text = unicodedata.normalize("NFC", text.replace("\r", ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def iter_txt_files(dirs: List[str]) -> List[str]:
    files = []
    for d in dirs:
        files.extend(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    return files

def read_file(path: str, min_chars: int) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            raw = f.read()
    doc = normalize(raw)
    if len(doc) >= min_chars:
        return doc
    return None

def hash_doc(doc: str) -> str:
    return blake3(doc.encode("utf-8")).hexdigest()

def dedup_docs_stream(docs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for d in docs:
        h = hash_doc(d)
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out

def batch_token_lengths(tokenizer, docs: List[str], batch_size: int = 4096) -> int:
    """
    Fast token counting using the Rust 'fast' tokenizer in batch with return_length=True.
    """
    total = 0
    for i in tqdm(range(0, len(docs), batch_size), desc="Tokenizing (batch)", unit="batch"):
        batch = docs[i:i+batch_size]
        enc = tokenizer(batch, add_special_tokens=False, return_length=True,
                        padding=False, truncation=False)
        # enc.length is available on fast tokenizers; otherwise derive from input_ids
        if hasattr(enc, "length") and enc.length is not None:
            total += sum(int(l) for l in enc.length)
        else:
            # fallback if return_length missing
            total += sum(len(ids) for ids in enc["input_ids"])
    return total

def write_lines(path: str, docs: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for d in tqdm(docs, desc=f"Writing {os.path.basename(path)}", unit="doc"):
            f.write(d + "\n")

def main():
    ap = argparse.ArgumentParser(description="Fast corpus prep from multiple directories of .txt files")
    ap.add_argument("dirs", nargs="+", help="Directories with .txt files")
    ap.add_argument("--train_out", default="train_corpus.txt")
    ap.add_argument("--eval_out", default="eval_corpus.txt")
    ap.add_argument("--min_chars", type=int, default=2000)
    ap.add_argument("--eval_fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Thread workers for parallel I/O/normalization")
    ap.add_argument("--tokenizer_name", default="gpt2")
    ap.add_argument("--batch_tokenize_size", type=int, default=8192,
                    help="Batch size for fast token length computation")
    ap.add_argument("--streaming", action="store_true",
                    help="Low-memory mode: dedup & split while streaming; slightly slower token counting")
    args = ap.parse_args()

    # List files
    files = iter_txt_files(args.dirs)
    if not files:
        raise SystemExit("No .txt files found.")
    print(f"Found {len(files):,} files.")

    # Parallel read + normalize + length filter
    print(f"Reading & normalizing with {args.workers} workers…")
    docs: List[str] = []
    if not args.streaming:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(read_file, p, args.min_chars): p for p in files}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Files", unit="file"):
                doc = fut.result()
                if doc:
                    docs.append(doc)

        print(f"Kept {len(docs):,} documents (≥{args.min_chars} chars)")

        # Dedup exact
        before = len(docs)
        docs = dedup_docs_stream(docs)
        print(f"Dedup: {before:,} → {len(docs):,}")

        # Shuffle & split
        random.seed(args.seed)
        random.shuffle(docs)
        n_eval = max(1, int(len(docs) * args.eval_fraction))
        eval_docs = docs[:n_eval]
        train_docs = docs[n_eval:]
    else:
        # Streaming mode: dedup + split without holding all docs in memory
        # Use hash-based split for determinism: put doc in eval if int(hash[:8],16) / 2^32 < eval_fraction
        seen = set()
        train_docs, eval_docs = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(read_file, p, args.min_chars): p for p in files}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Files", unit="file"):
                doc = fut.result()
                if not doc:
                    continue
                h = hash_doc(doc)
                if h in seen:
                    continue
                seen.add(h)
                bucket = (int(h[:8], 16) / 2**32)  # 0..1
                if bucket < args.eval_fraction:
                    eval_docs.append(doc)
                else:
                    train_docs.append(doc)
        print(f"Streaming split: train={len(train_docs):,} | eval={len(eval_docs):,} (fraction≈{args.eval_fraction:.3f})")

    # Load tokenizer (fast)
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    # Token counts via fast batch lengths
    print("Counting tokens (TRAIN)…")
    train_tokens = batch_token_lengths(tokenizer, train_docs, args.batch_tokenize_size)
    if eos_id is not None:
        train_tokens += len(train_docs)  # +1 EOS per doc (your pretokenizer behavior)

    print("Counting tokens (EVAL)…")
    eval_tokens = batch_token_lengths(tokenizer, eval_docs, args.batch_tokenize_size)
    if eos_id is not None:
        eval_tokens += len(eval_docs)

    # Write outputs
    write_lines(args.train_out, train_docs)
    write_lines(args.eval_out,  eval_docs)

    # Summary
    def avg_len(ds: List[str]) -> float:
        return (sum(len(x) for x in ds) / max(1, len(ds))) if ds else 0.0

    print("\n=== Summary ===")
    print(f"Train docs : {len(train_docs):,} → {args.train_out}")
    print(f"Eval docs  : {len(eval_docs):,} → {args.eval_out}")
    print(f"Avg chars  : train={avg_len(train_docs):.1f} | eval={avg_len(eval_docs):.1f}")
    print("Token counts (incl. +1 EOS per doc if tokenizer provides EOS):")
    print(f"  TRAIN: {train_tokens:,}")
    print(f"  EVAL : {eval_tokens:,}")
    print("Done.")
    
if __name__ == "__main__":
    main()