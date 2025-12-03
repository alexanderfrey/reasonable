#!/usr/bin/env python3
"""
Stage 4: Deduplication
- Exact dedup on whole-doc hash.
- Chunk-level dedup (fixed-size chunks) to drop repeated chapters/sections across books.
- Optional near-duplicate chunk detection via MinHash + LSH (Jaccard similarity).
- Optional aggregate corpus output.
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import List, Optional, Set, Tuple

from book_pipeline.common import auto_read_text, ensure_parent_dir, is_binary_file, iter_txt_files

try:  # Optional dependency
    from datasketch import MinHash, MinHashLSH
except Exception:  # pragma: no cover - handled at runtime
    MinHash = None
    MinHashLSH = None


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return []
    if overlap < 0 or overlap >= chunk_size:
        overlap = 0
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_minhash(text: str, num_perm: int, shingle_size: int) -> "MinHash":
    mh = MinHash(num_perm=num_perm)
    if len(text) <= shingle_size:
        mh.update(text.encode("utf-8", errors="ignore"))
        return mh
    for i in range(0, len(text) - shingle_size + 1):
        shingle = text[i : i + shingle_size]
        mh.update(shingle.encode("utf-8", errors="ignore"))
    return mh


def process_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    seen_docs: Set[str],
    seen_chunks: Set[str],
    doc_lock: Lock,
    chunk_lock: Lock,
    lsh: Optional["MinHashLSH"],
    lsh_lock: Optional[Lock],
    num_perm: int,
    shingle_size: int,
    chunk_size: int,
    chunk_overlap: int,
    dup_chunk_fraction: float,
    min_chars: int,
    overwrite: bool,
) -> Tuple[str, str]:
    rel = path.relative_to(input_root)
    out_path = output_root / rel
    if not overwrite and out_path.exists():
        return ("skipped_existing", "")
    if is_binary_file(path):
        return ("binary", "")
    try:
        text = auto_read_text(path)
    except Exception as exc:
        return (f"error:{exc}", "")
    if len(text) < min_chars:
        return ("too_short", "")

    doc_hash = hashlib.blake2s(text.encode("utf-8")).hexdigest()
    with doc_lock:
        if doc_hash in seen_docs:
            return ("dedup_doc", "")

    chunks = chunk_text(text, chunk_size, chunk_overlap) if chunk_size > 0 else []
    chunk_hashes = [hashlib.blake2s(c.encode("utf-8")).hexdigest() for c in chunks] if chunks else []
    chunk_minhashes = [chunk_minhash(c, num_perm, shingle_size) for c in chunks] if chunks and lsh else []

    if chunk_hashes:
        with chunk_lock:
            dup_chunks_exact = sum(1 for h in chunk_hashes if h in seen_chunks)
        dup_chunks_lsh = 0
        if lsh and lsh_lock and chunk_minhashes:
            for mh in chunk_minhashes:
                with lsh_lock:
                    hits = lsh.query(mh)
                if hits:
                    dup_chunks_lsh += 1
        dup_total = dup_chunks_exact + dup_chunks_lsh
        if chunk_hashes and (dup_total / len(chunk_hashes)) >= dup_chunk_fraction:
            reason = f"{dup_total}/{len(chunk_hashes)} dup chunks (exact={dup_chunks_exact}, lsh={dup_chunks_lsh})"
            return ("dedup_chunks", reason)

    ensure_parent_dir(out_path)
    out_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

    with doc_lock:
        seen_docs.add(doc_hash)
    if chunk_hashes:
        with chunk_lock:
            for h in chunk_hashes:
                seen_chunks.add(h)
    if lsh and lsh_lock and chunk_minhashes:
        with lsh_lock:
            for idx, mh in enumerate(chunk_minhashes):
                key = f"{rel}:{idx}"
                lsh.insert(key, mh)
    return ("written", "")


def main():
    ap = argparse.ArgumentParser(description="Stage 4: deduplicate cleaned book texts.")
    ap.add_argument("input_dir", help="Directory with stage3 outputs.")
    ap.add_argument("--output_dir", default="data/books_stage4_dedup", help="Where to write deduped files.")
    ap.add_argument("--aggregate_out", default="data/books_corpus.txt", help="Concatenate kept docs here (optional, empty to skip).")
    ap.add_argument("--chunk_size", type=int, default=4000, help="Chunk size in characters for chunk-level dedup.")
    ap.add_argument("--chunk_overlap", type=int, default=400, help="Overlap between chunks in characters.")
    ap.add_argument("--dup_chunk_fraction", type=float, default=0.4, help="Drop doc if >= this fraction of chunks already seen or near-duplicated.")
    ap.add_argument("--min_chars", type=int, default=1000, help="Drop documents below this before deduping.")
    ap.add_argument("--enable_lsh", action="store_true", help="Enable MinHash + LSH near-duplicate detection on chunks.")
    ap.add_argument("--lsh_threshold", type=float, default=0.8, help="Jaccard similarity threshold for LSH queries (higher = stricter).")
    ap.add_argument("--num_perm", type=int, default=128, help="Number of permutations for MinHash; must match LSH config.")
    ap.add_argument("--shingle_size", type=int, default=5, help="Character shingle size for MinHash.")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool workers.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite files even if output exists.")
    args = ap.parse_args()

    input_root = Path(args.input_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()
    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_path = Path(args.aggregate_out).expanduser() if args.aggregate_out else None
    agg_f = None
    if aggregate_path:
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        agg_f = aggregate_path.open("w", encoding="utf-8")

    manager_seen_docs: Set[str] = set()
    manager_seen_chunks: Set[str] = set()
    doc_lock = Lock()
    chunk_lock = Lock()
    lsh_lock = Lock() if MinHashLSH and args.enable_lsh else None
    lsh = None
    if args.enable_lsh:
        if MinHashLSH is None or MinHash is None:
            raise SystemExit("MinHash/LSH requested but 'datasketch' is not installed. Install with: pip install datasketch")
        lsh = MinHashLSH(threshold=args.lsh_threshold, num_perm=args.num_perm)

    stats = {
        "written": 0,
        "skipped_existing": 0,
        "binary": 0,
        "too_short": 0,
        "dedup_doc": 0,
        "dedup_chunks": 0,
        "errors": 0,
    }

    def task(p: Path) -> Tuple[str, str]:
        return process_file(
            p,
            input_root,
            output_root,
            manager_seen_docs,
            manager_seen_chunks,
            doc_lock,
            chunk_lock,
            lsh,
            lsh_lock,
            args.num_perm,
            args.shingle_size,
            args.chunk_size,
            args.chunk_overlap,
            args.dup_chunk_fraction,
            args.min_chars,
            args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, detail in ex.map(task, iter_txt_files([input_root]), chunksize=16):
            if status.startswith("error:"):
                stats["errors"] += 1
                print(f"[SKIP] {status}")
            elif status == "written":
                stats["written"] += 1
            elif status == "dedup_chunks":
                stats["dedup_chunks"] += 1
                if detail:
                    print(f"[DEDUP-CHUNKS] {detail}")
            else:
                stats[status] = stats.get(status, 0) + 1

    if agg_f:
        first = True
        for path in iter_txt_files([output_root]):
            try:
                text = auto_read_text(path).rstrip()
            except Exception:
                continue
            if not text:
                continue
            if not first:
                agg_f.write("\n\n")
            agg_f.write(text)
            first = False
        agg_f.close()

    print("=== Stage 4 summary ===")
    print(f"Written           : {stats['written']}")
    print(f"Skipped existing  : {stats['skipped_existing']}")
    print(f"Skipped binary    : {stats['binary']}")
    print(f"Too short         : {stats['too_short']}")
    print(f"Dedup (doc)       : {stats['dedup_doc']}")
    print(f"Dedup (chunks)    : {stats['dedup_chunks']}")
    print(f"Errors            : {stats['errors']}")
    if aggregate_path:
        print(f"Aggregate corpus  : {aggregate_path}")
    print(f"Output dir        : {output_root}")


if __name__ == "__main__":
    main()
