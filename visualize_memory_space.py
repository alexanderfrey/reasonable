#!/usr/bin/env python3
"""
Visualize episodic memory vectors from one or more checkpoints.

Example:
  python visualize_memory_space.py \
    --checkpoint memory_book_output_kv/memory_gpt_epoch_1.pt \
    --checkpoint memory_book_output_kv/memory_gpt_epoch_2.pt \
    --method pca --dims 2 --out memory_space.png
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


def _load_snapshot(path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    snapshot = checkpoint.get("episodic_memory")
    if snapshot is None:
        raise KeyError(f"{path} does not contain 'episodic_memory'.")
    return snapshot


def _collect_vectors(snapshot: Dict[str, object]) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    episodes = snapshot.get("episodes", [])
    if not episodes:
        return np.zeros((0, 0), dtype=np.float32), []

    vectors = []
    meta = []
    global_step = snapshot.get("global_step", 0)
    for ep in episodes:
        vec = ep.get_content_vector().detach().cpu().float().numpy()
        vectors.append(vec)
        meta.append({
            "salience": float(ep.salience),
            "retrieval_count": float(ep.retrieval_count),
            "timestamp": float(ep.timestamp),
            "age": float(max(0, global_step - ep.timestamp)),
            "episode_id": float(getattr(ep, "episode_id", -1)),
        })
    return np.vstack(vectors), meta


def _sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def _pca_reduce(x: np.ndarray, dims: int) -> np.ndarray:
    if x.size == 0:
        return x
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ vt[:dims].T


def _tsne_reduce(x: np.ndarray, dims: int, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for t-SNE.") from exc
    return TSNE(
        n_components=dims,
        perplexity=min(30, max(5, x.shape[0] // 20)),
        init="random",
        random_state=seed,
        learning_rate="auto"
    ).fit_transform(x)


def _umap_reduce(x: np.ndarray, dims: int, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required for UMAP.") from exc
    return umap.UMAP(
        n_components=dims,
        n_neighbors=min(15, max(5, x.shape[0] // 50)),
        random_state=seed
    ).fit_transform(x)


def _reduce(x: np.ndarray, method: str, dims: int, seed: int) -> np.ndarray:
    if method == "pca":
        return _pca_reduce(x, dims)
    if method == "tsne":
        return _tsne_reduce(x, dims, seed)
    if method == "umap":
        return _umap_reduce(x, dims, seed)
    raise ValueError(f"Unknown method: {method}")


def _write_csv(
    out_csv: str,
    coords: np.ndarray,
    meta: List[Dict[str, float]],
    checkpoint_ids: List[int],
    checkpoint_labels: List[str],
) -> None:
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["x", "y", "z", "checkpoint_id", "checkpoint_label",
                  "salience", "retrieval_count", "timestamp", "age", "episode_id"]
        writer.writerow(header)
        for i in range(coords.shape[0]):
            row = [
                coords[i, 0],
                coords[i, 1] if coords.shape[1] > 1 else 0.0,
                coords[i, 2] if coords.shape[1] > 2 else 0.0,
                checkpoint_ids[i],
                checkpoint_labels[i],
                meta[i]["salience"],
                meta[i]["retrieval_count"],
                meta[i]["timestamp"],
                meta[i]["age"],
                meta[i]["episode_id"],
            ]
            writer.writerow(row)


def _plot(
    out_path: str,
    coords: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    dims: int
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    if dims == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=color_values, cmap="viridis", s=8, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=color_values, cmap="viridis", s=8, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _animate(
    out_path: str,
    coords: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    time_values: np.ndarray,
    dims: int,
    frames: int,
    fps: int,
    window: float,
) -> None:
    if dims != 2:
        raise RuntimeError("Animation is only supported for 2D plots.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write MP4 animations.")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for animation.") from exc

    if coords.size == 0:
        return

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    t_min, t_max = time_values.min(), time_values.max()
    frames = max(2, frames)
    thresholds = np.linspace(t_min, t_max, frames)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    sc = ax.scatter([], [], c=[], cmap="viridis", s=8, alpha=0.7,
                    vmin=color_values.min(), vmax=color_values.max())
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(color_label)
    title = ax.set_title("")

    def update(frame_idx: int):
        t = thresholds[frame_idx]
        if window > 0:
            mask = (time_values > (t - window)) & (time_values <= t)
        else:
            mask = time_values <= t
        pts = coords[mask]
        sc.set_offsets(pts)
        sc.set_array(color_values[mask])
        title.set_text(f"time <= {t:.0f}  (n={pts.shape[0]})")
        return sc, title

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / max(1, fps), blit=False)
    anim.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize episodic memory space from checkpoints.")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Path to a checkpoint (.pt). Use multiple times to compare.")
    parser.add_argument("--method", action="append", choices=["pca", "tsne", "umap"],
                        help="Dimensionality reduction method (repeatable).")
    parser.add_argument("--dims", type=int, default=2, choices=[2, 3],
                        help="Output dimensions.")
    parser.add_argument("--max_points", type=int, default=2000,
                        help="Max points per checkpoint (0 = no limit).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling/reduction.")
    parser.add_argument("--color_by", choices=["checkpoint", "salience", "retrieval", "age"], default="checkpoint",
                        help="Color points by this attribute.")
    parser.add_argument("--out", default="memory_space.png",
                        help="Output image path (png). Used as base when multiple methods.")
    parser.add_argument("--out_csv", default="memory_space.csv",
                        help="Output CSV path (always written). Used as base when multiple methods.")
    parser.add_argument("--animate", action="store_true",
                        help="Write an MP4 animation showing memory evolution over time.")
    parser.add_argument("--animate_out", default="memory_space.mp4",
                        help="Output MP4 path. Used as base when multiple methods.")
    parser.add_argument("--animate_frames", type=int, default=60,
                        help="Number of animation frames.")
    parser.add_argument("--animate_fps", type=int, default=20,
                        help="Frames per second for MP4.")
    parser.add_argument("--animate_window", type=float, default=0.0,
                        help="Sliding time window (0 = cumulative).")
    args = parser.parse_args()

    all_vectors = []
    all_meta: List[Dict[str, float]] = []
    checkpoint_ids = []
    checkpoint_labels = []

    for idx, path in enumerate(args.checkpoint):
        snapshot = _load_snapshot(path)
        vectors, meta = _collect_vectors(snapshot)
        if vectors.size == 0:
            continue
        indices = _sample_indices(vectors.shape[0], args.max_points, args.seed + idx)
        vectors = vectors[indices]
        meta = [meta[i] for i in indices]
        all_vectors.append(vectors)
        all_meta.extend(meta)
        checkpoint_ids.extend([idx] * vectors.shape[0])
        checkpoint_labels.extend([os.path.basename(path)] * vectors.shape[0])

    if not all_vectors:
        print("No episodic memories found in provided checkpoints.", file=sys.stderr)
        return 1

    x = np.vstack(all_vectors)
    methods = args.method or ["pca"]

    if args.color_by == "checkpoint":
        color_values = np.array(checkpoint_ids, dtype=np.float32)
        color_label = "checkpoint"
    elif args.color_by == "salience":
        color_values = np.array([m["salience"] for m in all_meta], dtype=np.float32)
        color_label = "salience"
    elif args.color_by == "retrieval":
        color_values = np.array([m["retrieval_count"] for m in all_meta], dtype=np.float32)
        color_label = "retrieval_count"
    else:
        color_values = np.array([m["age"] for m in all_meta], dtype=np.float32)
        color_label = "age"

    for method in methods:
        coords = _reduce(x, method, args.dims, args.seed)
        out_csv = args.out_csv
        out_path = args.out
        out_mp4 = args.animate_out
        if len(methods) > 1:
            stem_csv, ext_csv = os.path.splitext(args.out_csv)
            stem_out, ext_out = os.path.splitext(args.out)
            stem_mp4, ext_mp4 = os.path.splitext(args.animate_out)
            out_csv = f"{stem_csv}_{method}{ext_csv}"
            out_path = f"{stem_out}_{method}{ext_out}"
            out_mp4 = f"{stem_mp4}_{method}{ext_mp4}"

        _write_csv(out_csv, coords, all_meta, checkpoint_ids, checkpoint_labels)

        try:
            _plot(out_path, coords, color_values, color_label, args.dims)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            print(f"Wrote CSV to {out_csv} instead.", file=sys.stderr)
            continue

        if args.animate:
            try:
                time_values = np.array([m["timestamp"] for m in all_meta], dtype=np.float32)
                _animate(
                    out_mp4,
                    coords,
                    color_values,
                    color_label,
                    time_values,
                    args.dims,
                    args.animate_frames,
                    args.animate_fps,
                    args.animate_window,
                )
                print(f"Wrote animation to {out_mp4}")
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)

        print(f"Wrote plot to {out_path}")
        print(f"Wrote CSV to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
