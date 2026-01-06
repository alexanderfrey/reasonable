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
import glob
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _load_snapshot(path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    snapshot = checkpoint.get("episodic_memory")
    if snapshot is None:
        raise KeyError(f"{path} does not contain 'episodic_memory'.")
    return snapshot


def _collect_vectors(
    snapshot: Dict[str, object],
    episode_filter: Optional[set[int]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    episodes = snapshot.get("episodes", [])
    if not episodes:
        return np.zeros((0, 0), dtype=np.float32), []

    vectors = []
    meta = []
    global_step = snapshot.get("global_step", 0)
    for ep in episodes:
        episode_id = int(getattr(ep, "episode_id", -1))
        if episode_filter is not None and episode_id not in episode_filter:
            continue
        vec = ep.get_content_vector().detach().cpu().float().numpy()
        vectors.append(vec)
        meta.append({
            "salience": float(ep.salience),
            "retrieval_count": float(ep.retrieval_count),
            "timestamp": float(ep.timestamp),
            "age": float(max(0, global_step - ep.timestamp)),
            "episode_id": float(episode_id),
        })
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.vstack(vectors), meta


def _sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def _sample_indices_with_forced(
    n: int,
    max_points: int,
    seed: int,
    forced: Optional[np.ndarray],
) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    if forced is None or forced.size == 0:
        return _sample_indices(n, max_points, seed)
    forced = np.unique(forced)
    if forced.size >= max_points:
        return forced[:max_points]
    remaining = np.setdiff1d(np.arange(n), forced, assume_unique=False)
    if remaining.size == 0:
        return forced
    rng = np.random.default_rng(seed)
    extra = rng.choice(remaining, size=max_points - forced.size, replace=False)
    return np.concatenate([forced, extra])


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
    dims: int,
    trajectory_indices: Optional[List[List[int]]] = None,
    trajectory_alpha: float = 0.35,
    trajectory_arrows: bool = False,
    focus_groups: Optional[List[List[int]]] = None,
    focus_group_labels: Optional[List[List[str]]] = None,
    focus_colors: Optional[List[str]] = None,
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

        if trajectory_indices:
            for idx_list in trajectory_indices:
                pts = coords[idx_list]
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color="black",
                    alpha=trajectory_alpha,
                    linewidth=0.7,
                )
                if trajectory_arrows and pts.shape[0] >= 2:
                    start = pts[-2]
                    end = pts[-1]
                    ax.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            alpha=trajectory_alpha,
                            linewidth=0.7,
                        ),
                    )

        if focus_groups:
            palette = focus_colors or [
                "crimson",
                "darkorange",
                "dodgerblue",
                "forestgreen",
                "purple",
            ]
            for group_idx, indices in enumerate(focus_groups):
                if not indices:
                    continue
                color = palette[group_idx % len(palette)]
                focus_pts = coords[indices]
                ax.plot(
                    focus_pts[:, 0],
                    focus_pts[:, 1],
                    color=color,
                    linewidth=1.8,
                    alpha=0.9,
                )
                ax.scatter(
                    focus_pts[:, 0],
                    focus_pts[:, 1],
                    color=color,
                    s=28,
                    zorder=3,
                )
                if focus_group_labels:
                    labels = focus_group_labels[group_idx]
                    for idx, label in zip(indices, labels):
                        ax.annotate(
                            label,
                            xy=(coords[idx, 0], coords[idx, 1]),
                            xytext=(4, 3),
                            textcoords="offset points",
                            fontsize=7,
                            color=color,
                        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _episode_index_map(
    episode_ids: np.ndarray,
    checkpoint_ids: List[int],
) -> Dict[int, List[Tuple[int, int]]]:
    trajectories: Dict[int, List[Tuple[int, int]]] = {}
    for idx, (eid, chk) in enumerate(zip(episode_ids, checkpoint_ids)):
        if eid < 0:
            continue
        trajectories.setdefault(int(eid), []).append((chk, idx))
    return trajectories


def _trajectory_indices(
    episode_ids: np.ndarray,
    checkpoint_ids: List[int],
    max_trajectories: int,
    seed: int,
) -> List[List[int]]:
    trajectories = _episode_index_map(episode_ids, checkpoint_ids)

    ordered = []
    for entries in trajectories.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda item: item[0])
        ordered.append([idx for _, idx in entries])

    if max_trajectories > 0 and len(ordered) > max_trajectories:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(ordered), size=max_trajectories, replace=False)
        ordered = [ordered[i] for i in picks]

    return ordered


def _checkpoint_sort_key(path: str) -> Tuple[int, int, str]:
    name = os.path.basename(path)
    epoch_match = re.search(r"epoch_(\d+)", name)
    step_match = re.search(r"step_(\d+)", name)
    epoch = int(epoch_match.group(1)) if epoch_match else 0
    step = int(step_match.group(1)) if step_match else 0
    return (epoch, step, name)


def _collect_checkpoints(
    checkpoints: List[str],
    checkpoint_dir: Optional[str],
    checkpoint_glob: str,
    checkpoint_sort: str,
) -> List[str]:
    paths = list(checkpoints) if checkpoints else []
    if checkpoint_dir:
        pattern = os.path.join(checkpoint_dir, checkpoint_glob)
        matches = glob.glob(pattern, recursive=True)
        paths.extend(sorted(matches))

    unique_paths = sorted({os.path.abspath(p) for p in paths})
    if not unique_paths:
        return []

    if checkpoint_sort == "mtime":
        unique_paths.sort(key=lambda p: os.path.getmtime(p))
    elif checkpoint_sort == "name":
        unique_paths.sort(key=lambda p: os.path.basename(p))
    elif checkpoint_sort == "step":
        unique_paths.sort(key=_checkpoint_sort_key)
    else:
        any_step = any(re.search(r"step_(\d+)", os.path.basename(p)) for p in unique_paths)
        if any_step:
            unique_paths.sort(key=_checkpoint_sort_key)
        else:
            unique_paths.sort(key=lambda p: os.path.basename(p))

    return unique_paths


def _top_retrieved_episode_ids(snapshot: Dict[str, object], k: int) -> List[int]:
    episodes = snapshot.get("episodes", [])
    scored = []
    for ep in episodes:
        episode_id = int(getattr(ep, "episode_id", -1))
        if episode_id < 0:
            continue
        scored.append((float(ep.retrieval_count), episode_id))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [episode_id for _, episode_id in scored[:k]]


def _longest_history_episode_ids(
    checkpoint_paths: List[str],
    k: int,
) -> List[int]:
    counts: Dict[int, int] = {}
    last_retrieval: Dict[int, float] = {}
    total_paths = len(checkpoint_paths)
    iter_paths = checkpoint_paths
    if tqdm is not None:
        iter_paths = tqdm(checkpoint_paths, desc="Scanning history", unit="ckpt")
    else:
        print(f"Scanning {total_paths} checkpoints for history...", file=sys.stderr)
    for idx, path in enumerate(iter_paths):
        snapshot = _load_snapshot(path)
        episodes = snapshot.get("episodes", [])
        for ep in episodes:
            episode_id = int(getattr(ep, "episode_id", -1))
            if episode_id < 0:
                continue
            counts[episode_id] = counts.get(episode_id, 0) + 1
            if idx == total_paths - 1:
                last_retrieval[episode_id] = float(ep.retrieval_count)
    scored = []
    for episode_id, count in counts.items():
        scored.append((count, last_retrieval.get(episode_id, 0.0), episode_id))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [episode_id for _, _, episode_id in scored[:k]]


def _init_focus_stats(focus_ids: List[int]) -> Dict[int, Dict[str, object]]:
    stats: Dict[int, Dict[str, object]] = {}
    for eid in focus_ids:
        stats[eid] = {
            "appearances": 0,
            "corrections": 0,
            "prev_mod": None,
            "last_mod": 0,
            "last_retrieval": 0.0,
            "first_idx": None,
            "last_idx": None,
        }
    return stats


def _update_focus_stats(
    snapshot: Dict[str, object],
    focus_id_set: set[int],
    stats: Dict[int, Dict[str, object]],
    checkpoint_idx: int,
) -> None:
    episodes = snapshot.get("episodes", [])
    for ep in episodes:
        episode_id = int(getattr(ep, "episode_id", -1))
        if episode_id not in focus_id_set:
            continue
        entry = stats.get(episode_id)
        if entry is None:
            continue
        mod_count = int(getattr(ep, "modification_count", 0))
        prev_mod = entry["prev_mod"]
        if prev_mod is not None and mod_count > prev_mod:
            entry["corrections"] = int(entry["corrections"]) + (mod_count - prev_mod)
        entry["prev_mod"] = mod_count
        entry["last_mod"] = mod_count
        entry["last_retrieval"] = float(getattr(ep, "retrieval_count", 0.0))
        entry["appearances"] = int(entry["appearances"]) + 1
        if entry["first_idx"] is None:
            entry["first_idx"] = checkpoint_idx
        entry["last_idx"] = checkpoint_idx


def _animate(
    out_path: str,
    coords: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    time_values: np.ndarray,
    time_label: str,
    dims: int,
    frames: int,
    fps: int,
    window: float,
    snapshot_mode: bool,
    focus_groups: Optional[List[List[int]]] = None,
    focus_colors: Optional[List[str]] = None,
    focus_only: bool = False,
    focus_step: bool = False,
    reverse: bool = False,
    focus_zoom: bool = False,
    focus_zoom_pad: float = 0.1,
    focus_amplify: float = 1.0,
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

    focus_paths = []
    if focus_groups:
        for indices in focus_groups:
            idx_arr = np.array(indices, dtype=np.int64)
            if idx_arr.size == 0:
                focus_paths.append(idx_arr)
                continue
            order = np.argsort(time_values[idx_arr])
            focus_paths.append(idx_arr[order])

    def apply_amplify(points: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        if focus_amplify == 1.0:
            return points
        return anchor + (points - anchor) * focus_amplify

    if (focus_only or focus_zoom) and focus_paths:
        focus_points = []
        for idx_arr in focus_paths:
            if idx_arr.size == 0:
                continue
            anchor = coords[idx_arr[0]]
            pts = apply_amplify(coords[idx_arr], anchor)
            focus_points.append(pts)
        if focus_points:
            focus_cloud = np.vstack(focus_points)
            x_min, x_max = focus_cloud[:, 0].min(), focus_cloud[:, 0].max()
            y_min, y_max = focus_cloud[:, 1].min(), focus_cloud[:, 1].max()
            pad_x = focus_zoom_pad * (x_max - x_min) if x_max > x_min else 1.0
            pad_y = focus_zoom_pad * (y_max - y_min) if y_max > y_min else 1.0
        else:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
            pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    else:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    if snapshot_mode:
        unique_times = np.unique(time_values)
        if unique_times.size == 0:
            return
        if frames > 0 and frames < unique_times.size:
            picks = np.linspace(0, unique_times.size - 1, frames)
            thresholds = unique_times[np.round(picks).astype(int)]
        else:
            thresholds = unique_times
        frames = thresholds.size
    else:
        t_min, t_max = time_values.min(), time_values.max()
        frames = max(2, frames)
        thresholds = np.linspace(t_min, t_max, frames)
    if reverse:
        thresholds = thresholds[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    sc = None
    if not focus_only:
        sc = ax.scatter([], [], c=[], cmap="viridis", s=8, alpha=0.7,
                        vmin=color_values.min(), vmax=color_values.max())
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(color_label)
    title = ax.set_title("")

    focus_lines = []
    focus_points = []
    focus_anchors = []
    if focus_groups:
        palette = focus_colors or [
            "crimson",
            "darkorange",
            "dodgerblue",
            "forestgreen",
            "purple",
        ]
        for group_idx, idx_arr in enumerate(focus_paths):
            if idx_arr.size == 0:
                focus_anchors.append(None)
            else:
                focus_anchors.append(coords[idx_arr[0]])
            color = palette[group_idx % len(palette)]
            line, = ax.plot([], [], color=color, linewidth=1.8, alpha=0.9, zorder=3)
            point = ax.scatter([], [], color=color, s=36, zorder=4)
            focus_lines.append(line)
            focus_points.append(point)

    def update(frame_idx: int):
        t = thresholds[frame_idx]
        if snapshot_mode:
            mask = np.isclose(time_values, t)
            title_text = f"{time_label} = {t:.0f}"
        else:
            if window > 0:
                mask = (time_values > (t - window)) & (time_values <= t)
            else:
                mask = time_values <= t
            title_text = f"{time_label} <= {t:.0f}"
        pts = coords[mask]
        if sc is not None:
            sc.set_offsets(pts)
            sc.set_array(color_values[mask])
        title.set_text(f"{title_text}  (n={pts.shape[0]})")
        if focus_groups:
            for group_idx, idx_arr in enumerate(focus_paths):
                if idx_arr.size == 0:
                    continue
                time_arr = time_values[idx_arr]
                if reverse:
                    line_mask = time_arr >= t
                else:
                    line_mask = time_arr <= t
                visible = idx_arr[line_mask]
                if focus_step:
                    if visible.size >= 2:
                        if reverse:
                            segment = visible[:2]
                        else:
                            segment = visible[-2:]
                        path_pts = coords[segment]
                    else:
                        path_pts = np.empty((0, 2))
                else:
                    path_pts = coords[visible]
                anchor = focus_anchors[group_idx]
                if anchor is not None and path_pts.size > 0:
                    path_pts = apply_amplify(path_pts, anchor)
                if path_pts.size == 0:
                    focus_lines[group_idx].set_data([], [])
                    focus_points[group_idx].set_offsets(np.empty((0, 2)))
                else:
                    focus_lines[group_idx].set_data(path_pts[:, 0], path_pts[:, 1])
                    if reverse:
                        focus_points[group_idx].set_offsets(path_pts[:1])
                    else:
                        focus_points[group_idx].set_offsets(path_pts[-1:])
        if sc is None:
            return title
        return sc, title

    def save_with_manual_writer():
        from matplotlib.animation import FFMpegWriter

        total = frames
        step = max(1, total // 20)
        print("Rendering animation...", file=sys.stderr)
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, out_path, dpi=200):
            for i in range(total):
                update(i)
                writer.grab_frame()
                if (i + 1) % step == 0 or (i + 1) == total:
                    print(f"Rendered {i + 1}/{total} frames", file=sys.stderr)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / max(1, fps), blit=False)
    if tqdm is None:
        save_with_manual_writer()
        plt.close(fig)
        return

    pbar = tqdm(total=frames, desc="Rendering animation", unit="frame")

    def progress_callback(frame_idx: int, total_frames: int):
        if total_frames:
            pbar.total = total_frames
        pbar.update(1)

    try:
        anim.save(out_path, writer="ffmpeg", fps=fps, progress_callback=progress_callback)
    except TypeError:
        pbar.close()
        save_with_manual_writer()
        plt.close(fig)
        return
    finally:
        pbar.close()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize episodic memory space from checkpoints.")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Path to a checkpoint (.pt). Use multiple times to compare.")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory of checkpoints to load.")
    parser.add_argument("--checkpoint_glob", default="*.pt",
                        help="Glob pattern for checkpoint_dir (default: *.pt).")
    parser.add_argument("--checkpoint_sort", default="auto",
                        choices=["auto", "name", "mtime", "step"],
                        help="Sort order for checkpoint_dir (default: auto).")
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
    parser.add_argument("--trajectories", action="store_true",
                        help="Draw per-episode trajectories across checkpoints (requires >=2 checkpoints).")
    parser.add_argument("--trajectory_max", type=int, default=200,
                        help="Max trajectories to draw (0 = no limit).")
    parser.add_argument("--trajectory_alpha", type=float, default=0.35,
                        help="Alpha for trajectory lines (default 0.35).")
    parser.add_argument("--trajectory_arrows", action="store_true",
                        help="Draw arrows on the final segment of each trajectory.")
    parser.add_argument("--focus_top_retrieved", action="store_true",
                        help="Highlight the most retrieved episode from the last checkpoint.")
    parser.add_argument("--focus_top_retrieved_k", type=int, default=1,
                        help="How many top retrieved episodes to highlight (default: 1).")
    parser.add_argument("--focus_longest_history", action="store_true",
                        help="Highlight episodes with the longest history across checkpoints.")
    parser.add_argument("--focus_longest_history_k", type=int, default=1,
                        help="How many longest-history episodes to highlight (default: 1).")
    parser.add_argument("--focus_episode_id", type=int, default=None,
                        help="Highlight a specific episode_id across checkpoints.")
    parser.add_argument("--focus_labels", action="store_true",
                        help="Annotate focus points with checkpoint labels.")
    parser.add_argument("--focus_only_points", action="store_true",
                        help="Only load focus episode IDs (faster for focus-only animations).")
    parser.add_argument("--animate", action="store_true",
                        help="Write an MP4 animation showing memory evolution over time.")
    parser.add_argument("--animate_by", choices=["timestamp", "checkpoint"], default="timestamp",
                        help="Animate by episode timestamp or checkpoint index.")
    parser.add_argument("--animate_snapshot", action="store_true",
                        help="Show only one snapshot per frame (useful with animate_by=checkpoint).")
    parser.add_argument("--animate_focus_only", action="store_true",
                        help="Only draw focus trajectories/points in animations.")
    parser.add_argument("--animate_focus_step", action="store_true",
                        help="Only draw the most recent focus segment per frame.")
    parser.add_argument("--animate_reverse", action="store_true",
                        help="Play the animation backwards (latest to earliest).")
    parser.add_argument("--animate_focus_zoom", action="store_true",
                        help="Zoom animation axes to focus trajectories.")
    parser.add_argument("--focus_zoom_pad", type=float, default=0.1,
                        help="Padding fraction for focus zoom (default: 0.1).")
    parser.add_argument("--focus_amplify", type=float, default=1.0,
                        help="Amplify focus movement around its first point (default: 1.0).")
    parser.add_argument("--animate_out", default="memory_space.mp4",
                        help="Output MP4 path. Used as base when multiple methods.")
    parser.add_argument("--animate_frames", type=int, default=60,
                        help="Number of animation frames.")
    parser.add_argument("--animate_fps", type=int, default=20,
                        help="Frames per second for MP4.")
    parser.add_argument("--animate_window", type=float, default=0.0,
                        help="Sliding time window (0 = cumulative).")
    args = parser.parse_args()

    checkpoint_paths = _collect_checkpoints(
        args.checkpoint,
        args.checkpoint_dir,
        args.checkpoint_glob,
        args.checkpoint_sort,
    )
    if not checkpoint_paths:
        print("No checkpoints provided. Use --checkpoint or --checkpoint_dir.", file=sys.stderr)
        return 1

    focus_ids: Optional[List[int]] = None
    focus_episode_mode = args.focus_episode_id is not None
    focus_top_mode = args.focus_top_retrieved
    focus_long_mode = args.focus_longest_history
    focus_requested = (
        args.animate_focus_only
        or args.animate_focus_step
        or args.focus_only_points
        or args.focus_labels
    )
    if not (focus_episode_mode or focus_top_mode or focus_long_mode) and focus_requested:
        focus_long_mode = True
        print("No focus mode specified; defaulting to longest_history.", file=sys.stderr)

    focus_modes = sum([focus_episode_mode, focus_top_mode, focus_long_mode])
    if focus_modes > 1:
        print("Choose only one focus mode: episode_id, top_retrieved, or longest_history.", file=sys.stderr)
        return 1
    if focus_episode_mode:
        focus_ids = [args.focus_episode_id]
    elif focus_long_mode:
        top_k = max(1, args.focus_longest_history_k)
        focus_ids = _longest_history_episode_ids(checkpoint_paths, top_k)
        if not focus_ids:
            print("No focus episodes found across checkpoints.", file=sys.stderr)
    elif focus_top_mode:
        last_snapshot = _load_snapshot(checkpoint_paths[-1])
        top_k = max(1, args.focus_top_retrieved_k)
        focus_ids = _top_retrieved_episode_ids(last_snapshot, top_k)
        if not focus_ids:
            print("No focus episodes found in last checkpoint.", file=sys.stderr)

    if focus_ids:
        print(f"Focus episodes: {focus_ids}", file=sys.stderr)

    episode_filter = None
    if args.focus_only_points and focus_ids:
        episode_filter = set(focus_ids)

    all_vectors = []
    all_meta: List[Dict[str, float]] = []
    checkpoint_ids = []
    checkpoint_labels = []

    iter_paths = checkpoint_paths
    focus_stats = _init_focus_stats(focus_ids) if focus_ids else None
    focus_id_set = set(focus_ids) if focus_ids else None
    log_every = None
    if tqdm is not None:
        iter_paths = tqdm(checkpoint_paths, desc="Loading checkpoints", unit="ckpt")
    else:
        total_paths = len(checkpoint_paths)
        log_every = max(1, total_paths // 10)
        print(f"Loading {total_paths} checkpoints...", file=sys.stderr)
    for idx, path in enumerate(iter_paths):
        snapshot = _load_snapshot(path)
        if focus_id_set and focus_stats is not None:
            _update_focus_stats(snapshot, focus_id_set, focus_stats, idx)
        vectors, meta = _collect_vectors(snapshot, episode_filter=episode_filter)
        if vectors.size == 0:
            continue
        forced_indices = None
        if focus_ids is not None and episode_filter is None and args.max_points > 0:
            focus_id_set = set(focus_ids)
            forced_indices = np.array(
                [i for i, m in enumerate(meta) if int(m["episode_id"]) in focus_id_set],
                dtype=np.int64,
            )
        indices = _sample_indices_with_forced(
            vectors.shape[0],
            args.max_points,
            args.seed + idx,
            forced_indices,
        )
        vectors = vectors[indices]
        meta = [meta[i] for i in indices]
        all_vectors.append(vectors)
        all_meta.extend(meta)
        checkpoint_ids.extend([idx] * vectors.shape[0])
        checkpoint_labels.extend([os.path.basename(path)] * vectors.shape[0])
        if tqdm is None and log_every and (idx + 1) % log_every == 0:
            print(f"Loaded {idx + 1}/{total_paths} checkpoints", file=sys.stderr)

    if focus_stats:
        print("Focus episode history:", file=sys.stderr)
        for eid in focus_ids or []:
            entry = focus_stats.get(eid)
            if not entry or entry["appearances"] == 0:
                print(f"  episode_id={eid} appearances=0 (not found)", file=sys.stderr)
                continue
            first_idx = entry["first_idx"]
            last_idx = entry["last_idx"]
            first_label = os.path.basename(checkpoint_paths[first_idx]) if first_idx is not None else "?"
            last_label = os.path.basename(checkpoint_paths[last_idx]) if last_idx is not None else "?"
            print(
                f"  episode_id={eid} appearances={entry['appearances']} "
                f"corrections={entry['corrections']} last_mod={entry['last_mod']} "
                f"last_retrieval={entry['last_retrieval']:.0f} range={first_label}..{last_label}",
                file=sys.stderr,
            )

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

    focus_groups = None
    focus_group_labels = None
    if args.focus_episode_id is not None or args.focus_top_retrieved or args.focus_longest_history:
        episode_ids = np.array([int(m["episode_id"]) for m in all_meta], dtype=np.int64)
        episode_map = _episode_index_map(episode_ids, checkpoint_ids)
        if not focus_ids:
            print("No focus episode found.", file=sys.stderr)
        else:
            focus_groups = []
            focus_group_labels = []
            for focus_id in focus_ids:
                entries = episode_map.get(int(focus_id), [])
                entries.sort(key=lambda item: item[0])
                indices = [idx for _, idx in entries]
                if not indices:
                    continue
                focus_groups.append(indices)
                if args.focus_labels:
                    focus_group_labels.append([checkpoint_labels[idx] for idx in indices])
            if not focus_groups:
                print("Focus episode not found in checkpoints.", file=sys.stderr)

    trajectory_indices = None
    if args.trajectories:
        if len(checkpoint_paths) < 2:
            print("Trajectories require at least two checkpoints.", file=sys.stderr)
        elif args.dims != 2:
            print("Trajectories are only supported for 2D plots.", file=sys.stderr)
        else:
            episode_ids = np.array([int(m["episode_id"]) for m in all_meta], dtype=np.int64)
            trajectory_indices = _trajectory_indices(
                episode_ids,
                checkpoint_ids,
                args.trajectory_max,
                args.seed,
            )
            if not trajectory_indices:
                print("No trajectories found (need matching episode_id across checkpoints).", file=sys.stderr)

    for method in methods:
        if method in ("tsne", "umap"):
            print(f"Reducing {x.shape[0]} points with {method}...", file=sys.stderr)
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
            _plot(
                out_path,
                coords,
                color_values,
                color_label,
                args.dims,
                trajectory_indices=trajectory_indices,
                trajectory_alpha=args.trajectory_alpha,
                trajectory_arrows=args.trajectory_arrows,
                focus_groups=focus_groups,
                focus_group_labels=focus_group_labels,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            print(f"Wrote CSV to {out_csv} instead.", file=sys.stderr)
            continue

        if args.animate:
            try:
                if args.animate_by == "checkpoint":
                    time_values = np.array(checkpoint_ids, dtype=np.float32)
                    time_label = "checkpoint"
                else:
                    time_values = np.array([m["timestamp"] for m in all_meta], dtype=np.float32)
                    time_label = "timestamp"
                _animate(
                    out_mp4,
                    coords,
                    color_values,
                    color_label,
                    time_values,
                    time_label,
                    args.dims,
                    args.animate_frames,
                    args.animate_fps,
                    args.animate_window,
                    args.animate_snapshot,
                    focus_groups=focus_groups,
                    focus_only=args.animate_focus_only,
                    focus_step=args.animate_focus_step,
                    reverse=args.animate_reverse,
                    focus_zoom=args.animate_focus_zoom,
                    focus_zoom_pad=args.focus_zoom_pad,
                    focus_amplify=args.focus_amplify,
                )
                print(f"Wrote animation to {out_mp4}")
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)

        print(f"Wrote plot to {out_path}")
        print(f"Wrote CSV to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
