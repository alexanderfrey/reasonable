#!/usr/bin/env python3
"""
HTTP wrapper that turns GRPO-style generations into reward scores by
asking a VLLM/OpenAI-compatible endpoint to rank each group of candidates.

Run:
    python reward_vllm_server.py \
        --host 0.0.0.0 --port 8000 \
        --model-id openai/gpt-oss-20b \
        --openai-api-base http://100.68.45.88/v1 \
        --group-size 4

Then point rlhf_train_grpo_dapo.py at http://localhost:8000/score with
--reward_model_url and set --reward_model_extra '{"group_size":4}' if you
need to override the default chunk size.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - required dependency
    raise SystemExit(
        "The 'openai' package is required for reward_vllm_server.py.\n"
        "Install it with `pip install openai` (>=1.0)."
    ) from exc


DEFAULT_TIMEOUT = 60.0
DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous judge. Rank candidate answers primarily by originality and usefulness "
    "(i.e., how novel, insightful, and practically helpful they are). When entries are similar, "
    "resolve ties by preferring the option that is more factually accurate, clear, and safe."
)
DEFAULT_USER_TEMPLATE = (
    "User prompt:\n{prompt}\n\n"
    "You will be given {k} candidate answers. Judge each one on originality (new ideas, non-generic "
    "reasoning) and usefulness (actionable, practically valuable guidance).\n"
    "{candidates}\n\n"
    "Return the indices of the candidates from best to worst as a space-separated "
    "list of numbers (e.g., '1 3 2'). Respond with numbers only."
)


def chunk_list(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def format_candidates(candidates: Sequence[str]) -> str:
    return "\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(candidates))


def parse_rank_response(text: str, group_size: int) -> List[int]:
    found = []
    for match in re.findall(r"\d+", text):
        idx = int(match) - 1  # convert to 0-based
        if 0 <= idx < group_size and idx not in found:
            found.append(idx)
    for idx in range(group_size):
        if idx not in found:
            found.append(idx)
    return found[:group_size]


def ranking_to_scores(order: Sequence[int]) -> List[float]:
    K = len(order)
    scores = [0.0] * K
    for rank_position, candidate_idx in enumerate(order):
        # Higher reward for better-ranked answers
        scores[candidate_idx] = float(K - rank_position) / max(1, K)
    return scores


def _truncate_text(value: str, *, max_chars: int = 160) -> str:
    """Return a single-line preview of potentially long text."""
    sanitized = value.replace("\n", "\\n")
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[: max_chars - 3] + "..."


def _preview_list(values: Sequence[str], *, label: str, max_items: int = 2, max_chars: int = 160) -> str:
    """Format a short preview for logging."""
    if not values:
        return f"{label}=[]"
    shown = []
    for idx, val in enumerate(values[:max_items]):
        shown.append(f"{idx}:{_truncate_text(str(val), max_chars=max_chars)}")
    if len(values) > max_items:
        shown.append(f"...(+{len(values) - max_items} more)")
    return f"{label}=" + "; ".join(shown)


class VLLMRanker:
    def __init__(
        self,
        *,
        model_id: str,
        api_base: str,
        api_key: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float,
        system_prompt: str,
        user_template: str,
        append_no_think: bool,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.append_no_think = append_no_think
        self.user_template = user_template
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=api_base,
            timeout=timeout,
        )

    def _build_messages(self, prompt: str, candidates: Sequence[str]) -> List[dict]:
        candidate_block = format_candidates(candidates)
        content = self.user_template.format(
            prompt=prompt.strip(),
            candidates=candidate_block,
            k=len(candidates),
        )
        if self.append_no_think and "Qwen3" in self.model_id:
            content += "\n/no_think"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    def rank_group(self, prompt: str, candidates: Sequence[str]) -> Tuple[List[int], List[float]]:
        if len(candidates) == 0:
            return [], []
        if len(candidates) == 1:
            return [0], [1.0]

        messages = self._build_messages(prompt, candidates)
        response_text = ""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            if isinstance(content, str):
                response_text = content.strip()
            else:
                response_text = ""
        except Exception as exc:  # pragma: no cover - network path
            logging.exception("Ranking request failed: %s", exc)
            order = list(range(len(candidates)))
            return order, ranking_to_scores(order)

        order = parse_rank_response(response_text, len(candidates))
        scores = ranking_to_scores(order)
        return order, scores

    def score_batch(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        *,
        group_size: int,
    ) -> Tuple[List[float], List[List[int]]]:
        if group_size <= 0:
            group_size = 1
        if not outputs:
            return [], []

        prompt_chunks = chunk_list(prompts, group_size)
        output_chunks = chunk_list(outputs, group_size)
        scores: List[float] = []
        orders: List[List[int]] = []

        for chunk_idx, candidate_group in enumerate(output_chunks):
            prompt_text = ""
            if chunk_idx < len(prompt_chunks):
                prompt_group = prompt_chunks[chunk_idx]
                prompt_text = prompt_group[0] if prompt_group else ""
            order, group_scores = self.rank_group(prompt_text, candidate_group)
            if not group_scores:
                group_scores = [0.0] * len(candidate_group)
            if len(group_scores) < len(candidate_group):
                group_scores.extend([0.0] * (len(candidate_group) - len(group_scores)))
            scores.extend(group_scores[: len(candidate_group)])
            orders.append(order if order else list(range(len(candidate_group))))
        return scores, orders


class RewardRequestHandler(BaseHTTPRequestHandler):
    ranker: VLLMRanker = None  # type: ignore
    default_group_size: int = 1

    def _read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/score", ""):
            self._write_json(404, {"error": "not found"})
            return
        payload = self._read_payload()
        prompts = payload.get("prompts", []) or []
        outputs = payload.get("outputs", []) or []
        extra = payload.get("extra", {}) or {}
        group_size = int(extra.get("group_size", self.default_group_size) or self.default_group_size)

        logging.info(
            "Incoming /score request from %s | prompts=%d outputs=%d group_size=%d",
            self.client_address[0] if self.client_address else "unknown",
            len(prompts),
            len(outputs),
            group_size,
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "Payload preview | %s | %s | extra=%s",
                _preview_list(prompts, label="prompts"),
                _preview_list(outputs, label="outputs"),
                json.dumps(extra, ensure_ascii=True)[:512],
            )

        start = time.time()
        try:
            scores, orders = self.ranker.score_batch(prompts, outputs, group_size=group_size)
            elapsed = (time.time() - start) * 1000.0
            logging.info(
                "Ranked %d outputs (group_size=%d) in %.2f ms",
                len(outputs),
                group_size,
                elapsed,
            )
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "Response preview | scores=%s | orders=%s",
                    _preview_list([f"{s:.4f}" for s in scores], label="scores", max_items=6, max_chars=32),
                    json.dumps(orders, ensure_ascii=True)[:512],
                )
            self._write_json(200, {"scores": scores, "orders": orders})
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logging.exception("Failed to score batch: %s", exc)
            zeros = [0.0] * len(outputs)
            self._write_json(500, {"scores": zeros, "error": str(exc)})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLLM reward-model HTTP wrapper.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--openai-api-base", type=str, required=True,
                        help="Base URL for the VLLM/OpenAI-compatible endpoint, e.g. http://100.68.45.88/v1")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the endpoint (defaults to VLLM_API_KEY env or 'EMPTY').")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", type=str, default=DEFAULT_USER_TEMPLATE)
    parser.add_argument("--append-no-think", action="store_true",
                        help="Append '/no_think' for Qwen-style judge models.")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Default group size when the caller does not provide one via payload.extra.group_size.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ranker = VLLMRanker(
        model_id=args.model_id,
        api_base=args.openai_api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        system_prompt=args.system_prompt,
        user_template=args.user_template,
        append_no_think=args.append_no_think,
    )
    RewardRequestHandler.ranker = ranker
    RewardRequestHandler.default_group_size = max(1, args.group_size)

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    logging.info("Reward server listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
