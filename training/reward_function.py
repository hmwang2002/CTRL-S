"""Custom verl reward function used for CTRL-S reinforcement learning.

The reward combines:

1. Output-format gate.
2. DINOv2 similarity between the rendered prediction and reference image.
3. Long-CLIP similarity between the rendered prediction and instruction.
4. SVG code-efficiency reward relative to the reference SVG.

DINOv2 and Long-CLIP scores are provided by ``training/reward_server.py``.
This module intentionally has no direct import dependency on verl: verl loads
``compute_score`` dynamically through ``custom_reward_function.path``.
"""

from __future__ import annotations

import os
import random
import re
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cairosvg
import requests
from func_timeout import FunctionTimedOut, func_timeout


SUPPORTED_DATA_SOURCES = {
    "internsvg/sophia_i2s_rl",
    "internsvg/sophia_t2s_rl",
    "internsvg/sophia_refine_rl",
}

FAILURE_RESULT = {
    "score": 0.0,
    "dino_reward": 0.0,
    "clip_reward": 0.0,
    "code_efficiency_reward": 0.0,
}

W_CODE_EFFICIENCY = 0.5
W_DINO = 1.0
W_CLIP = 0.5


def _failure_result() -> dict[str, float]:
    """Return an independent zero-reward result for one sample."""

    return FAILURE_RESULT.copy()


def _reward_server_urls() -> list[str]:
    """Build reward-server URLs from environment variables."""

    host = os.getenv("REWARD_SERVER_HOST", "127.0.0.1")
    ports = os.getenv(
        "REWARD_SERVER_PORTS",
        "8000,8001,8002,8003,8004,8005,8006,8007",
    )
    urls = [
        f"http://{host}:{port.strip()}"
        for port in ports.split(",")
        if port.strip()
    ]
    if not urls:
        raise ValueError("REWARD_SERVER_PORTS must contain at least one port")
    return urls


def _reward_output_dir() -> Path:
    """Return the directory used for rendered rollout artifacts."""

    default_dir = Path(tempfile.gettempdir()) / "ctrl_s_reward"
    return Path(os.getenv("REWARD_OUTPUT_DIR", default_dir)).expanduser()


class RewardClient:
    """Lightweight client for the DINOv2 and Long-CLIP reward services."""

    def __init__(self) -> None:
        self.server_urls = _reward_server_urls()
        self.timeout = float(os.getenv("REWARD_REQUEST_TIMEOUT", "30"))

    def _post_score(self, endpoint: str, payload: dict[str, Any]) -> float:
        server_url = random.choice(self.server_urls)
        url = f"{server_url}/{endpoint}"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                return float(response.json().get("score", 0.0))

            print(
                f"Reward request failed for {url}: "
                f"HTTP {response.status_code} - {response.text}"
            )
            return 0.0
        except Exception as exc:
            print(f"Reward request failed for {url}: {exc}")
            return 0.0

    def compute_clip(self, image_path: Path, caption: str) -> float:
        return self._post_score(
            "compute_clip",
            {
                "image_path": str(image_path),
                "caption": caption,
            },
        )

    def compute_dino(self, pred_path: Path, gt_path: str) -> float:
        return self._post_score(
            "compute_dino",
            {
                "pred_path": str(pred_path),
                "gt_path": gt_path,
            },
        )

    @staticmethod
    def code_efficiency_reward(pred_len: int, gt_len: int) -> float:
        eps = 1e-8
        diff = max(pred_len - gt_len / 2, 0)
        reward = 1 - (diff / (gt_len + eps)) ** 2
        return max(0.0, reward)


def is_valid_svg(path: Path) -> bool:
    try:
        ET.parse(path)
        return True
    except (ET.ParseError, OSError):
        return False


def raster_svg(
    svg_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> bool:
    """Rasterize a generated SVG with a bounded execution time."""

    try:
        if not is_valid_svg(svg_path):
            return False

        func_timeout(
            10,
            cairosvg.svg2png,
            kwargs={
                "url": str(svg_path),
                "write_to": str(output_path),
                "background_color": "white",
                "output_width": width,
                "output_height": height,
            },
        )
        return True
    except FunctionTimedOut:
        print(f"Raster timeout: SVG at {svg_path} took too long to render")
        return False
    except Exception as exc:
        print(f"Raster error: {exc}")
        return False


def extract_refine_caption(text: str) -> str:
    """Extract the instruction between ``Instruction:`` and ``Draft Code:``."""

    match = re.search(
        r"Instruction:\s*(.*?)\s*Draft Code:",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def is_output_format_valid(solution_str: str) -> bool:
    """Check the reasoning and fenced-SVG format expected during training."""

    output = solution_str.strip()

    think_blocks = re.findall(r"<think>.*?</think>", output, flags=re.DOTALL)
    if len(think_blocks) != 1:
        return False

    svg_codeblocks = re.findall(
        r"```svg\s*(.*?)```",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(svg_codeblocks) != 1:
        return False

    svg_block_content = svg_codeblocks[0].strip()
    svgs_in_block = re.findall(
        r"<svg\b[^>]*>.*?</svg>",
        svg_block_content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(svgs_in_block) != 1:
        return False

    return "```" not in svg_block_content


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute the weighted CTRL-S reward for one verl rollout."""

    if data_source not in SUPPORTED_DATA_SOURCES:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}"
        )

    extra_info = extra_info or {}
    question = extra_info.get("question", "")
    gt_images = extra_info.get("gt_images", [])
    if not gt_images:
        return _failure_result()

    if not is_output_format_valid(solution_str):
        return _failure_result()

    pred_svg_match = re.search(
        r"<svg[^>]*>.*?</svg>",
        solution_str,
        flags=re.DOTALL,
    )
    gt_svg_match = re.search(
        r"<svg[^>]*>.*?</svg>",
        ground_truth,
        flags=re.DOTALL,
    )
    if not pred_svg_match or not gt_svg_match:
        return _failure_result()

    pred_svg_content = pred_svg_match.group(0)
    gt_svg_content = gt_svg_match.group(0)

    if data_source in {
        "internsvg/sophia_i2s_rl",
        "internsvg/sophia_t2s_rl",
    }:
        question_parts = question.split("Instruction: ", 1)
        if len(question_parts) != 2:
            return _failure_result()
        caption = question_parts[1].strip()
    else:
        caption = extract_refine_caption(question)

    output_dir = _reward_output_dir()
    svg_dir = output_dir / "svg"
    image_dir = output_dir / "images"
    svg_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    sample_index = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        str(extra_info.get("index", "unknown")),
    )
    artifact_id = f"{sample_index}_{int(time.time())}_{uuid.uuid4().hex}"
    pred_svg_path = svg_dir / f"gen_{artifact_id}.svg"
    pred_image_path = image_dir / f"gen_{artifact_id}.png"
    pred_svg_path.write_text(pred_svg_content, encoding="utf-8")

    if not raster_svg(pred_svg_path, pred_image_path, 448, 448):
        return _failure_result()

    reward_client = RewardClient()
    code_efficiency_reward = reward_client.code_efficiency_reward(
        len(pred_svg_content),
        len(gt_svg_content),
    )
    dino_reward = reward_client.compute_dino(
        pred_image_path,
        str(gt_images[0]),
    )
    clip_reward = reward_client.compute_clip(pred_image_path, caption)

    final_score = (
        W_CODE_EFFICIENCY * code_efficiency_reward
        + W_DINO * dino_reward
        + W_CLIP * clip_reward
    )

    return {
        "score": float(final_score),
        "dino_reward": dino_reward,
        "clip_reward": clip_reward,
        "code_efficiency_reward": code_efficiency_reward,
    }
