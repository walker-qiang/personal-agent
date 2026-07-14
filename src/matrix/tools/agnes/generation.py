"""Agnes image and video generation tools."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from typing import Any

from ..base import ToolDefinition

# Use the same API key as Agnes LLM
AGNES_BASE_URL = os.environ.get("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")

# ---- Prompt Quality Enhancers ----
# These suffixes are appended to every prompt to guarantee baseline quality.
# The LLM is responsible for the creative visual description;
# the code guarantees technical quality keywords.

IMAGE_QUALITY_SUFFIX = (
    ", photorealistic, highly detailed, 8k resolution, professional photography, "
    "perfect lighting, sharp focus, no text, no watermark, no distortion, no extra limbs, "
    "correct anatomy, natural proportions"
)

VIDEO_QUALITY_SUFFIX = (
    ", cinematic, smooth motion, professional lighting, 4k quality, "
    "stable camera, natural movement, no text, no watermark"
)


def _enhance_image_prompt(prompt: str) -> str:
    """Ensure image prompt has quality keywords. Only adds if not already present."""
    prompt = prompt.strip()
    if "photorealistic" in prompt.lower() and "highly detailed" in prompt.lower():
        return prompt
    return prompt + IMAGE_QUALITY_SUFFIX


def _enhance_video_prompt(prompt: str) -> str:
    """Ensure video prompt has quality keywords. Only adds if not already present."""
    prompt = prompt.strip()
    if "cinematic" in prompt.lower() and "smooth motion" in prompt.lower():
        return prompt
    return prompt + VIDEO_QUALITY_SUFFIX


# ---- Image Generation ----


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "hd",
    n: int = 1,
) -> dict[str, Any]:
    """Generate an image using Agnes-Image-2.1-Flash.

    Args:
        prompt: Image description (text prompt, English preferred)
        size: Image size, one of 1024x1024, 1792x1024, 1024x1792
        quality: standard or hd (default hd for best quality)
        n: Number of images to generate (1-4)
    """
    if not AGNES_API_KEY:
        return {"error": "AGNES_API_KEY not configured", "images": []}

    n = max(1, min(n, 4))
    valid_sizes = {"1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"}
    size = size if size in valid_sizes else "1024x1024"
    quality = quality if quality in ("standard", "hd") else "hd"

    # Code-level quality enhancement: guarantees baseline quality keywords
    prompt = _enhance_image_prompt(prompt)

    payload = json.dumps({
        "model": "agnes-image-2.1-flash",
        "prompt": prompt,
        "n": n,
        "size": size,
        "quality": quality,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{AGNES_BASE_URL}/images/generations",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AGNES_API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        images = []
        for item in data.get("data", []):
            url = item.get("url", "")
            b64 = item.get("b64_json", "")
            images.append({"url": url, "b64_json": b64})
        return {"prompt": prompt, "size": size, "count": len(images), "images": images}
    except Exception as err:
        return {"error": f"Image generation failed: {err}", "images": []}


image_tool = ToolDefinition(
    name="agnes.generate_image",
    description="使用 Agnes Image 2.1 Flash 生成高质量图片。LLM 负责描述画面内容，代码自动追加质量关键词。",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "画面描述（英文），只需描述画面内容：主体、场景、动作、光线、构图、氛围。不需要加 photorealistic/8k 等质量词（代码自动添加）。",
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"],
                "description": "图片尺寸，默认 1024x1024",
            },
            "quality": {
                "type": "string",
                "enum": ["standard", "hd"],
                "description": "图片质量，默认 hd（高清）",
            },
            "n": {
                "type": "integer",
                "description": "生成数量，默认 1，最大 4",
            },
        },
        "required": ["prompt"],
    },
    handler=generate_image,
)


# ---- Video Generation ----


def generate_video(
    prompt: str,
    width: int = 1152,
    height: int = 768,
    num_frames: int = 121,
    frame_rate: int = 24,
) -> dict[str, Any]:
    """Generate a video using Agnes-Video-V2.0 (async task-based API).

    Args:
        prompt: Video description (text prompt, English preferred)
        width: Video width in pixels (default 1152)
        height: Video height in pixels (default 768)
        num_frames: Number of frames, must be 8n+1, max 441 (default 121 ≈ 5s at 24fps)
        frame_rate: Frame rate 1-60 (default 24)
    """
    if not AGNES_API_KEY:
        return {"error": "AGNES_API_KEY not configured", "videos": []}

    width = max(256, min(width, 1920))
    height = max(256, min(height, 1080))
    num_frames = max(1, min(num_frames, 441))
    frame_rate = max(1, min(frame_rate, 60))

    # Code-level quality enhancement: guarantees baseline quality keywords
    prompt = _enhance_video_prompt(prompt)

    # Step 1: Submit video generation task
    payload = json.dumps({
        "model": "agnes-video-v2.0",
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{AGNES_BASE_URL}/videos",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AGNES_API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        return {"error": f"Video task submission failed: {err}", "videos": []}

    task_id = data.get("task_id", "")
    if not task_id:
        return {"error": f"No task_id in response: {data}", "videos": []}

    # Step 2: Poll for completion (max 5 minutes)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(5)
        try:
            req = urllib.request.Request(
                f"{AGNES_BASE_URL}/videos/{task_id}",
                headers={"Authorization": f"Bearer {AGNES_API_KEY}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as err:
            return {"error": f"Video polling failed: {err}", "videos": []}

        status = result.get("status", "")
        if status == "completed":
            video_url = (
                result.get("metadata", {}).get("url", "")
                or result.get("video_url", "")
                or result.get("remixed_from_video_id", "")
            )
            return {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "duration_sec": round(num_frames / frame_rate, 1),
                "count": 1,
                "videos": [{"url": video_url, "task_id": task_id}],
            }
        elif status == "failed":
            return {"error": f"Video generation failed: {result.get('error', 'unknown')}", "videos": []}

    return {"error": "Video generation timed out after 5 minutes", "videos": []}


video_tool = ToolDefinition(
    name="agnes.generate_video",
    description="使用 Agnes Video V2.0 生成高质量视频（异步任务模式，提交后轮询等待结果，通常 2-3 分钟完成）。LLM 负责描述视频内容，代码自动追加质量关键词。",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "视频描述（英文），只需描述画面内容：场景、动作、运镜、氛围。不需要加 cinematic/4k 等质量词（代码自动添加）。",
            },
            "width": {
                "type": "integer",
                "description": "视频宽度（像素），默认 1152",
            },
            "height": {
                "type": "integer",
                "description": "视频高度（像素），默认 768",
            },
            "num_frames": {
                "type": "integer",
                "description": "帧数，格式 8n+1，最大 441。默认 121（约 5 秒@24fps）",
            },
            "frame_rate": {
                "type": "integer",
                "description": "帧率 1-60，默认 24",
            },
        },
        "required": ["prompt"],
    },
    handler=generate_video,
)