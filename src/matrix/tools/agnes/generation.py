"""Agnes image and video generation tools."""

from __future__ import annotations

import base64
import json
import os
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
    style: str = "photorealistic",
) -> dict[str, Any]:
    """Generate an image using Agnes-Image-2.0-Flash.

    Args:
        prompt: Image description (text prompt, English preferred)
        size: Image size, one of 1024x1024, 1792x1024, 1024x1792
        quality: standard or hd (default hd for best quality)
        n: Number of images to generate (1-4)
        style: Visual style — photorealistic, artistic, anime, oil-painting, sketch
    """
    if not AGNES_API_KEY:
        return {"error": "AGNES_API_KEY not configured", "images": []}

    n = max(1, min(n, 4))
    valid_sizes = {"1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"}
    size = size if size in valid_sizes else "1024x1024"
    quality = quality if quality in ("standard", "hd") else "hd"
    valid_styles = {"photorealistic", "artistic", "anime", "oil-painting", "sketch", "3d-render", "watercolor"}
    style = style if style in valid_styles else "photorealistic"

    # Code-level quality enhancement: guarantees baseline quality keywords
    prompt = _enhance_image_prompt(prompt)

    payload = json.dumps({
        "model": "agnes-image-2.0-flash",
        "prompt": prompt,
        "n": n,
        "size": size,
        "quality": quality,
        "style": style,
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
    description="使用 Agnes Image 2.0 Flash 生成高质量图片。LLM 负责描述画面内容（场景、主体、动作、氛围），代码自动追加质量关键词。支持多种视觉风格。",
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
            "style": {
                "type": "string",
                "enum": ["photorealistic", "artistic", "anime", "oil-painting", "sketch", "3d-render", "watercolor"],
                "description": "视觉风格，默认 photorealistic（逼真摄影）",
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
    duration: int = 5,
    resolution: str = "1080p",
    style: str = "cinematic",
) -> dict[str, Any]:
    """Generate a video using Agnes-Video-V2.0.

    Args:
        prompt: Video description (text prompt, English preferred)
        duration: Video duration in seconds (1-30)
        resolution: Video resolution, 480p, 720p, or 1080p (default 1080p)
        style: Visual style — cinematic, animation, documentary, timelapse
    """
    if not AGNES_API_KEY:
        return {"error": "AGNES_API_KEY not configured", "videos": []}

    duration = max(1, min(duration, 30))
    valid_resolutions = {"480p", "720p", "1080p"}
    resolution = resolution if resolution in valid_resolutions else "1080p"
    valid_styles = {"cinematic", "animation", "documentary", "timelapse", "slow-motion", "aerial"}
    style = style if style in valid_styles else "cinematic"

    # Code-level quality enhancement: guarantees baseline quality keywords
    prompt = _enhance_video_prompt(prompt)

    payload = json.dumps({
        "model": "agnes-video-v2.0",
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "style": style,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{AGNES_BASE_URL}/video/generations",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AGNES_API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        videos = []
        for item in data.get("data", []):
            url = item.get("url", "")
            videos.append({
                "url": url,
                "duration": duration,
                "resolution": resolution,
            })
        return {"prompt": prompt, "duration": duration, "resolution": resolution, "count": len(videos), "videos": videos}
    except Exception as err:
        return {"error": f"Video generation failed: {err}", "videos": []}


video_tool = ToolDefinition(
    name="agnes.generate_video",
    description="使用 Agnes Video V2.0 生成高质量视频。LLM 负责描述视频内容，代码自动追加质量关键词。支持多种风格和分辨率。",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "视频描述（英文），只需描述画面内容：场景、动作、运镜、氛围。不需要加 cinematic/4k 等质量词（代码自动添加）。",
            },
            "duration": {
                "type": "integer",
                "description": "视频时长（秒），默认 5，最大 30",
            },
            "resolution": {
                "type": "string",
                "enum": ["480p", "720p", "1080p"],
                "description": "视频分辨率，默认 1080p",
            },
            "style": {
                "type": "string",
                "enum": ["cinematic", "animation", "documentary", "timelapse", "slow-motion", "aerial"],
                "description": "视觉风格，默认 cinematic（电影感）",
            },
        },
        "required": ["prompt"],
    },
    handler=generate_video,
)