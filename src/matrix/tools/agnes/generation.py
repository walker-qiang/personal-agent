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


# ---- Image Generation ----


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
) -> dict[str, Any]:
    """Generate an image using Agnes-Image-2.0-Flash.

    Args:
        prompt: Image description (text prompt)
        size: Image size, one of 1024x1024, 1792x1024, 1024x1792
        quality: standard or hd
        n: Number of images to generate (1-4)
    """
    if not AGNES_API_KEY:
        return {"error": "AGNES_API_KEY not configured", "images": []}

    n = max(1, min(n, 4))
    valid_sizes = {"1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"}
    size = size if size in valid_sizes else "1024x1024"
    quality = quality if quality in ("standard", "hd") else "standard"

    payload = json.dumps({
        "model": "agnes-image-2.0-flash",
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
    description="使用 Agnes Image 2.0 Flash 生成图片。支持文生图，可指定尺寸和质量。用于创建插图、海报、概念图等。",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "图片描述（提示词），英文效果更好",
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"],
                "description": "图片尺寸，默认 1024x1024",
            },
            "quality": {
                "type": "string",
                "enum": ["standard", "hd"],
                "description": "图片质量，默认 standard",
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
    resolution: str = "720p",
) -> dict[str, Any]:
    """Generate a video using Agnes-Video-V2.0.

    Args:
        prompt: Video description (text prompt)
        duration: Video duration in seconds (1-30)
        resolution: Video resolution, 480p, 720p, or 1080p
    """
    if not AGNES_API_KEY:
        return {"error": "AGNES_API_KEY not configured", "videos": []}

    duration = max(1, min(duration, 30))
    valid_resolutions = {"480p", "720p", "1080p"}
    resolution = resolution if resolution in valid_resolutions else "720p"

    payload = json.dumps({
        "model": "agnes-video-v2.0",
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
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
    description="使用 Agnes Video V2.0 生成视频。支持文生视频，可指定时长和分辨率。用于创建短视频、动画等。",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "视频描述（提示词），英文效果更好",
            },
            "duration": {
                "type": "integer",
                "description": "视频时长（秒），默认 5，最大 30",
            },
            "resolution": {
                "type": "string",
                "enum": ["480p", "720p", "1080p"],
                "description": "视频分辨率，默认 720p",
            },
        },
        "required": ["prompt"],
    },
    handler=generate_video,
)