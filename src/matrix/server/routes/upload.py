"""File upload route for multi-modal input."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".pdf", ".txt", ".md", ".csv", ".json", ".yaml", ".yml",
}


def _extract_text(file_path: Path, mime_type: str) -> str:
    """Extract text content from uploaded file."""
    if mime_type and mime_type.startswith("text/"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    ext = file_path.suffix.lower()
    if ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            return f"[PDF 文件: {file_path.name}]"

    return ""


@router.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a file for multi-modal input.

    Supports images (PNG/JPEG/WebP), PDFs, and text files up to 10MB.
    Returns file metadata including extracted text content.
    """
    if not file.filename:
        raise HTTPException(400, "filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_SIZE // 1024 // 1024}MB limit")

    # Derive upload dir from config
    config = request.app.state.config
    upload_dir = Path(config.root_path).parent / "var" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    file_path = upload_dir / f"{file_id}{ext}"
    file_path.write_bytes(content)

    mime_type = file.content_type or ""
    extracted_text = _extract_text(file_path, mime_type)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "mime_type": mime_type,
        "size": len(content),
        "ext": ext,
        "is_image": ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"),
        "text": extracted_text[:5000] if extracted_text else "",  # Truncate for response
    }