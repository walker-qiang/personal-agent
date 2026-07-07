"""Project Matrix entry point."""

from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from .config import load_config


def main() -> None:
    """Start the Project Matrix Agent server."""
    load_dotenv()
    config = load_config()
    uvicorn.run(
        "matrix.server.app:create_app",
        host=config.host,
        port=config.port,
        factory=True,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()