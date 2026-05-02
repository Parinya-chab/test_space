from __future__ import annotations

import uvicorn

try:
    from .asr_service import app
except ImportError:  # Allows: python typhoon/run_service.py
    from asr_service import app


def main() -> int:
    uvicorn.run(app, host="127.0.0.1", port=8001)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
