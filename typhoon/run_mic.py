from __future__ import annotations

try:
    from .mic_cli import main
except ImportError:  # Allows: python typhoon/run_mic.py
    from mic_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
