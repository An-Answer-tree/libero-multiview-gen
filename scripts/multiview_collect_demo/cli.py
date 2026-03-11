"""CLI entrypoint for multiview replay dataset reconstruction."""

from __future__ import annotations

from typing import Optional, Sequence

import init_path  # noqa: F401

from .config import build_replay_config, parse_args
from .runner import run_replay


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parses CLI arguments and runs the multiview replay pipeline."""

    args = parse_args(argv)
    config = build_replay_config(args)
    run_replay(config)


if __name__ == "__main__":
    main()
