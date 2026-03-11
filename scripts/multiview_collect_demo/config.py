"""Configuration helpers for multiview replay dataset reconstruction."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import init_path  # noqa: F401
import numpy as np
from libero.libero import get_libero_path
from replay_dataset_utils import DEFAULT_BENCHMARKS, DEFAULT_CAMERA_NAMES


DEFAULT_LEGACY_ASSET_MARKERS = (
    "chiliocosm/assets/",
    "libero/libero/assets/",
    "libero/assets/",
)
DEFAULT_CAMERA_HEIGHT = 128
DEFAULT_CAMERA_WIDTH = 128
DEFAULT_NO_PROPRIO = False
DEFAULT_STATE_ERROR_THRESHOLD = 0.01
DEFAULT_VALUE_CHECK_ATOL = 1e-6
DEFAULT_TRAJECTORY_CAMERA_COUNT = 10
DEFAULT_TRAJECTORY_INTERP_SAMPLES = 300
DEFAULT_TRAJECTORY_OFFSET_FILE = str(
    Path(__file__).resolve().parent.parent / "camera_offsets_example.txt"
)
DEFAULT_OPERATION_CAMERA_BASE_NAME = "agentview"
DEFAULT_OPERATION_CAMERA_NAMES = {
    "top": "operation_topview",
    "left": "operation_leftview",
    "right": "operation_rightview",
    "back": "operation_backview",
}


@dataclass(frozen=True)
class OperationCameraConfig:
    """Configuration for the generated fixed operation cameras.

    Attributes:
        base_camera_name: Fallback reference camera used to infer the scene center.
        camera_names: Mapping from logical camera role to generated camera name.
    """

    base_camera_name: str = DEFAULT_OPERATION_CAMERA_BASE_NAME
    camera_names: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_OPERATION_CAMERA_NAMES)
    )


@dataclass(frozen=True)
class TrajectoryCameraConfig:
    """Configuration for the generated trajectory cameras."""

    base_camera_name: str
    offset_file: str
    d_phi: np.ndarray
    d_theta: np.ndarray
    d_r: np.ndarray
    num_cameras: int
    interp_samples: int
    camera_names: list[str]


@dataclass(frozen=True)
class ReplayConfig:
    """Resolved runtime configuration for the replay pipeline."""

    source_root: str
    output_root: str
    benchmarks: list[str]
    tasks: Optional[list[str]]
    dry_run: bool
    overwrite: bool
    camera_height: int
    camera_width: int
    camera_names: list[str]
    operation_camera_names: list[str] = field(default_factory=list)
    trajectory_camera_names: list[str] = field(default_factory=list)
    operation_camera_config: Optional[OperationCameraConfig] = None
    trajectory_camera_config: Optional[TrajectoryCameraConfig] = None
    no_proprio: bool = DEFAULT_NO_PROPRIO
    divergence_threshold: float = DEFAULT_STATE_ERROR_THRESHOLD
    value_check_atol: float = DEFAULT_VALUE_CHECK_ATOL
    legacy_asset_markers: tuple[str, ...] = DEFAULT_LEGACY_ASSET_MARKERS

    @property
    def has_generated_cameras(self) -> bool:
        """Returns whether replay adds any generated cameras to the dataset."""

        return bool(self.operation_camera_config or self.trajectory_camera_config)


def dedupe_keep_order(items: Sequence[str]) -> list[str]:
    """Returns items with duplicates removed while keeping the original order."""

    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def trajectory_camera_names(base_camera_name: str, num_cameras: int) -> list[str]:
    """Returns deterministic names for generated trajectory cameras."""

    return [f"{base_camera_name}_traj_{idx:02d}" for idx in range(num_cameras)]


def parse_camera_offset_file(offset_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses the trajectory offset file used to generate extra cameras.

    Args:
        offset_file: Path to the offset file. The file must contain three
            non-empty lines for ``d_phi``, ``d_theta``, and ``d_r``.

    Returns:
        A tuple containing the parsed offset arrays.

    Raises:
        ValueError: If the file format is invalid.
    """

    with open(offset_file, "r", encoding="utf-8") as file_obj:
        lines = [line.strip() for line in file_obj.readlines() if line.strip()]

    if len(lines) != 3:
        raise ValueError(
            f"Offset file must contain exactly 3 non-empty lines, got {len(lines)}: {offset_file}"
        )

    sequences = []
    for line in lines:
        try:
            values = [float(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError(
                f"Offset file contains non-numeric value: {offset_file}"
            ) from exc
        sequences.append(values)

    for line_index, sequence in enumerate(sequences, start=1):
        if len(sequence) < 2 or len(sequence) > 25:
            raise ValueError(
                f"Offset line {line_index} length must be in [2, 25], got {len(sequence)}"
            )
        if abs(sequence[0]) > 1e-9:
            raise ValueError(
                f"Offset line {line_index} must start with 0, got {sequence[0]} in {offset_file}"
            )

    return (
        np.asarray(sequences[0], dtype=np.float64),
        np.asarray(sequences[1], dtype=np.float64),
        np.asarray(sequences[2], dtype=np.float64),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the replay entrypoint."""

    parser = argparse.ArgumentParser(
        description=(
            "Replay downloaded LIBERO datasets using source actions "
            "and reconstruct equivalent hdf5 files."
        )
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Source dataset root, default get_libero_path('datasets').",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/szliutong/Desktop",
        help="Output dataset root, default '/home/szliutong/Desktop'.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to process. Default includes five LIBERO suites.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task allowlist by exact task names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print source/target file mapping, do not reconstruct.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Replay render height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Replay render width.",
    )
    parser.add_argument(
        "--no-operation-cameras",
        action="store_true",
        help="Disable the default operation cameras.",
    )
    parser.add_argument(
        "--no-trajectory-cameras",
        action="store_true",
        help="Disable the default trajectory cameras.",
    )
    parser.add_argument(
        "--camera-offset-file",
        type=str,
        default=DEFAULT_TRAJECTORY_OFFSET_FILE,
        help=(
            "Optional txt file with 3 offset lines (d_phi, d_theta, d_r). "
            "Trajectory cameras are enabled by default using this file."
        ),
    )
    parser.add_argument(
        "--camera-base-name",
        type=str,
        default=DEFAULT_OPERATION_CAMERA_BASE_NAME,
        help="Base camera name used as trajectory reference.",
    )
    parser.add_argument(
        "--trajectory-camera-count",
        type=int,
        default=DEFAULT_TRAJECTORY_CAMERA_COUNT,
        help="Number of RGB cameras uniformly sampled on the generated trajectory.",
    )
    parser.add_argument(
        "--trajectory-interp-samples",
        type=int,
        default=DEFAULT_TRAJECTORY_INTERP_SAMPLES,
        help="Dense interpolation samples before uniform trajectory sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target hdf5 file.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses CLI arguments for the replay entrypoint."""

    return build_arg_parser().parse_args(argv)


def build_replay_config(args: argparse.Namespace) -> ReplayConfig:
    """Builds a validated :class:`ReplayConfig` from parsed CLI arguments."""

    if args.trajectory_camera_count < 1:
        raise ValueError("--trajectory-camera-count must be >= 1")
    if args.trajectory_interp_samples < 2:
        raise ValueError("--trajectory-interp-samples must be >= 2")

    source_root = os.path.abspath(
        os.path.expanduser(args.source_root or get_libero_path("datasets"))
    )
    output_root = os.path.abspath(os.path.expanduser(args.output_root))

    operation_camera_names = []
    operation_camera_config = None
    if not args.no_operation_cameras:
        operation_camera_names = list(DEFAULT_OPERATION_CAMERA_NAMES.values())
        operation_camera_config = OperationCameraConfig()

    trajectory_camera_name_list = []
    trajectory_camera_config = None
    if not args.no_trajectory_cameras and args.camera_offset_file:
        offset_file = os.path.abspath(os.path.expanduser(args.camera_offset_file))
        d_phi, d_theta, d_r = parse_camera_offset_file(offset_file)
        trajectory_camera_name_list = trajectory_camera_names(
            args.camera_base_name,
            args.trajectory_camera_count,
        )
        trajectory_camera_config = TrajectoryCameraConfig(
            base_camera_name=args.camera_base_name,
            offset_file=offset_file,
            d_phi=d_phi,
            d_theta=d_theta,
            d_r=d_r,
            num_cameras=args.trajectory_camera_count,
            interp_samples=args.trajectory_interp_samples,
            camera_names=trajectory_camera_name_list,
        )

    camera_names = dedupe_keep_order(
        list(DEFAULT_CAMERA_NAMES)
        + list(operation_camera_names)
        + list(trajectory_camera_name_list)
    )

    return ReplayConfig(
        source_root=source_root,
        output_root=output_root,
        benchmarks=list(args.benchmarks),
        tasks=list(args.tasks) if args.tasks else None,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        camera_height=int(args.camera_height),
        camera_width=int(args.camera_width),
        camera_names=camera_names,
        operation_camera_names=operation_camera_names,
        trajectory_camera_names=trajectory_camera_name_list,
        operation_camera_config=operation_camera_config,
        trajectory_camera_config=trajectory_camera_config,
    )
