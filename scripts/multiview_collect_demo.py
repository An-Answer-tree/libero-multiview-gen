"""Replay LIBERO datasets into multiview HDF5 files."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"
for search_path in (str(REPO_ROOT), str(ROBOSUITE_ROOT)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

import replay_dataset_utils as replay_utils
import robosuite.utils.camera_utils as camera_utils
import robosuite.utils.transform_utils as T
from libero.libero import get_libero_path
from replay_dataset_utils import (
    DEFAULT_BENCHMARKS,
    DEFAULT_CAMERA_NAMES,
    discover_benchmark_tasks,
    reconstruct_dataset_file,
)

from multiview_collect_demo.camera_injection import (
    DEFAULT_LEGACY_ASSET_MARKERS,
    DEFAULT_OPERATION_CAMERA_BASE_NAME,
    DEFAULT_OPERATION_CAMERA_NAMES,
    OperationCameraConfig,
    TrajectoryCameraConfig,
    dedupe_keep_order,
    install_model_xml_remapper,
)


DEFAULT_CAMERA_HEIGHT = 256
DEFAULT_CAMERA_WIDTH = 256
DEFAULT_STATE_ERROR_THRESHOLD = 0.01
DEFAULT_VALUE_CHECK_ATOL = 1e-6
DEFAULT_TRAJECTORY_CAMERA_COUNT = 10
DEFAULT_TRAJECTORY_INTERP_SAMPLES = 300
DEFAULT_TRAJECTORY_OFFSET_FILE = str(
    Path(__file__).resolve().parent / "camera_offsets_example.txt"
)
CAMERA_INFO_DATASET_NAMES = (
    "intrinsics",
    "extrinsics_base_to_camera",
    "extrinsics_camera_to_base",
)
CAMERA_INFO_METADATA_NAMES = ("frame_count", "image_height", "image_width")
REQUIRED_EPISODE_KEYS = ("actions", "states", "robot_states", "rewards", "dones", "obs")


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
    divergence_threshold: float = DEFAULT_STATE_ERROR_THRESHOLD
    value_check_atol: float = DEFAULT_VALUE_CHECK_ATOL
    legacy_asset_markers: tuple[str, ...] = DEFAULT_LEGACY_ASSET_MARKERS


@dataclass(frozen=True)
class ValidationResult:
    """Stores replay validation errors."""

    errors: list[str]

    @property
    def ok(self) -> bool:
        """Returns whether validation succeeded."""

        return not self.errors


@dataclass
class ReplayStats:
    """Tracks aggregate replay progress."""

    processed: int = 0
    failed: int = 0
    total_samples: int = 0


def get_robot_base_body_name(env: Any) -> str:
    """Returns the robot base body used as the camera reference frame."""

    if not getattr(env, "robots", None):
        raise ValueError("Replay environment does not expose env.robots")
    return env.robots[0].robot_model.root_body


def get_body_pose(sim: Any, body_name: str) -> np.ndarray:
    """Returns a body pose matrix in the world frame."""

    body_id = sim.model.body_name2id(body_name)
    body_pos = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
    body_rot = np.array(sim.data.body_xmat[body_id].reshape(3, 3), dtype=np.float64)
    return T.make_pose(body_pos, body_rot)


def get_camera_info_in_base(
    env: Any,
    camera_name: str,
    camera_height: int,
    camera_width: int,
    base_body_name: str,
) -> dict[str, np.ndarray]:
    """Collects per-frame intrinsics and base-frame extrinsics for one camera."""

    intrinsics = np.array(
        camera_utils.get_camera_intrinsic_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=int(camera_height),
            camera_width=int(camera_width),
        ),
        dtype=np.float64,
    )
    camera_pose_world = np.array(
        camera_utils.get_camera_extrinsic_matrix(sim=env.sim, camera_name=camera_name),
        dtype=np.float64,
    )
    base_world_pose = get_body_pose(env.sim, base_body_name)
    camera_to_base = np.array(
        T.pose_inv(base_world_pose) @ camera_pose_world,
        dtype=np.float64,
    )
    base_to_camera = np.array(T.pose_inv(camera_to_base), dtype=np.float64)
    return {
        "intrinsics": intrinsics,
        "extrinsics_base_to_camera": base_to_camera,
        "extrinsics_camera_to_base": camera_to_base,
    }


def replay_demo_episode_with_camera_info(
    env: Any,
    source_episode_group: h5py.Group,
    camera_names: Optional[Sequence[str]] = None,
    no_proprio: bool = False,
    divergence_threshold: float = DEFAULT_STATE_ERROR_THRESHOLD,
    camera_height: int = DEFAULT_CAMERA_HEIGHT,
    camera_width: int = DEFAULT_CAMERA_WIDTH,
) -> dict[str, Any]:
    """Replays one episode and records per-frame camera intrinsics and extrinsics."""

    del no_proprio
    if camera_names is None:
        camera_names = DEFAULT_CAMERA_NAMES

    source_states = np.array(source_episode_group["states"][()])
    actions = np.array(source_episode_group["actions"][()])
    source_obs_group = source_episode_group["obs"]
    model_xml = replay_utils.decode_attr(source_episode_group.attrs["model_file"])

    if len(source_states) == 0:
        raise ValueError("Episode has empty states")
    if len(actions) == 0:
        raise ValueError("Episode has empty actions")
    if len(source_states) != len(actions):
        raise ValueError(
            f"Episode states/actions length mismatch: {len(source_states)} vs {len(actions)}"
        )

    model_xml = replay_utils.libero_utils.postprocess_model_xml(model_xml, {})
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.forward()
    model_xml = env.sim.model.get_xml()

    obs_arrays: dict[str, list[np.ndarray]] = {}
    camera_info: dict[str, dict[str, list[np.ndarray]]] = {}
    for camera_name in camera_names:
        obs_arrays[replay_utils.camera_name_to_obs_key(camera_name)] = []
        camera_info[camera_name] = {
            dataset_name: [] for dataset_name in CAMERA_INFO_DATASET_NAMES
        }

    base_body_name = get_robot_base_body_name(env)
    replay_states = []
    max_restore_error = 0.0
    num_restore_mismatches = 0

    for source_state in source_states:
        obs = replay_utils.restore_observations_from_state(env, source_state)
        restored_state = env.sim.get_state().flatten().copy()
        replay_states.append(restored_state)

        restore_error = float(np.linalg.norm(restored_state - source_state))
        max_restore_error = max(max_restore_error, restore_error)
        if restore_error > divergence_threshold:
            num_restore_mismatches += 1

        for camera_name in camera_names:
            obs_arrays[replay_utils.camera_name_to_obs_key(camera_name)].append(
                replay_utils.render_camera_observation(
                    env=env,
                    obs=obs,
                    camera_name=camera_name,
                    camera_height=camera_height,
                    camera_width=camera_width,
                )
            )
            camera_pose_info = get_camera_info_in_base(
                env=env,
                camera_name=camera_name,
                camera_height=camera_height,
                camera_width=camera_width,
                base_body_name=base_body_name,
            )
            for dataset_name, value in camera_pose_info.items():
                camera_info[camera_name][dataset_name].append(value)

    obs_data = {}
    for obs_key in source_obs_group.keys():
        if replay_utils.is_rgb_obs_key(obs_key):
            continue
        obs_data[obs_key] = np.array(source_obs_group[obs_key][()])
    for obs_key, values in obs_arrays.items():
        obs_data[obs_key] = np.stack(values, axis=0)

    camera_info_data = {}
    for camera_name, camera_data in camera_info.items():
        camera_info_data[camera_name] = {
            dataset_name: np.stack(values, axis=0)
            for dataset_name, values in camera_data.items()
        }

    if "rewards" in source_episode_group:
        rewards = np.array(source_episode_group["rewards"][()])
    else:
        rewards = np.zeros(len(actions), dtype=np.uint8)
        rewards[-1] = 1

    if "dones" in source_episode_group:
        dones = np.array(source_episode_group["dones"][()])
    else:
        dones = np.zeros(len(actions), dtype=np.uint8)
        dones[-1] = 1

    if "robot_states" in source_episode_group:
        robot_states = np.array(source_episode_group["robot_states"][()])
    else:
        raise ValueError("Episode missing robot_states")

    return {
        "actions": actions,
        "states": source_states,
        "robot_states": robot_states,
        "obs_data": obs_data,
        "camera_info": camera_info_data,
        "camera_base_body_name": base_body_name,
        "camera_height": int(camera_height),
        "camera_width": int(camera_width),
        "rewards": rewards,
        "dones": dones,
        "num_samples": int(len(actions)),
        "model_file": model_xml,
        "init_state": source_states[0],
        "replay_states": np.stack(replay_states, axis=0),
        "max_restore_error": max_restore_error,
        "num_restore_mismatches": num_restore_mismatches,
    }


def write_episode_to_hdf5_with_camera_info(
    target_data_group: h5py.Group,
    demo_key: str,
    source_episode_group: h5py.Group,
    replay_episode: dict[str, Any],
) -> None:
    """Writes one replayed episode including ``camera_info``."""

    target_ep = target_data_group.create_group(demo_key)
    for attr_key, attr_val in source_episode_group.attrs.items():
        target_ep.attrs[attr_key] = attr_val

    obs_group = target_ep.create_group("obs")
    for obs_key, obs_value in replay_episode["obs_data"].items():
        obs_group.create_dataset(obs_key, data=obs_value)

    target_ep.create_dataset("actions", data=replay_episode["actions"])
    target_ep.create_dataset("states", data=replay_episode["states"])
    target_ep.create_dataset("robot_states", data=replay_episode["robot_states"])
    target_ep.create_dataset("rewards", data=replay_episode["rewards"])
    target_ep.create_dataset("dones", data=replay_episode["dones"])

    target_ep.attrs["num_samples"] = replay_episode["num_samples"]
    target_ep.attrs["model_file"] = replay_episode["model_file"]
    target_ep.attrs["init_state"] = replay_episode["init_state"]

    camera_info_group = target_ep.create_group("camera_info")
    camera_info_group.attrs["base_body_name"] = replay_episode["camera_base_body_name"]
    for camera_name, camera_data in replay_episode["camera_info"].items():
        camera_group = camera_info_group.create_group(camera_name)
        camera_group.attrs["base_body_name"] = replay_episode["camera_base_body_name"]
        camera_group.create_dataset(
            "frame_count",
            data=np.array(camera_data["intrinsics"].shape[0], dtype=np.int32),
        )
        camera_group.create_dataset(
            "image_height",
            data=np.array(replay_episode["camera_height"], dtype=np.int32),
        )
        camera_group.create_dataset(
            "image_width",
            data=np.array(replay_episode["camera_width"], dtype=np.int32),
        )
        for dataset_name, value in camera_data.items():
            camera_group.create_dataset(dataset_name, data=value)


def install_replay_camera_info_patch() -> None:
    """Installs runtime patches so official replay code writes ``camera_info``."""

    replay_utils.replay_demo_episode = replay_demo_episode_with_camera_info
    replay_utils.write_episode_to_hdf5 = write_episode_to_hdf5_with_camera_info


def trajectory_camera_names(base_camera_name: str, num_cameras: int) -> list[str]:
    """Returns deterministic names for generated trajectory cameras."""

    return [f"{base_camera_name}_traj_{index:02d}" for index in range(num_cameras)]


def parse_camera_offset_file(offset_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses the trajectory offset file.

    Args:
        offset_file: Path to the offset file.

    Returns:
        Parsed ``(d_phi, d_theta, d_r)`` sequences.

    Raises:
        ValueError: If the file contents are malformed.
    """

    with open(offset_file, "r", encoding="utf-8") as file_obj:
        lines = [line.strip() for line in file_obj.readlines() if line.strip()]

    if len(lines) != 3:
        raise ValueError(
            f"Offset file must contain exactly 3 non-empty lines: {offset_file}"
        )

    sequences = []
    for line in lines:
        try:
            sequences.append([float(value) for value in line.split()])
        except ValueError as exc:
            raise ValueError(f"Offset file contains non-numeric value: {offset_file}") from exc

    for line_index, sequence in enumerate(sequences, start=1):
        if len(sequence) < 2 or len(sequence) > 25:
            raise ValueError(
                f"Offset line {line_index} length must be in [2, 25], got {len(sequence)}"
            )
        if abs(sequence[0]) > 1e-9:
            raise ValueError(
                f"Offset line {line_index} must start with 0, got {sequence[0]}"
            )

    return tuple(np.asarray(sequence, dtype=np.float64) for sequence in sequences)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Replay downloaded LIBERO datasets using source actions and "
            "reconstruct equivalent multiview HDF5 files."
        )
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Source dataset root. Defaults to get_libero_path('datasets').",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/szliutong/Desktop",
        help="Output dataset root.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to process.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task allowlist by exact task name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print source and target mapping without reconstruction.",
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
        help="Disable the generated operation cameras.",
    )
    parser.add_argument(
        "--no-trajectory-cameras",
        action="store_true",
        help="Disable the generated trajectory cameras.",
    )
    parser.add_argument(
        "--camera-offset-file",
        type=str,
        default=DEFAULT_TRAJECTORY_OFFSET_FILE,
        help="Offset file with d_phi, d_theta, and d_r lines.",
    )
    parser.add_argument(
        "--camera-base-name",
        type=str,
        default=DEFAULT_OPERATION_CAMERA_BASE_NAME,
        help="Base camera name used for trajectory generation.",
    )
    parser.add_argument(
        "--trajectory-camera-count",
        type=int,
        default=DEFAULT_TRAJECTORY_CAMERA_COUNT,
        help="Number of generated trajectory cameras.",
    )
    parser.add_argument(
        "--trajectory-interp-samples",
        type=int,
        default=DEFAULT_TRAJECTORY_INTERP_SAMPLES,
        help="Interpolation samples used before uniform trajectory sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing target HDF5 file.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""

    return build_arg_parser().parse_args(argv)


def build_replay_config(args: argparse.Namespace) -> ReplayConfig:
    """Builds a replay config from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Resolved replay configuration.

    Raises:
        ValueError: If trajectory arguments are invalid.
    """

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
        + operation_camera_names
        + trajectory_camera_name_list
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


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    """Returns episode keys sorted by numeric suffix."""

    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def is_rgb_obs_key(obs_key: str) -> bool:
    """Returns whether an observation key stores RGB frames."""

    return obs_key.endswith("_rgb")


def has_valid_rgb_shape(shape: tuple[int, ...]) -> bool:
    """Checks whether a tensor matches ``(T, H, W, 3)``."""

    return len(shape) == 4 and shape[-1] == 3 and shape[1] > 0 and shape[2] > 0


def camera_name_to_obs_key(camera_name: str) -> str:
    """Maps a MuJoCo camera name to its observation key."""

    if camera_name == "robot0_eye_in_hand":
        return "eye_in_hand_rgb"
    return f"{camera_name}_rgb"


def validate_replay_output(
    source_hdf5_path: str,
    rebuilt_hdf5_path: str,
    required_camera_names: Sequence[str],
    atol: float = DEFAULT_VALUE_CHECK_ATOL,
) -> ValidationResult:
    """Validates rebuilt replay output with minimal structural checks."""

    errors: list[str] = []
    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(
        rebuilt_hdf5_path, "r"
    ) as rebuilt_file:
        if "data" not in source_file:
            return ValidationResult(["Source missing data group"])
        if "data" not in rebuilt_file:
            return ValidationResult(["Rebuilt missing data group"])

        source_data = source_file["data"]
        rebuilt_data = rebuilt_file["data"]
        source_demo_key_list = sorted_demo_keys(source_data)
        rebuilt_demo_key_list = sorted_demo_keys(rebuilt_data)

        missing_demos = sorted(set(source_demo_key_list) - set(rebuilt_demo_key_list))
        extra_demos = sorted(set(rebuilt_demo_key_list) - set(source_demo_key_list))
        if missing_demos:
            errors.append(f"Rebuilt missing demos: {missing_demos}")
        if extra_demos:
            errors.append(f"Rebuilt has extra demos: {extra_demos}")

        for demo_key in source_demo_key_list:
            if demo_key not in rebuilt_data:
                continue

            rebuilt_episode = rebuilt_data[demo_key]
            for required_key in REQUIRED_EPISODE_KEYS:
                if required_key not in rebuilt_episode:
                    errors.append(f"Rebuilt {demo_key} missing key '{required_key}'")

            if "obs" not in rebuilt_episode:
                continue

            source_obs = source_data[demo_key]["obs"]
            rebuilt_obs = rebuilt_episode["obs"]
            for obs_key in sorted(source_obs.keys()):
                if not is_rgb_obs_key(obs_key):
                    continue
                if obs_key not in rebuilt_obs:
                    errors.append(f"{demo_key}/obs missing key in rebuilt: {obs_key}")
                    continue

                source_shape = source_obs[obs_key].shape
                rebuilt_shape = rebuilt_obs[obs_key].shape
                if not has_valid_rgb_shape(rebuilt_shape):
                    errors.append(
                        f"{demo_key}/obs/{obs_key} invalid rebuilt rgb shape {rebuilt_shape}"
                    )
                    continue
                if source_shape[0] != rebuilt_shape[0]:
                    errors.append(
                        f"{demo_key}/obs/{obs_key} frame count mismatch "
                        f"{source_shape[0]} vs {rebuilt_shape[0]}"
                    )

            if "camera_info" not in rebuilt_episode:
                errors.append(f"{demo_key} missing camera_info group")
                continue

            camera_info = rebuilt_episode["camera_info"]
            for camera_name in required_camera_names:
                if camera_name not in camera_info:
                    errors.append(f"{demo_key}/camera_info missing camera {camera_name}")
                    continue

                camera_group = camera_info[camera_name]
                missing_names = [
                    name
                    for name in CAMERA_INFO_DATASET_NAMES + CAMERA_INFO_METADATA_NAMES
                    if name not in camera_group
                ]
                if missing_names:
                    errors.append(
                        f"{demo_key}/camera_info/{camera_name} missing {missing_names}"
                    )
                    continue

                intrinsics = np.asarray(camera_group["intrinsics"][()])
                base_to_camera = np.asarray(
                    camera_group["extrinsics_base_to_camera"][()]
                )
                camera_to_base = np.asarray(
                    camera_group["extrinsics_camera_to_base"][()]
                )
                frame_count = int(np.asarray(camera_group["frame_count"][()]))
                image_height = int(np.asarray(camera_group["image_height"][()]))
                image_width = int(np.asarray(camera_group["image_width"][()]))

                if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
                    errors.append(
                        f"{demo_key}/camera_info/{camera_name}/intrinsics invalid shape "
                        f"{intrinsics.shape}"
                    )
                    continue
                if base_to_camera.ndim != 3 or base_to_camera.shape[1:] != (4, 4):
                    errors.append(
                        f"{demo_key}/camera_info/{camera_name}/extrinsics_base_to_camera "
                        f"invalid shape {base_to_camera.shape}"
                    )
                    continue
                if camera_to_base.ndim != 3 or camera_to_base.shape[1:] != (4, 4):
                    errors.append(
                        f"{demo_key}/camera_info/{camera_name}/extrinsics_camera_to_base "
                        f"invalid shape {camera_to_base.shape}"
                    )
                    continue

                expected_shapes = {
                    "intrinsics": (frame_count, 3, 3),
                    "extrinsics_base_to_camera": (frame_count, 4, 4),
                    "extrinsics_camera_to_base": (frame_count, 4, 4),
                }
                actual_shapes = {
                    "intrinsics": intrinsics.shape,
                    "extrinsics_base_to_camera": base_to_camera.shape,
                    "extrinsics_camera_to_base": camera_to_base.shape,
                }
                for dataset_name, expected_shape in expected_shapes.items():
                    if actual_shapes[dataset_name] != expected_shape:
                        errors.append(
                            f"{demo_key}/camera_info/{camera_name}/{dataset_name} "
                            f"shape mismatch {actual_shapes[dataset_name]} vs {expected_shape}"
                        )

                obs_key = camera_name_to_obs_key(camera_name)
                if obs_key not in rebuilt_obs:
                    errors.append(f"{demo_key}/obs missing key in rebuilt: {obs_key}")
                    continue

                obs_shape = rebuilt_obs[obs_key].shape
                if not has_valid_rgb_shape(obs_shape):
                    errors.append(
                        f"{demo_key}/obs/{obs_key} invalid rebuilt rgb shape {obs_shape}"
                    )
                    continue
                if frame_count != obs_shape[0]:
                    errors.append(
                        f"{demo_key}/camera_info/{camera_name}/frame_count mismatch "
                        f"{frame_count} vs {obs_shape[0]}"
                    )
                if image_height != obs_shape[1] or image_width != obs_shape[2]:
                    errors.append(
                        f"{demo_key}/camera_info/{camera_name} image size mismatch "
                        f"{image_width}x{image_height} vs {obs_shape[2]}x{obs_shape[1]}"
                    )

                compare_frames = min(
                    frame_count,
                    base_to_camera.shape[0],
                    camera_to_base.shape[0],
                )
                for frame_index in range(compare_frames):
                    identity = base_to_camera[frame_index] @ camera_to_base[frame_index]
                    if not np.allclose(identity, np.eye(4), atol=atol, rtol=0.0):
                        errors.append(
                            f"{demo_key}/camera_info/{camera_name} inverse mismatch "
                            f"at frame {frame_index}"
                        )
                        break

    return ValidationResult(errors)


def format_task(task_info: dict[str, Any]) -> str:
    """Formats a benchmark task for console output."""

    return (
        f"[{task_info['benchmark_name']}] {task_info['task_name']} "
        f"({task_info['relative_demo_path']})"
    )


def print_verify_commands(source_path: str, target_path: str) -> None:
    """Prints follow-up inspection commands."""

    print("  verify commands:")
    print(f"    python scripts/get_dataset_info.py --dataset {source_path}")
    print(f"    python scripts/get_dataset_info.py --dataset {target_path}")


def print_run_context(
    config: ReplayConfig,
    robosuite_root: Optional[Path],
    libero_assets_root: Path,
    legacy_markers: tuple[str, ...],
    available_tasks: list[dict[str, Any]],
    missing_tasks: list[dict[str, Any]],
) -> None:
    """Prints the resolved replay context."""

    print(f"[info] source-root: {config.source_root}")
    print(f"[info] output-root: {config.output_root}")
    print(f"[info] selected benchmarks: {config.benchmarks}")
    print(f"[info] selected tasks: {'all' if config.tasks is None else len(config.tasks)}")
    print(f"[info] available source files: {len(available_tasks)}")
    print(f"[info] missing source files: {len(missing_tasks)}")
    print(f"[info] robosuite root for replay xml: {robosuite_root}")
    print(f"[info] libero assets root for replay xml: {libero_assets_root}")
    print(f"[info] legacy asset markers: {list(legacy_markers)}")
    print(
        "[info] replay settings: "
        f"camera_names={config.camera_names}, "
        f"camera_height={config.camera_height}, "
        f"camera_width={config.camera_width}"
    )
    if config.operation_camera_config is not None:
        print(
            f"[info] operation camera base: "
            f"{config.operation_camera_config.base_camera_name}"
        )
        print(f"[info] generated operation cameras: {config.operation_camera_names}")
    if config.trajectory_camera_config is not None:
        print(
            f"[info] trajectory offset file: {config.trajectory_camera_config.offset_file}"
        )
        print(
            f"[info] trajectory base camera: "
            f"{config.trajectory_camera_config.base_camera_name}"
        )
        print(
            f"[info] trajectory camera count: "
            f"{config.trajectory_camera_config.num_cameras}"
        )
        print(
            "[info] trajectory interpolation samples: "
            f"{config.trajectory_camera_config.interp_samples}"
        )
        print(f"[info] generated trajectory cameras: {config.trajectory_camera_names}")


def print_dry_run_tasks(config: ReplayConfig, tasks: list[dict[str, Any]]) -> None:
    """Prints source-to-target mappings without reconstruction."""

    for task_info in tasks:
        source_path = task_info["source_demo_path"]
        target_path = os.path.join(config.output_root, task_info["relative_demo_path"])
        print(f"[dry-run] {format_task(task_info)}")
        print(f"          src={source_path}")
        print(f"          dst={target_path}")


def print_validation_failure(
    source_path: str,
    target_path: str,
    result: ValidationResult,
) -> None:
    """Prints validation errors for one rebuilt dataset."""

    print(f"[error] validation failed: {target_path}")
    for error in result.errors:
        print(f"  - {error}")
    print_verify_commands(source_path, target_path)


def handle_successful_replay(
    source_path: str,
    target_path: str,
    summary: dict[str, Any],
    stats: ReplayStats,
) -> None:
    """Updates counters and prints the replay summary."""

    stats.processed += 1
    stats.total_samples += summary["total_samples"]
    max_restore_error = (
        max(episode["max_restore_error"] for episode in summary["episodes"])
        if summary["episodes"]
        else 0.0
    )
    restore_mismatch_steps = sum(
        episode["num_restore_mismatches"] for episode in summary["episodes"]
    )
    print(
        "[ok] demos={num_demos}, transitions={num_samples}, "
        "rendered_from_source_states={rendered}, max_restore_error={max_err:.6f}, "
        "restore_mismatches={mismatches}".format(
            num_demos=summary["num_demos"],
            num_samples=summary["total_samples"],
            rendered=summary["total_samples"],
            max_err=max_restore_error,
            mismatches=restore_mismatch_steps,
        )
    )
    print_verify_commands(source_path, target_path)


def run_replay(config: ReplayConfig) -> None:
    """Runs the replay pipeline.

    Args:
        config: Replay configuration.

    Raises:
        FileNotFoundError: If no source datasets match the filters.
        RuntimeError: If one or more files fail replay or validation.
    """

    install_replay_camera_info_patch()
    robosuite_root, libero_assets_root, legacy_markers = install_model_xml_remapper(
        libero_assets_root=None,
        legacy_asset_markers=config.legacy_asset_markers,
        operation_config=config.operation_camera_config,
        trajectory_config=config.trajectory_camera_config,
    )
    available_tasks, missing_tasks = discover_benchmark_tasks(
        source_root=config.source_root,
        benchmark_names=config.benchmarks,
        task_filter=config.tasks,
    )
    print_run_context(
        config=config,
        robosuite_root=robosuite_root,
        libero_assets_root=libero_assets_root,
        legacy_markers=legacy_markers,
        available_tasks=available_tasks,
        missing_tasks=missing_tasks,
    )

    for missing_task in missing_tasks:
        print(f"[warning] missing source demo: {missing_task['source_demo_path']}")

    if not available_tasks:
        raise FileNotFoundError("No source HDF5 found for selected benchmark or task filters")

    if config.dry_run:
        print_dry_run_tasks(config, available_tasks)
        return

    stats = ReplayStats()
    for task_info in available_tasks:
        source_path = task_info["source_demo_path"]
        target_path = os.path.join(config.output_root, task_info["relative_demo_path"])
        target_file = Path(target_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if target_file.exists():
            if config.overwrite:
                target_file.unlink()
            else:
                raise FileExistsError(f"Target file exists: {target_path}. Use --overwrite.")

        print(f"[replay] {format_task(task_info)}")
        try:
            summary = reconstruct_dataset_file(
                source_hdf5_path=source_path,
                output_hdf5_path=target_path,
                camera_names=config.camera_names,
                divergence_threshold=config.divergence_threshold,
                camera_height=config.camera_height,
                camera_width=config.camera_width,
            )
            result = validate_replay_output(
                source_hdf5_path=source_path,
                rebuilt_hdf5_path=target_path,
                required_camera_names=config.camera_names,
                atol=config.value_check_atol,
            )
        except Exception as exc:
            stats.failed += 1
            print(f"[error] replay failed: {exc}")
            continue

        if not result.ok:
            stats.failed += 1
            print_validation_failure(source_path, target_path, result)
            continue

        handle_successful_replay(
            source_path=source_path,
            target_path=target_path,
            summary=summary,
            stats=stats,
        )

    print("========================================")
    print(
        "[done] processed={processed}, failed={failed}, total_transitions={total_samples}".format(
            processed=stats.processed,
            failed=stats.failed,
            total_samples=stats.total_samples,
        )
    )
    print("[note] step3/step4 are deferred in this script.")

    if stats.failed > 0:
        raise RuntimeError(f"Reconstruction finished with {stats.failed} failed files")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parses arguments and runs the replay pipeline."""

    args = parse_args(argv)
    config = build_replay_config(args)
    run_replay(config)


if __name__ == "__main__":
    main()
