"""Replay LIBERO datasets into multiview HDF5 files."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOSUITE_ROOT = REPO_ROOT / "third_party" / "robosuite"


def _install_repo_python_paths() -> None:
    """Adds repo-local import roots to ``sys.path``."""

    for search_path in (REPO_ROOT, ROBOSUITE_ROOT):
        path_str = os.fspath(search_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_install_repo_python_paths()

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
SCENE4_MISSING_GEOM_NAME = "new_salad_dressing_1_g0"
SCENE4_SOURCE_OBJECT_NAME = "salad_dressing_1"
SCENE4_TARGET_OBJECT_NAME = "new_salad_dressing_1"
DEFAULT_PROGRESS_DESC = "Replaying tasks"
PROGRESS_TASK_NAME_LIMIT = 60


@dataclass(frozen=True)
class ReplayConfig:
    """Resolved runtime configuration for the replay pipeline."""

    source_root: Path
    output_root: Path
    benchmarks: tuple[str, ...]
    tasks: Optional[tuple[str, ...]]
    dry_run: bool
    overwrite: bool
    camera_height: int
    camera_width: int
    camera_names: tuple[str, ...]
    operation_camera_names: tuple[str, ...] = ()
    trajectory_camera_names: tuple[str, ...] = ()
    operation_camera_config: Optional[OperationCameraConfig] = None
    trajectory_camera_config: Optional[TrajectoryCameraConfig] = None
    divergence_threshold: float = DEFAULT_STATE_ERROR_THRESHOLD
    value_check_atol: float = DEFAULT_VALUE_CHECK_ATOL
    legacy_asset_markers: tuple[str, ...] = DEFAULT_LEGACY_ASSET_MARKERS


@dataclass(frozen=True)
class SourceEpisodeData:
    """Source tensors and metadata for a single demo episode."""

    actions: np.ndarray
    states: np.ndarray
    robot_states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    obs_group: h5py.Group
    model_xml: str

    @property
    def num_samples(self) -> int:
        """Returns the episode length."""

        return int(len(self.actions))


@dataclass(frozen=True)
class TaskSpec:
    """Static metadata for one replay task."""

    benchmark_name: str
    task_idx: int
    task_name: str
    relative_demo_path: str
    source_demo_path: Path

    @classmethod
    def from_mapping(cls, task_info: Mapping[str, Any]) -> "TaskSpec":
        """Builds a task spec from ``discover_benchmark_tasks`` output."""

        return cls(
            benchmark_name=str(task_info["benchmark_name"]),
            task_idx=int(task_info["task_idx"]),
            task_name=str(task_info["task_name"]),
            relative_demo_path=str(task_info["relative_demo_path"]),
            source_demo_path=Path(os.fspath(task_info["source_demo_path"])),
        )

    @property
    def label(self) -> str:
        """Returns a human-readable task label."""

        return (
            f"[{self.benchmark_name}] {self.task_name} "
            f"({self.relative_demo_path})"
        )

    def target_demo_path(self, output_root: Path) -> Path:
        """Returns the rebuilt HDF5 target path under ``output_root``."""

        return output_root / self.relative_demo_path


@dataclass(frozen=True)
class ValidationResult:
    """Validation result for one rebuilt dataset file."""

    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Returns whether validation succeeded."""

        return not self.errors


@dataclass(frozen=True)
class FileReplaySummary:
    """Compact replay summary for one rebuilt HDF5 file."""

    source_hdf5_path: Path
    output_hdf5_path: Path
    num_demos: int
    total_samples: int
    max_restore_error: float
    restore_mismatches: int

    @classmethod
    def from_mapping(cls, summary: Mapping[str, Any]) -> "FileReplaySummary":
        """Converts ``reconstruct_dataset_file`` output to a typed summary.

        Args:
            summary: Legacy summary mapping returned by ``reconstruct_dataset_file``.

        Returns:
            A typed replay summary.
        """

        episode_summaries = tuple(summary.get("episodes", ()))
        max_restore_error = 0.0
        restore_mismatches = 0
        if episode_summaries:
            max_restore_error = max(
                float(episode["max_restore_error"]) for episode in episode_summaries
            )
            restore_mismatches = sum(
                int(episode["num_restore_mismatches"])
                for episode in episode_summaries
            )

        return cls(
            source_hdf5_path=Path(os.fspath(summary["source_hdf5_path"])),
            output_hdf5_path=Path(os.fspath(summary["output_hdf5_path"])),
            num_demos=int(summary["num_demos"]),
            total_samples=int(summary["total_samples"]),
            max_restore_error=max_restore_error,
            restore_mismatches=restore_mismatches,
        )


@dataclass(frozen=True)
class TaskReplayResult:
    """Replay and validation outcome for a single task."""

    task: TaskSpec
    source_path: Path
    target_path: Path
    summary: Optional[FileReplaySummary] = None
    validation: Optional[ValidationResult] = None
    error_message: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Returns whether replay and validation both succeeded."""

        return (
            self.error_message is None
            and self.summary is not None
            and self.validation is not None
            and self.validation.ok
        )


@dataclass
class ReplayStats:
    """Aggregates run-wide replay counts."""

    processed: int = 0
    failed: int = 0
    total_samples: int = 0

    def record_success(self, total_samples: int) -> None:
        """Updates counters after a successful task replay.

        Args:
            total_samples: Number of transitions written by the task.
        """

        self.processed += 1
        self.total_samples += int(total_samples)

    def record_failure(self) -> None:
        """Updates counters after a failed task replay."""

        self.failed += 1


@dataclass
class ConsoleReporter:
    """Writes console output without breaking a live ``tqdm`` progress bar."""

    progress_bar: Optional[Any] = None

    def write(self, message: str) -> None:
        """Prints one line of output.

        Args:
            message: Message to print.
        """

        if self.progress_bar is None:
            print(message)
            return
        self.progress_bar.write(message)


def get_robot_base_body_name(env: Any) -> str:
    """Returns the robot base body used as the camera reference frame.

    Args:
        env: Replay environment.

    Returns:
        The MuJoCo body name used as the base frame.

    Raises:
        ValueError: If the replay environment does not expose robots.
    """

    if not getattr(env, "robots", None):
        raise ValueError("Replay environment does not expose env.robots")
    return env.robots[0].robot_model.root_body


def get_body_pose(sim: Any, body_name: str) -> np.ndarray:
    """Returns a body pose matrix in the world frame.

    Args:
        sim: MuJoCo simulator.
        body_name: Name of the body to query.

    Returns:
        A ``4 x 4`` pose matrix in the world frame.
    """

    body_id = sim.model.body_name2id(body_name)
    body_pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    body_rot = np.asarray(
        sim.data.body_xmat[body_id].reshape(3, 3),
        dtype=np.float64,
    )
    return T.make_pose(body_pos, body_rot)


def get_camera_info_in_base(
    env: Any,
    camera_name: str,
    camera_height: int,
    camera_width: int,
    base_body_name: str,
) -> dict[str, np.ndarray]:
    """Collects one camera's intrinsics and base-frame extrinsics.

    Args:
        env: Replay environment.
        camera_name: MuJoCo camera name.
        camera_height: Render height.
        camera_width: Render width.
        base_body_name: Robot base body name used as the reference frame.

    Returns:
        A mapping that contains camera intrinsics plus forward and inverse
        extrinsics relative to the robot base frame.
    """

    intrinsics = np.asarray(
        camera_utils.get_camera_intrinsic_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=int(camera_height),
            camera_width=int(camera_width),
        ),
        dtype=np.float64,
    )
    camera_pose_world = np.asarray(
        camera_utils.get_camera_extrinsic_matrix(sim=env.sim, camera_name=camera_name),
        dtype=np.float64,
    )
    base_world_pose = get_body_pose(env.sim, base_body_name)
    camera_to_base = np.asarray(
        T.pose_inv(base_world_pose) @ camera_pose_world,
        dtype=np.float64,
    )
    base_to_camera = np.asarray(T.pose_inv(camera_to_base), dtype=np.float64)
    return {
        "intrinsics": intrinsics,
        "extrinsics_base_to_camera": base_to_camera,
        "extrinsics_camera_to_base": camera_to_base,
    }


def build_terminal_signal(num_samples: int) -> np.ndarray:
    """Builds the default reward or done signal for one episode.

    Args:
        num_samples: Number of timesteps in the episode.

    Returns:
        A ``uint8`` vector with only the last element set to ``1``.
    """

    signal = np.zeros(num_samples, dtype=np.uint8)
    signal[-1] = 1
    return signal


def load_source_episode_data(source_episode_group: h5py.Group) -> SourceEpisodeData:
    """Loads and validates source tensors for one episode.

    Args:
        source_episode_group: Source HDF5 group for one episode.

    Returns:
        Parsed episode data.

    Raises:
        ValueError: If required tensors are missing or have invalid shapes.
        KeyError: If required HDF5 groups are missing.
    """

    states = np.asarray(source_episode_group["states"][()])
    actions = np.asarray(source_episode_group["actions"][()])
    obs_group = source_episode_group["obs"]
    model_xml = replay_utils.decode_attr(source_episode_group.attrs["model_file"])

    if len(states) == 0:
        raise ValueError("Episode has empty states")
    if len(actions) == 0:
        raise ValueError("Episode has empty actions")
    if len(states) != len(actions):
        raise ValueError(
            f"Episode states/actions length mismatch: {len(states)} vs {len(actions)}"
        )

    if "robot_states" not in source_episode_group:
        raise ValueError("Episode missing robot_states")

    rewards = (
        np.asarray(source_episode_group["rewards"][()])
        if "rewards" in source_episode_group
        else build_terminal_signal(len(actions))
    )
    dones = (
        np.asarray(source_episode_group["dones"][()])
        if "dones" in source_episode_group
        else build_terminal_signal(len(actions))
    )

    return SourceEpisodeData(
        actions=actions,
        states=states,
        robot_states=np.asarray(source_episode_group["robot_states"][()]),
        rewards=rewards,
        dones=dones,
        obs_group=obs_group,
        model_xml=model_xml,
    )


def copy_non_rgb_observations(obs_group: h5py.Group) -> dict[str, np.ndarray]:
    """Copies non-RGB observation tensors from the source episode.

    Args:
        obs_group: Source observation group.

    Returns:
        A mapping from observation key to numpy array.
    """

    obs_data: dict[str, np.ndarray] = {}
    for obs_key in obs_group.keys():
        if replay_utils.is_rgb_obs_key(obs_key):
            continue
        obs_data[obs_key] = np.asarray(obs_group[obs_key][()])
    return obs_data


def initialize_camera_buffers(
    camera_names: Sequence[str],
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict[str, list[np.ndarray]]]]:
    """Creates per-camera buffers for RGB frames and camera parameters.

    Args:
        camera_names: Cameras that should be rendered.

    Returns:
        A tuple of ``(obs_buffers, camera_info_buffers)``.
    """

    obs_buffers = {
        replay_utils.camera_name_to_obs_key(camera_name): []
        for camera_name in camera_names
    }
    camera_info_buffers = {
        camera_name: {dataset_name: [] for dataset_name in CAMERA_INFO_DATASET_NAMES}
        for camera_name in camera_names
    }
    return obs_buffers, camera_info_buffers


def append_camera_frame_data(
    env: Any,
    obs: Mapping[str, Any],
    camera_name: str,
    camera_height: int,
    camera_width: int,
    base_body_name: str,
    obs_buffers: dict[str, list[np.ndarray]],
    camera_info_buffers: dict[str, dict[str, list[np.ndarray]]],
) -> None:
    """Appends one frame of RGB data and camera parameters.

    Args:
        env: Replay environment.
        obs: Environment observation mapping after state restore.
        camera_name: Camera to render.
        camera_height: Render height.
        camera_width: Render width.
        base_body_name: Robot base body used as the reference frame.
        obs_buffers: RGB frame buffers keyed by observation key.
        camera_info_buffers: Camera parameter buffers keyed by camera name.
    """

    obs_key = replay_utils.camera_name_to_obs_key(camera_name)
    obs_buffers[obs_key].append(
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
        camera_info_buffers[camera_name][dataset_name].append(value)


def finalize_camera_buffers(
    source_episode: SourceEpisodeData,
    obs_buffers: dict[str, list[np.ndarray]],
    camera_info_buffers: dict[str, dict[str, list[np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Stacks buffered observations and camera parameters.

    Args:
        source_episode: Parsed source episode tensors.
        obs_buffers: RGB frame buffers.
        camera_info_buffers: Camera parameter buffers.

    Returns:
        A tuple of ``(obs_data, camera_info_data)`` ready for HDF5 writing.
    """

    obs_data = copy_non_rgb_observations(source_episode.obs_group)
    for obs_key, values in obs_buffers.items():
        obs_data[obs_key] = np.stack(values, axis=0)

    camera_info_data = {}
    for camera_name, camera_data in camera_info_buffers.items():
        camera_info_data[camera_name] = {
            dataset_name: np.stack(values, axis=0)
            for dataset_name, values in camera_data.items()
        }

    return obs_data, camera_info_data


def maybe_patch_scene4_salad_dressing_xml(
    model_xml: str,
    reset_error: Exception,
) -> Optional[str]:
    """Repairs the known Scene4 salad-dressing naming mismatch.

    Args:
        model_xml: Postprocessed episode XML.
        reset_error: Exception raised by ``env.reset_from_xml_string``.

    Returns:
        The patched XML if the known naming mismatch is detected. Otherwise
        returns ``None``.
    """

    if SCENE4_MISSING_GEOM_NAME not in str(reset_error):
        return None
    source_geom_name = f"{SCENE4_SOURCE_OBJECT_NAME}_g0"
    if source_geom_name not in model_xml:
        return None

    patched_model_xml = model_xml.replace(
        SCENE4_SOURCE_OBJECT_NAME,
        SCENE4_TARGET_OBJECT_NAME,
    )
    if patched_model_xml == model_xml:
        return None
    return patched_model_xml


def reset_env_from_model_xml(env: Any, model_xml: str) -> str:
    """Resets the replay environment from the serialized episode XML.

    Args:
        env: Replay environment.
        model_xml: Serialized MuJoCo XML from the source episode.

    Returns:
        The fully resolved XML currently loaded in the simulator.

    Raises:
        ValueError: If MuJoCo fails to load the XML and no compatibility patch
            applies.
    """

    resolved_model_xml = replay_utils.libero_utils.postprocess_model_xml(model_xml, {})
    try:
        env.reset_from_xml_string(resolved_model_xml)
    except ValueError as exc:
        patched_model_xml = maybe_patch_scene4_salad_dressing_xml(
            resolved_model_xml,
            exc,
        )
        if patched_model_xml is None:
            raise
        resolved_model_xml = patched_model_xml
        env.reset_from_xml_string(resolved_model_xml)

    env.sim.reset()
    env.sim.forward()
    return env.sim.model.get_xml()


def replay_demo_episode_with_camera_info(
    env: Any,
    source_episode_group: h5py.Group,
    camera_names: Optional[Sequence[str]] = None,
    no_proprio: bool = False,
    divergence_threshold: float = DEFAULT_STATE_ERROR_THRESHOLD,
    camera_height: int = DEFAULT_CAMERA_HEIGHT,
    camera_width: int = DEFAULT_CAMERA_WIDTH,
) -> dict[str, Any]:
    """Replays one episode and records per-frame camera intrinsics/extrinsics.

    Args:
        env: Replay environment.
        source_episode_group: Source episode group.
        camera_names: Cameras to render and record.
        no_proprio: Unused compatibility argument from the official replay API.
        divergence_threshold: Threshold used to count restore mismatches.
        camera_height: Render height.
        camera_width: Render width.

    Returns:
        A replay payload compatible with ``replay_dataset_utils``.
    """

    del no_proprio
    if camera_names is None:
        camera_names = DEFAULT_CAMERA_NAMES

    source_episode = load_source_episode_data(source_episode_group)
    model_xml = reset_env_from_model_xml(env, source_episode.model_xml)

    obs_buffers, camera_info_buffers = initialize_camera_buffers(camera_names)
    base_body_name = get_robot_base_body_name(env)
    replay_states = []
    max_restore_error = 0.0
    num_restore_mismatches = 0

    for source_state in source_episode.states:
        obs = replay_utils.restore_observations_from_state(env, source_state)
        restored_state = env.sim.get_state().flatten().copy()
        replay_states.append(restored_state)

        restore_error = float(np.linalg.norm(restored_state - source_state))
        max_restore_error = max(max_restore_error, restore_error)
        if restore_error > divergence_threshold:
            num_restore_mismatches += 1

        for camera_name in camera_names:
            append_camera_frame_data(
                env=env,
                obs=obs,
                camera_name=camera_name,
                camera_height=camera_height,
                camera_width=camera_width,
                base_body_name=base_body_name,
                obs_buffers=obs_buffers,
                camera_info_buffers=camera_info_buffers,
            )

    obs_data, camera_info_data = finalize_camera_buffers(
        source_episode=source_episode,
        obs_buffers=obs_buffers,
        camera_info_buffers=camera_info_buffers,
    )

    return {
        "actions": source_episode.actions,
        "states": source_episode.states,
        "robot_states": source_episode.robot_states,
        "obs_data": obs_data,
        "camera_info": camera_info_data,
        "camera_base_body_name": base_body_name,
        "camera_height": int(camera_height),
        "camera_width": int(camera_width),
        "rewards": source_episode.rewards,
        "dones": source_episode.dones,
        "num_samples": source_episode.num_samples,
        "model_file": model_xml,
        "init_state": source_episode.states[0],
        "replay_states": np.stack(replay_states, axis=0),
        "max_restore_error": max_restore_error,
        "num_restore_mismatches": num_restore_mismatches,
    }


def write_scalar_dataset(
    group: h5py.Group,
    dataset_name: str,
    value: int,
) -> None:
    """Writes one scalar int32 dataset.

    Args:
        group: Parent HDF5 group.
        dataset_name: Dataset name to create.
        value: Scalar integer value to write.
    """

    group.create_dataset(dataset_name, data=np.asarray(value, dtype=np.int32))


def write_camera_info_group(
    camera_info_group: h5py.Group,
    replay_episode: Mapping[str, Any],
) -> None:
    """Writes the ``camera_info`` subtree for one episode.

    Args:
        camera_info_group: Destination HDF5 group.
        replay_episode: Replay payload compatible with the patched replay API.
    """

    base_body_name = replay_episode["camera_base_body_name"]
    camera_info_group.attrs["base_body_name"] = base_body_name

    for camera_name, camera_data in replay_episode["camera_info"].items():
        camera_group = camera_info_group.create_group(camera_name)
        camera_group.attrs["base_body_name"] = base_body_name
        write_scalar_dataset(
            camera_group,
            "frame_count",
            int(camera_data["intrinsics"].shape[0]),
        )
        write_scalar_dataset(
            camera_group,
            "image_height",
            int(replay_episode["camera_height"]),
        )
        write_scalar_dataset(
            camera_group,
            "image_width",
            int(replay_episode["camera_width"]),
        )
        for dataset_name, value in camera_data.items():
            camera_group.create_dataset(dataset_name, data=value)


def write_episode_to_hdf5_with_camera_info(
    target_data_group: h5py.Group,
    demo_key: str,
    source_episode_group: h5py.Group,
    replay_episode: Mapping[str, Any],
) -> None:
    """Writes one replayed episode including ``camera_info``.

    Args:
        target_data_group: Destination ``data`` group in the rebuilt HDF5.
        demo_key: Episode key, for example ``demo_0``.
        source_episode_group: Source episode group.
        replay_episode: Replay payload compatible with the patched replay API.
    """

    target_episode = target_data_group.create_group(demo_key)
    for attr_key, attr_val in source_episode_group.attrs.items():
        target_episode.attrs[attr_key] = attr_val

    obs_group = target_episode.create_group("obs")
    for obs_key, obs_value in replay_episode["obs_data"].items():
        obs_group.create_dataset(obs_key, data=obs_value)

    target_episode.create_dataset("actions", data=replay_episode["actions"])
    target_episode.create_dataset("states", data=replay_episode["states"])
    target_episode.create_dataset("robot_states", data=replay_episode["robot_states"])
    target_episode.create_dataset("rewards", data=replay_episode["rewards"])
    target_episode.create_dataset("dones", data=replay_episode["dones"])

    target_episode.attrs["num_samples"] = replay_episode["num_samples"]
    target_episode.attrs["model_file"] = replay_episode["model_file"]
    target_episode.attrs["init_state"] = replay_episode["init_state"]

    write_camera_info_group(
        target_episode.create_group("camera_info"),
        replay_episode,
    )


def install_replay_camera_info_patch() -> None:
    """Installs runtime patches so replay writes ``camera_info`` datasets."""

    replay_utils.replay_demo_episode = replay_demo_episode_with_camera_info
    replay_utils.write_episode_to_hdf5 = write_episode_to_hdf5_with_camera_info


def trajectory_camera_names(base_camera_name: str, num_cameras: int) -> list[str]:
    """Returns deterministic names for generated trajectory cameras.

    Args:
        base_camera_name: Camera name used as the naming prefix.
        num_cameras: Number of generated cameras.

    Returns:
        Deterministic trajectory camera names.
    """

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
            raise ValueError(
                f"Offset file contains non-numeric value: {offset_file}"
            ) from exc

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
    """Builds the command-line parser.

    Returns:
        Configured argument parser for the replay script.
    """

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
    """Parses command-line arguments.

    Args:
        argv: Optional argument list. When ``None``, argparse reads ``sys.argv``.

    Returns:
        Parsed argparse namespace.
    """

    return build_arg_parser().parse_args(argv)


def build_operation_camera_settings(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], Optional[OperationCameraConfig]]:
    """Builds generated fixed-camera settings from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple of ``(camera_names, config)`` for generated operation cameras.
    """

    if args.no_operation_cameras:
        return (), None
    return tuple(DEFAULT_OPERATION_CAMERA_NAMES.values()), OperationCameraConfig()


def build_trajectory_camera_settings(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], Optional[TrajectoryCameraConfig]]:
    """Builds trajectory-camera settings from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple of ``(camera_names, config)`` for generated trajectory cameras.

    Raises:
        ValueError: If trajectory arguments are invalid.
    """

    if args.trajectory_camera_count < 1:
        raise ValueError("--trajectory-camera-count must be >= 1")
    if args.trajectory_interp_samples < 2:
        raise ValueError("--trajectory-interp-samples must be >= 2")
    if args.no_trajectory_cameras or not args.camera_offset_file:
        return (), None

    offset_file = os.path.abspath(os.path.expanduser(args.camera_offset_file))
    d_phi, d_theta, d_r = parse_camera_offset_file(offset_file)
    camera_name_list = tuple(
        trajectory_camera_names(
            args.camera_base_name,
            args.trajectory_camera_count,
        )
    )
    trajectory_config = TrajectoryCameraConfig(
        base_camera_name=args.camera_base_name,
        offset_file=offset_file,
        d_phi=d_phi,
        d_theta=d_theta,
        d_r=d_r,
        num_cameras=args.trajectory_camera_count,
        interp_samples=args.trajectory_interp_samples,
        camera_names=list(camera_name_list),
    )
    return camera_name_list, trajectory_config


def build_replay_config(args: argparse.Namespace) -> ReplayConfig:
    """Builds a replay config from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Resolved replay configuration.

    Raises:
        ValueError: If trajectory arguments are invalid.
    """

    operation_camera_names, operation_camera_config = build_operation_camera_settings(
        args
    )
    trajectory_camera_names_, trajectory_camera_config = (
        build_trajectory_camera_settings(args)
    )

    source_root = Path(
        os.path.abspath(
            os.path.expanduser(args.source_root or get_libero_path("datasets"))
        )
    )
    output_root = Path(os.path.abspath(os.path.expanduser(args.output_root)))
    camera_names = tuple(
        dedupe_keep_order(
            list(DEFAULT_CAMERA_NAMES)
            + list(operation_camera_names)
            + list(trajectory_camera_names_)
        )
    )

    return ReplayConfig(
        source_root=source_root,
        output_root=output_root,
        benchmarks=tuple(args.benchmarks),
        tasks=tuple(args.tasks) if args.tasks else None,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        camera_height=int(args.camera_height),
        camera_width=int(args.camera_width),
        camera_names=camera_names,
        operation_camera_names=operation_camera_names,
        trajectory_camera_names=trajectory_camera_names_,
        operation_camera_config=operation_camera_config,
        trajectory_camera_config=trajectory_camera_config,
    )


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    """Returns episode keys sorted by numeric suffix.

    Args:
        data_group: Source or rebuilt HDF5 ``data`` group.

    Returns:
        Sorted demo keys such as ``["demo_0", "demo_1"]``.
    """

    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def is_rgb_obs_key(obs_key: str) -> bool:
    """Returns whether an observation key stores RGB frames."""

    return obs_key.endswith("_rgb")


def has_valid_rgb_shape(shape: tuple[int, ...]) -> bool:
    """Checks whether a tensor matches ``(T, H, W, 3)``."""

    return len(shape) == 4 and shape[-1] == 3 and shape[1] > 0 and shape[2] > 0


def camera_name_to_obs_key(camera_name: str) -> str:
    """Maps a MuJoCo camera name to its RGB observation key."""

    if camera_name == "robot0_eye_in_hand":
        return "eye_in_hand_rgb"
    return f"{camera_name}_rgb"


def validate_demo_membership(
    source_demo_keys: Sequence[str],
    rebuilt_demo_keys: Sequence[str],
) -> list[str]:
    """Validates whether rebuilt demos match the source demo set."""

    errors = []
    missing_demos = sorted(set(source_demo_keys) - set(rebuilt_demo_keys))
    extra_demos = sorted(set(rebuilt_demo_keys) - set(source_demo_keys))
    if missing_demos:
        errors.append(f"Rebuilt missing demos: {missing_demos}")
    if extra_demos:
        errors.append(f"Rebuilt has extra demos: {extra_demos}")
    return errors


def validate_required_episode_keys(
    demo_key: str,
    rebuilt_episode: h5py.Group,
) -> list[str]:
    """Validates the required top-level datasets for one rebuilt episode."""

    errors = []
    for required_key in REQUIRED_EPISODE_KEYS:
        if required_key not in rebuilt_episode:
            errors.append(f"Rebuilt {demo_key} missing key '{required_key}'")
    return errors


def validate_rgb_observations(
    demo_key: str,
    source_obs: h5py.Group,
    rebuilt_obs: h5py.Group,
) -> list[str]:
    """Validates RGB observation keys and frame counts for one demo."""

    errors = []
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
    return errors


def validate_camera_metadata(
    demo_key: str,
    camera_name: str,
    camera_group: h5py.Group,
    rebuilt_obs: h5py.Group,
    atol: float,
) -> list[str]:
    """Validates one camera's ``camera_info`` datasets against RGB observations."""

    errors = []
    missing_names = [
        name
        for name in CAMERA_INFO_DATASET_NAMES + CAMERA_INFO_METADATA_NAMES
        if name not in camera_group
    ]
    if missing_names:
        errors.append(f"{demo_key}/camera_info/{camera_name} missing {missing_names}")
        return errors

    intrinsics = np.asarray(camera_group["intrinsics"][()])
    base_to_camera = np.asarray(camera_group["extrinsics_base_to_camera"][()])
    camera_to_base = np.asarray(camera_group["extrinsics_camera_to_base"][()])
    frame_count = int(np.asarray(camera_group["frame_count"][()]))
    image_height = int(np.asarray(camera_group["image_height"][()]))
    image_width = int(np.asarray(camera_group["image_width"][()]))

    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        errors.append(
            f"{demo_key}/camera_info/{camera_name}/intrinsics invalid shape "
            f"{intrinsics.shape}"
        )
        return errors
    if base_to_camera.ndim != 3 or base_to_camera.shape[1:] != (4, 4):
        errors.append(
            f"{demo_key}/camera_info/{camera_name}/extrinsics_base_to_camera "
            f"invalid shape {base_to_camera.shape}"
        )
        return errors
    if camera_to_base.ndim != 3 or camera_to_base.shape[1:] != (4, 4):
        errors.append(
            f"{demo_key}/camera_info/{camera_name}/extrinsics_camera_to_base "
            f"invalid shape {camera_to_base.shape}"
        )
        return errors

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
        return errors

    obs_shape = rebuilt_obs[obs_key].shape
    if not has_valid_rgb_shape(obs_shape):
        errors.append(f"{demo_key}/obs/{obs_key} invalid rebuilt rgb shape {obs_shape}")
        return errors
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

    compare_frames = min(frame_count, base_to_camera.shape[0], camera_to_base.shape[0])
    identity = np.eye(4, dtype=np.float64)
    for frame_index in range(compare_frames):
        transform_product = base_to_camera[frame_index] @ camera_to_base[frame_index]
        if not np.allclose(transform_product, identity, atol=atol, rtol=0.0):
            errors.append(
                f"{demo_key}/camera_info/{camera_name} inverse mismatch "
                f"at frame {frame_index}"
            )
            break

    return errors


def validate_rebuilt_episode(
    demo_key: str,
    source_episode: h5py.Group,
    rebuilt_episode: h5py.Group,
    required_camera_names: Sequence[str],
    atol: float,
) -> list[str]:
    """Validates one rebuilt episode.

    Args:
        demo_key: Demo key under validation.
        source_episode: Source episode group.
        rebuilt_episode: Rebuilt episode group.
        required_camera_names: Cameras that must exist under ``camera_info``.
        atol: Numerical tolerance for transform inverse checks.

    Returns:
        Flat validation error messages for the episode.
    """

    errors = validate_required_episode_keys(demo_key, rebuilt_episode)
    if "obs" not in rebuilt_episode:
        return errors

    source_obs = source_episode["obs"]
    rebuilt_obs = rebuilt_episode["obs"]
    errors.extend(validate_rgb_observations(demo_key, source_obs, rebuilt_obs))

    if "camera_info" not in rebuilt_episode:
        errors.append(f"{demo_key} missing camera_info group")
        return errors

    camera_info = rebuilt_episode["camera_info"]
    for camera_name in required_camera_names:
        if camera_name not in camera_info:
            errors.append(f"{demo_key}/camera_info missing camera {camera_name}")
            continue
        errors.extend(
            validate_camera_metadata(
                demo_key=demo_key,
                camera_name=camera_name,
                camera_group=camera_info[camera_name],
                rebuilt_obs=rebuilt_obs,
                atol=atol,
            )
        )

    return errors


def validate_replay_output(
    source_hdf5_path: str | Path,
    rebuilt_hdf5_path: str | Path,
    required_camera_names: Sequence[str],
    atol: float = DEFAULT_VALUE_CHECK_ATOL,
) -> ValidationResult:
    """Validates rebuilt replay output with lightweight structural checks.

    Args:
        source_hdf5_path: Source HDF5 path.
        rebuilt_hdf5_path: Rebuilt HDF5 path.
        required_camera_names: Cameras that must exist in every rebuilt episode.
        atol: Numerical tolerance for inverse transform checks.

    Returns:
        Validation result for the rebuilt file.
    """

    errors: list[str] = []
    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(
        rebuilt_hdf5_path,
        "r",
    ) as rebuilt_file:
        if "data" not in source_file:
            return ValidationResult(("Source missing data group",))
        if "data" not in rebuilt_file:
            return ValidationResult(("Rebuilt missing data group",))

        source_data = source_file["data"]
        rebuilt_data = rebuilt_file["data"]
        source_demo_keys = sorted_demo_keys(source_data)
        rebuilt_demo_keys = sorted_demo_keys(rebuilt_data)

        errors.extend(validate_demo_membership(source_demo_keys, rebuilt_demo_keys))

        for demo_key in source_demo_keys:
            if demo_key not in rebuilt_data:
                continue
            errors.extend(
                validate_rebuilt_episode(
                    demo_key=demo_key,
                    source_episode=source_data[demo_key],
                    rebuilt_episode=rebuilt_data[demo_key],
                    required_camera_names=required_camera_names,
                    atol=atol,
                )
            )

    return ValidationResult(tuple(errors))


def discover_task_specs(
    source_root: Path,
    benchmark_names: Sequence[str],
    task_filter: Optional[Sequence[str]],
) -> tuple[list[TaskSpec], list[TaskSpec]]:
    """Discovers available and missing source datasets as typed task specs."""

    available_tasks, missing_tasks = discover_benchmark_tasks(
        source_root=os.fspath(source_root),
        benchmark_names=list(benchmark_names),
        task_filter=list(task_filter) if task_filter else None,
    )
    return (
        [TaskSpec.from_mapping(task_info) for task_info in available_tasks],
        [TaskSpec.from_mapping(task_info) for task_info in missing_tasks],
    )


def truncate_progress_text(text: str, limit: int = PROGRESS_TASK_NAME_LIMIT) -> str:
    """Truncates long task names for progress-bar display."""

    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def report_verify_commands(
    reporter: ConsoleReporter,
    source_path: Path,
    target_path: Path,
) -> None:
    """Prints follow-up inspection commands for one rebuilt file."""

    reporter.write("  verify commands:")
    reporter.write(
        f"    python scripts/get_dataset_info.py --dataset {source_path}"
    )
    reporter.write(
        f"    python scripts/get_dataset_info.py --dataset {target_path}"
    )


def report_run_context(
    reporter: ConsoleReporter,
    config: ReplayConfig,
    robosuite_root: Optional[Path],
    libero_assets_root: Path,
    legacy_markers: tuple[str, ...],
    available_tasks: Sequence[TaskSpec],
    missing_tasks: Sequence[TaskSpec],
) -> None:
    """Prints the resolved replay context before running tasks."""

    reporter.write(f"[info] source-root: {config.source_root}")
    reporter.write(f"[info] output-root: {config.output_root}")
    reporter.write(f"[info] selected benchmarks: {list(config.benchmarks)}")
    reporter.write(
        f"[info] selected tasks: {'all' if config.tasks is None else len(config.tasks)}"
    )
    reporter.write(f"[info] available source files: {len(available_tasks)}")
    reporter.write(f"[info] missing source files: {len(missing_tasks)}")
    reporter.write(f"[info] robosuite root for replay xml: {robosuite_root}")
    reporter.write(f"[info] libero assets root for replay xml: {libero_assets_root}")
    reporter.write(f"[info] legacy asset markers: {list(legacy_markers)}")
    reporter.write(
        "[info] replay settings: "
        f"camera_names={list(config.camera_names)}, "
        f"camera_height={config.camera_height}, "
        f"camera_width={config.camera_width}"
    )
    if config.operation_camera_config is not None:
        reporter.write(
            f"[info] operation camera base: "
            f"{config.operation_camera_config.base_camera_name}"
        )
        reporter.write(
            f"[info] generated operation cameras: {list(config.operation_camera_names)}"
        )
    if config.trajectory_camera_config is not None:
        reporter.write(
            f"[info] trajectory offset file: "
            f"{config.trajectory_camera_config.offset_file}"
        )
        reporter.write(
            f"[info] trajectory base camera: "
            f"{config.trajectory_camera_config.base_camera_name}"
        )
        reporter.write(
            f"[info] trajectory camera count: "
            f"{config.trajectory_camera_config.num_cameras}"
        )
        reporter.write(
            "[info] trajectory interpolation samples: "
            f"{config.trajectory_camera_config.interp_samples}"
        )
        reporter.write(
            f"[info] generated trajectory cameras: "
            f"{list(config.trajectory_camera_names)}"
        )


def report_dry_run_tasks(
    reporter: ConsoleReporter,
    config: ReplayConfig,
    tasks: Sequence[TaskSpec],
) -> None:
    """Prints source-to-target mappings without reconstruction."""

    for task in tasks:
        reporter.write(f"[dry-run] {task.label}")
        reporter.write(f"          src={task.source_demo_path}")
        reporter.write(f"          dst={task.target_demo_path(config.output_root)}")


def report_validation_failure(
    reporter: ConsoleReporter,
    result: TaskReplayResult,
) -> None:
    """Prints validation errors for one rebuilt dataset file."""

    reporter.write(f"[error] validation failed: {result.target_path}")
    for error in result.validation.errors:
        reporter.write(f"  - {error}")
    report_verify_commands(reporter, result.source_path, result.target_path)


def report_task_failure(
    reporter: ConsoleReporter,
    result: TaskReplayResult,
) -> None:
    """Prints a replay exception for one failed task."""

    reporter.write(
        f"[error] replay failed: {result.task.label}: {result.error_message}"
    )


def report_task_success(
    reporter: ConsoleReporter,
    result: TaskReplayResult,
) -> None:
    """Prints success summary for one rebuilt task."""

    reporter.write(
        "[ok] demos={num_demos}, transitions={num_samples}, "
        "rendered_from_source_states={rendered}, max_restore_error={max_err:.6f}, "
        "restore_mismatches={mismatches}".format(
            num_demos=result.summary.num_demos,
            num_samples=result.summary.total_samples,
            rendered=result.summary.total_samples,
            max_err=result.summary.max_restore_error,
            mismatches=result.summary.restore_mismatches,
        )
    )
    report_verify_commands(reporter, result.source_path, result.target_path)


def report_missing_tasks(
    reporter: ConsoleReporter,
    missing_tasks: Sequence[TaskSpec],
) -> None:
    """Prints warnings for missing source datasets."""

    for task in missing_tasks:
        reporter.write(f"[warning] missing source demo: {task.source_demo_path}")


def report_run_summary(
    reporter: ConsoleReporter,
    stats: ReplayStats,
) -> None:
    """Prints the final run summary."""

    reporter.write("========================================")
    reporter.write(
        "[done] processed={processed}, failed={failed}, total_transitions={total_samples}".format(
            processed=stats.processed,
            failed=stats.failed,
            total_samples=stats.total_samples,
        )
    )
    reporter.write("[note] step3/step4 are deferred in this script.")


def prepare_target_path(target_path: Path, overwrite: bool) -> None:
    """Creates the target directory and handles overwrite behavior.

    Args:
        target_path: Destination HDF5 path.
        overwrite: Whether to delete an existing target file.

    Raises:
        FileExistsError: If the target file exists and overwrite is disabled.
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        return
    if overwrite:
        target_path.unlink()
        return
    raise FileExistsError(f"Target file exists: {target_path}. Use --overwrite.")


def replay_task(task: TaskSpec, config: ReplayConfig) -> TaskReplayResult:
    """Rebuilds and validates one task dataset.

    Args:
        task: Task to rebuild.
        config: Resolved replay configuration.

    Returns:
        Replay result for the task. Exceptions during replay are captured inside
        the returned result. Target-path overwrite errors are raised before this
        function returns.
    """

    target_path = task.target_demo_path(config.output_root)
    prepare_target_path(target_path, config.overwrite)

    try:
        summary = FileReplaySummary.from_mapping(
            reconstruct_dataset_file(
                source_hdf5_path=os.fspath(task.source_demo_path),
                output_hdf5_path=os.fspath(target_path),
                camera_names=list(config.camera_names),
                divergence_threshold=config.divergence_threshold,
                camera_height=config.camera_height,
                camera_width=config.camera_width,
            )
        )
        validation = validate_replay_output(
            source_hdf5_path=task.source_demo_path,
            rebuilt_hdf5_path=target_path,
            required_camera_names=config.camera_names,
            atol=config.value_check_atol,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return TaskReplayResult(
            task=task,
            source_path=task.source_demo_path,
            target_path=target_path,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return TaskReplayResult(
        task=task,
        source_path=task.source_demo_path,
        target_path=target_path,
        summary=summary,
        validation=validation,
    )


def run_replay(config: ReplayConfig) -> None:
    """Runs the replay pipeline.

    Args:
        config: Replay configuration.

    Raises:
        FileNotFoundError: If no source datasets match the filters.
        RuntimeError: If one or more files fail replay or validation.
        FileExistsError: If an output file already exists and overwrite is off.
    """

    install_replay_camera_info_patch()
    robosuite_root, libero_assets_root, legacy_markers = install_model_xml_remapper(
        libero_assets_root=None,
        legacy_asset_markers=config.legacy_asset_markers,
        operation_config=config.operation_camera_config,
        trajectory_config=config.trajectory_camera_config,
    )

    available_tasks, missing_tasks = discover_task_specs(
        source_root=config.source_root,
        benchmark_names=config.benchmarks,
        task_filter=config.tasks,
    )

    reporter = ConsoleReporter()
    report_run_context(
        reporter=reporter,
        config=config,
        robosuite_root=robosuite_root,
        libero_assets_root=libero_assets_root,
        legacy_markers=legacy_markers,
        available_tasks=available_tasks,
        missing_tasks=missing_tasks,
    )
    report_missing_tasks(reporter, missing_tasks)

    if not available_tasks:
        raise FileNotFoundError("No source HDF5 found for selected benchmark or task filters")

    if config.dry_run:
        report_dry_run_tasks(reporter, config, available_tasks)
        return

    stats = ReplayStats()
    with tqdm(
        total=len(available_tasks),
        desc=DEFAULT_PROGRESS_DESC,
        unit="task",
        dynamic_ncols=True,
    ) as progress_bar:
        reporter.progress_bar = progress_bar
        for task in available_tasks:
            progress_bar.set_description(
                f"{DEFAULT_PROGRESS_DESC} [{task.benchmark_name}]"
            )
            progress_bar.set_postfix_str(
                truncate_progress_text(task.task_name),
                refresh=False,
            )
            reporter.write(f"[replay] {task.label}")

            result = replay_task(task, config)
            if result.error_message is not None:
                stats.record_failure()
                report_task_failure(reporter, result)
            elif not result.validation.ok:
                stats.record_failure()
                report_validation_failure(reporter, result)
            else:
                stats.record_success(result.summary.total_samples)
                report_task_success(reporter, result)

            progress_bar.update(1)

    reporter.progress_bar = None
    report_run_summary(reporter, stats)

    if stats.failed > 0:
        raise RuntimeError(f"Reconstruction finished with {stats.failed} failed files")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parses arguments and runs the replay pipeline.

    Args:
        argv: Optional CLI argument list.
    """

    args = parse_args(argv)
    config = build_replay_config(args)
    run_replay(config)


if __name__ == "__main__":
    main()
