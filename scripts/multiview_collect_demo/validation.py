"""Validation helpers for multiview replay dataset reconstruction."""

from __future__ import annotations

from typing import Any, Sequence

import h5py
import numpy as np


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    """Returns episode keys sorted by the numeric suffix in ``demo_<id>``."""

    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def is_rgb_obs_key(obs_key: str) -> bool:
    """Returns whether an observation key stores RGB frames."""

    return obs_key.endswith("_rgb")


def has_valid_rgb_shape(shape: tuple[int, ...]) -> bool:
    """Checks that an RGB tensor follows ``(T, H, W, 3)``."""

    return len(shape) == 4 and shape[-1] == 3 and shape[1] > 0 and shape[2] > 0


def arrays_match(src_array: np.ndarray, dst_array: np.ndarray, atol: float) -> bool:
    """Checks numeric equality with a strict absolute tolerance."""

    return np.allclose(src_array, dst_array, atol=atol, rtol=0.0)


def validate_action_state_consistency(
    source_hdf5_path: str,
    rebuilt_hdf5_path: str,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Checks that replay preserved source actions and simulator states."""

    errors = []

    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(
        rebuilt_hdf5_path, "r"
    ) as rebuilt_file:
        source_data = source_file["data"]
        rebuilt_data = rebuilt_file["data"]

        for demo_key in sorted_demo_keys(source_data):
            if demo_key not in rebuilt_data:
                errors.append(f"Missing demo group in rebuilt: {demo_key}")
                continue

            src_episode = source_data[demo_key]
            dst_episode = rebuilt_data[demo_key]

            src_actions = np.asarray(src_episode["actions"][()])
            dst_actions = np.asarray(dst_episode["actions"][()])
            if src_actions.shape != dst_actions.shape:
                errors.append(
                    f"{demo_key}/actions shape mismatch {src_actions.shape} vs {dst_actions.shape}"
                )
            elif not np.allclose(src_actions, dst_actions, atol=atol, rtol=0.0):
                errors.append(f"{demo_key}/actions values mismatch (atol={atol})")

            src_states = np.asarray(src_episode["states"][()])
            dst_states = np.asarray(dst_episode["states"][()])
            if src_states.shape != dst_states.shape:
                errors.append(
                    f"{demo_key}/states shape mismatch {src_states.shape} vs {dst_states.shape}"
                )
            elif not arrays_match(src_states, dst_states, atol=atol):
                errors.append(f"{demo_key}/states values mismatch (atol={atol})")

            src_rewards = np.asarray(src_episode["rewards"][()])
            dst_rewards = np.asarray(dst_episode["rewards"][()])
            if src_rewards.shape != dst_rewards.shape:
                errors.append(
                    f"{demo_key}/rewards shape mismatch {src_rewards.shape} vs {dst_rewards.shape}"
                )
            elif not arrays_match(src_rewards, dst_rewards, atol=atol):
                errors.append(f"{demo_key}/rewards values mismatch (atol={atol})")

            src_dones = np.asarray(src_episode["dones"][()])
            dst_dones = np.asarray(dst_episode["dones"][()])
            if src_dones.shape != dst_dones.shape:
                errors.append(
                    f"{demo_key}/dones shape mismatch {src_dones.shape} vs {dst_dones.shape}"
                )
            elif not arrays_match(src_dones, dst_dones, atol=atol):
                errors.append(f"{demo_key}/dones values mismatch (atol={atol})")

            src_robot_states = np.asarray(src_episode["robot_states"][()])
            dst_robot_states = np.asarray(dst_episode["robot_states"][()])
            if src_robot_states.shape != dst_robot_states.shape:
                errors.append(
                    f"{demo_key}/robot_states shape mismatch "
                    f"{src_robot_states.shape} vs {dst_robot_states.shape}"
                )
            elif not arrays_match(src_robot_states, dst_robot_states, atol=atol):
                errors.append(f"{demo_key}/robot_states values mismatch (atol={atol})")

            src_obs = src_episode["obs"]
            dst_obs = dst_episode["obs"]
            for obs_key in sorted(src_obs.keys()):
                if obs_key not in dst_obs:
                    errors.append(f"{demo_key}/obs missing key in rebuilt: {obs_key}")
                    continue

                src_obs_array = np.asarray(src_obs[obs_key][()])
                dst_obs_array = np.asarray(dst_obs[obs_key][()])
                if is_rgb_obs_key(obs_key):
                    if src_obs_array.shape[0] != dst_obs_array.shape[0]:
                        errors.append(
                            f"{demo_key}/obs/{obs_key} frame count mismatch "
                            f"{src_obs_array.shape[0]} vs {dst_obs_array.shape[0]}"
                        )
                    if not has_valid_rgb_shape(dst_obs_array.shape):
                        errors.append(
                            f"{demo_key}/obs/{obs_key} invalid rebuilt rgb shape "
                            f"{dst_obs_array.shape}"
                        )
                    continue

                if src_obs_array.shape != dst_obs_array.shape:
                    errors.append(
                        f"{demo_key}/obs/{obs_key} shape mismatch "
                        f"{src_obs_array.shape} vs {dst_obs_array.shape}"
                    )
                elif not arrays_match(src_obs_array, dst_obs_array, atol=atol):
                    errors.append(f"{demo_key}/obs/{obs_key} values mismatch (atol={atol})")

    return {"ok": len(errors) == 0, "errors": errors}


def split_generated_camera_validation_errors(
    errors: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Splits replay validation errors into ignored and retained groups."""

    ignored_errors = []
    kept_errors = []
    for error in errors:
        if "/obs extra keys in rebuilt:" in error:
            ignored_errors.append(error)
        else:
            kept_errors.append(error)
    return ignored_errors, kept_errors
