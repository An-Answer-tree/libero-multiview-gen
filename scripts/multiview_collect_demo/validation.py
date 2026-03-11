"""Validation helpers for multiview replay dataset reconstruction."""

from __future__ import annotations

from typing import Any, Sequence

import h5py
import numpy as np


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    """Returns episode keys sorted by the numeric suffix in ``demo_<id>``."""

    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


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
            elif not np.allclose(src_states, dst_states, atol=atol, rtol=0.0):
                errors.append(f"{demo_key}/states values mismatch (atol={atol})")

            if "robot_states" in dst_episode and dst_episode["robot_states"].shape[0] != len(
                dst_actions
            ):
                errors.append(
                    f"{demo_key}/robot_states length {dst_episode['robot_states'].shape[0]} != "
                    f"actions length {len(dst_actions)}"
                )

    return {"ok": len(errors) == 0, "errors": errors}


def split_generated_camera_validation_errors(
    errors: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Splits replay validation errors into ignored and retained groups."""

    ignored_errors = []
    kept_errors = []
    for error in errors:
        if "/obs missing keys in rebuilt" in error or "/obs/" in error:
            ignored_errors.append(error)
        else:
            kept_errors.append(error)
    return ignored_errors, kept_errors
