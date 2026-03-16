"""Replay pipeline runner for multiview dataset reconstruction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import init_path  # noqa: F401
from replay_dataset_utils import (
    discover_benchmark_tasks,
    reconstruct_dataset_file,
    validate_reconstructed_file,
)

from .camera_injection import install_model_xml_remapper
from .config import ReplayConfig
from .validation import (
    split_generated_camera_validation_errors,
    validate_action_state_consistency,
)


@dataclass
class ReplayStats:
    """Aggregate replay progress counters."""

    processed: int = 0
    failed: int = 0
    total_samples: int = 0


def _format_task(task_info: dict[str, Any]) -> str:
    """Formats a benchmark task description for console logging."""

    return (
        f"[{task_info['benchmark_name']}] {task_info['task_name']} "
        f"({task_info['relative_demo_path']})"
    )


def _print_verify_commands(source_path: str, target_path: str) -> None:
    """Prints follow-up verification commands for a replayed dataset."""

    print("  verify commands:")
    print(f"    python scripts/get_dataset_info.py --dataset {source_path}")
    print(f"    python scripts/get_dataset_info.py --dataset {target_path}")


def _print_header(
    config: ReplayConfig,
    robosuite_root: Path | None,
    libero_assets_root: Path,
    legacy_markers: tuple[str, ...],
    available_tasks: list[dict[str, Any]],
    missing_tasks: list[dict[str, Any]],
) -> None:
    """Prints the top-level replay configuration summary."""

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
        f"camera_width={config.camera_width}, "
        f"no_proprio={config.no_proprio}"
    )
    if config.operation_camera_config is not None:
        print(
            f"[info] operation camera base: {config.operation_camera_config.base_camera_name}"
        )
        print(f"[info] generated operation cameras: {config.operation_camera_names}")
    if config.trajectory_camera_config is not None:
        print(
            f"[info] trajectory offset file: {config.trajectory_camera_config.offset_file}"
        )
        print(
            f"[info] trajectory base camera: {config.trajectory_camera_config.base_camera_name}"
        )
        print(
            f"[info] trajectory camera count: {config.trajectory_camera_config.num_cameras}"
        )
        print(
            "[info] trajectory interpolation samples: "
            f"{config.trajectory_camera_config.interp_samples}"
        )
        print(f"[info] generated trajectory cameras: {config.trajectory_camera_names}")


def _maybe_filter_validation_errors(
    validation: dict[str, Any],
    has_generated_cameras: bool,
) -> dict[str, Any]:
    """Filters expected observation-key validation errors from generated cameras."""

    if not has_generated_cameras or not validation["errors"]:
        return validation

    ignored_errors, kept_errors = split_generated_camera_validation_errors(
        validation["errors"]
    )
    if ignored_errors:
        print(
            "[warning] ignored source-vs-rebuilt obs-key mismatch errors "
            "because generated cameras intentionally change observation keys."
        )

    filtered_validation = dict(validation)
    filtered_validation["errors"] = kept_errors
    filtered_validation["ok"] = len(kept_errors) == 0
    return filtered_validation


def _handle_successful_replay(
    src_path: str,
    dst_path: str,
    summary: dict[str, Any],
    stats: ReplayStats,
) -> None:
    """Updates aggregate counters and prints the replay summary."""

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
    restored_from_source_states = summary["total_samples"]
    print(
        "[ok] demos={num_demos}, transitions={num_samples}, "
        "rendered_from_source_states={rendered}, max_restore_error={max_err:.6f}, "
        "restore_mismatches={mismatches}".format(
            num_demos=summary["num_demos"],
            num_samples=summary["total_samples"],
            rendered=restored_from_source_states,
            max_err=max_restore_error,
            mismatches=restore_mismatch_steps,
        )
    )
    _print_verify_commands(src_path, dst_path)


def run_replay(config: ReplayConfig) -> None:
    """Runs the dataset replay pipeline using the resolved configuration."""

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
    _print_header(
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
        raise FileNotFoundError("No source hdf5 found for selected benchmark/task filters")

    if config.dry_run:
        for task_info in available_tasks:
            src_path = task_info["source_demo_path"]
            dst_path = os.path.join(config.output_root, task_info["relative_demo_path"])
            print(f"[dry-run] {_format_task(task_info)}")
            print(f"          src={src_path}")
            print(f"          dst={dst_path}")
        return

    stats = ReplayStats()
    for task_info in available_tasks:
        src_path = task_info["source_demo_path"]
        dst_path = os.path.join(config.output_root, task_info["relative_demo_path"])
        dst_file = Path(dst_path)
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists():
            if config.overwrite:
                dst_file.unlink()
            else:
                raise FileExistsError(f"Target file exists: {dst_path}. Use --overwrite.")

        print(f"[replay] {_format_task(task_info)}")
        try:
            summary = reconstruct_dataset_file(
                source_hdf5_path=src_path,
                output_hdf5_path=dst_path,
                camera_names=config.camera_names,
                no_proprio=config.no_proprio,
                divergence_threshold=config.divergence_threshold,
                camera_height=config.camera_height,
                camera_width=config.camera_width,
            )
            validation = validate_reconstructed_file(src_path, dst_path)
            validation = _maybe_filter_validation_errors(
                validation=validation,
                has_generated_cameras=config.has_generated_cameras,
            )
        except Exception as exc:
            stats.failed += 1
            print(f"[error] replay failed: {exc}")
            continue

        if not validation["ok"]:
            stats.failed += 1
            print(f"[error] validation failed: {dst_path}")
            for error in validation["errors"]:
                print(f"  - {error}")
            _print_verify_commands(src_path, dst_path)
            continue

        consistency = validate_action_state_consistency(
            source_hdf5_path=src_path,
            rebuilt_hdf5_path=dst_path,
            atol=config.value_check_atol,
        )
        if not consistency["ok"]:
            stats.failed += 1
            print(f"[error] action/state consistency failed: {dst_path}")
            for error in consistency["errors"]:
                print(f"  - {error}")
            _print_verify_commands(src_path, dst_path)
            continue

        for warning in validation["warnings"]:
            print(f"[warning] {warning}")

        _handle_successful_replay(
            src_path=src_path,
            dst_path=dst_path,
            summary=summary,
            stats=stats,
        )

    print("========================================")
    print(
        "[done] processed={processed}, failed={failed}, "
        "total_transitions={total_samples}".format(
            processed=stats.processed,
            failed=stats.failed,
            total_samples=stats.total_samples,
        )
    )
    print("[note] step3/step4 are deferred in this script.")

    if stats.failed > 0:
        raise RuntimeError(f"Reconstruction finished with {stats.failed} failed files")
