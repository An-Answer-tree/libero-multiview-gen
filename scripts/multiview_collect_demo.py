import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import init_path
import numpy as np
import robosuite
from libero.libero import get_libero_path
import replay_dataset_utils as replay_utils
from replay_dataset_utils import (
    DEFAULT_BENCHMARKS,
    discover_benchmark_tasks,
    reconstruct_dataset_file,
    validate_reconstructed_file,
)

DEFAULT_LEGACY_ASSET_MARKERS = (
    "chiliocosm/assets/",
    "libero/libero/assets/",
    "libero/assets/",
)


def _sorted_demo_keys(data_group):
    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def _resolve_robosuite_root():
    candidates = []
    package_file = getattr(robosuite, "__file__", None)
    if package_file:
        candidates.append(Path(package_file).resolve().parent)
    for package_path in list(getattr(robosuite, "__path__", [])):
        candidates.append(Path(package_path).resolve())

    dedup = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(candidate)

    for candidate in dedup:
        if (candidate / "models").exists():
            return candidate
    return dedup[0] if dedup else None


def _resolve_libero_assets_root(user_path=None):
    if user_path:
        return Path(os.path.abspath(os.path.expanduser(user_path)))
    try:
        configured_path = get_libero_path("assets")
        return Path(os.path.abspath(os.path.expanduser(configured_path)))
    except Exception:
        return Path(__file__).resolve().parents[1] / "libero" / "libero" / "assets"


def _iter_asset_elements(root):
    asset = root.find("asset")
    if asset is None:
        return []
    return asset.findall("mesh") + asset.findall("texture")


def _rewrite_model_xml_paths(xml_str, robosuite_root, libero_assets_root, legacy_markers):
    tree = ET.fromstring(xml_str)
    changed = False

    for elem in _iter_asset_elements(tree):
        old_path = elem.get("file")
        if old_path is None:
            continue

        normalized_path = old_path.replace("\\", "/")
        new_path = None

        if robosuite_root is not None and "robosuite/" in normalized_path:
            suffix = normalized_path.rsplit("robosuite/", 1)[1]
            candidate = robosuite_root / suffix
            if candidate.exists():
                new_path = str(candidate)

        if new_path is None:
            for marker in legacy_markers:
                marker = marker.rstrip("/") + "/"
                if marker not in normalized_path:
                    continue
                suffix = normalized_path.rsplit(marker, 1)[1]
                candidate = libero_assets_root / suffix
                if candidate.exists():
                    new_path = str(candidate)
                    break

        if new_path is not None and new_path != old_path:
            elem.set("file", new_path)
            changed = True

    if not changed:
        return xml_str
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def install_model_xml_remapper(libero_assets_root=None, legacy_asset_markers=None):
    legacy_asset_markers = tuple(legacy_asset_markers or DEFAULT_LEGACY_ASSET_MARKERS)
    robosuite_root = _resolve_robosuite_root()
    libero_assets_root = _resolve_libero_assets_root(libero_assets_root)
    original_postprocess = replay_utils.libero_utils.postprocess_model_xml

    def patched_postprocess(xml_str, cameras_dict=None):
        if cameras_dict is None:
            cameras_dict = {}
        xml = original_postprocess(xml_str, cameras_dict)
        return _rewrite_model_xml_paths(
            xml_str=xml,
            robosuite_root=robosuite_root,
            libero_assets_root=libero_assets_root,
            legacy_markers=legacy_asset_markers,
        )

    replay_utils.libero_utils.postprocess_model_xml = patched_postprocess
    return robosuite_root, libero_assets_root, legacy_asset_markers


def validate_action_state_consistency(source_hdf5_path, rebuilt_hdf5_path, atol=1e-6):
    errors = []

    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(rebuilt_hdf5_path, "r") as rebuilt_file:
        source_data = source_file["data"]
        rebuilt_data = rebuilt_file["data"]
        src_demo_keys = _sorted_demo_keys(source_data)

        for demo_key in src_demo_keys:
            if demo_key not in rebuilt_data:
                errors.append(f"Missing demo group in rebuilt: {demo_key}")
                continue

            src_ep = source_data[demo_key]
            dst_ep = rebuilt_data[demo_key]

            src_actions = np.array(src_ep["actions"][()])
            dst_actions = np.array(dst_ep["actions"][()])
            if src_actions.shape != dst_actions.shape:
                errors.append(
                    f"{demo_key}/actions shape mismatch {src_actions.shape} vs {dst_actions.shape}"
                )
            elif not np.allclose(src_actions, dst_actions, atol=atol, rtol=0.0):
                errors.append(f"{demo_key}/actions values mismatch (atol={atol})")

            src_states = np.array(src_ep["states"][()])
            dst_states = np.array(dst_ep["states"][()])
            if src_states.shape != dst_states.shape:
                errors.append(
                    f"{demo_key}/states shape mismatch {src_states.shape} vs {dst_states.shape}"
                )
            elif not np.allclose(src_states, dst_states, atol=atol, rtol=0.0):
                errors.append(f"{demo_key}/states values mismatch (atol={atol})")

            if "robot_states" in dst_ep and dst_ep["robot_states"].shape[0] != len(dst_actions):
                errors.append(
                    f"{demo_key}/robot_states length {dst_ep['robot_states'].shape[0]} != "
                    f"actions length {len(dst_actions)}"
                )

    return {"ok": len(errors) == 0, "errors": errors}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay LIBERO source datasets to reconstruct semantically equivalent hdf5 files. "
            "Step3/4 (extra fixed cameras) are intentionally deferred."
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
        help="Output dataset root, default '<source-root>_replay'.",
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
        "--skip-existing",
        action="store_true",
        help="Skip target hdf5 file if already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target hdf5 file.",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help=(
            "Camera names used during replay rendering. "
            "Default follows source env_args camera_names, fallback to robot0_eye_in_hand agentview."
        ),
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=128,
        help="Replay render height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=128,
        help="Replay render width.",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Record depth observations during replay (off by default).",
    )
    parser.add_argument(
        "--no-proprio",
        action="store_true",
        help="Disable proprioceptive observation writing.",
    )
    parser.add_argument(
        "--state-error-threshold",
        type=float,
        default=0.01,
        help="Warn threshold for replay-vs-source state l2 divergence.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved for future parallelism. Current version always runs single-worker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved source/target mapping without reconstruction.",
    )
    parser.add_argument(
        "--libero-assets-root",
        type=str,
        default=None,
        help=(
            "Override local LIBERO assets root for legacy XML path remapping. "
            "Default uses get_libero_path('assets')."
        ),
    )
    parser.add_argument(
        "--legacy-asset-markers",
        nargs="+",
        default=list(DEFAULT_LEGACY_ASSET_MARKERS),
        help=(
            "Legacy XML path markers to remap into --libero-assets-root "
            "(e.g. chiliocosm/assets/)."
        ),
    )
    parser.add_argument(
        "--value-check-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for action/state value consistency checks.",
    )
    parser.add_argument(
        "--max-diverged-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum allowed ratio of diverged replay steps in one file. "
            "Only takes effect as a hard failure when --enforce-divergence-check is set."
        ),
    )
    parser.add_argument(
        "--enforce-divergence-check",
        action="store_true",
        help="Treat files over --max-diverged-ratio as failed.",
    )
    return parser.parse_args()


def format_task(task_info):
    return (
        f"[{task_info['benchmark_name']}] {task_info['task_name']} "
        f"({task_info['relative_demo_path']})"
    )


def print_verify_commands(source_path, target_path):
    print("  verify commands:")
    print(f"    python scripts/get_dataset_info.py --dataset {source_path}")
    print(f"    python scripts/get_dataset_info.py --dataset {target_path}")


def main():
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite cannot be enabled together")
    if args.max_diverged_ratio < 0 or args.max_diverged_ratio > 1:
        raise ValueError("--max-diverged-ratio must be in [0, 1]")

    if args.num_workers != 1:
        print("[warning] --num-workers is reserved; running with a single worker.")

    robosuite_root, libero_assets_root, legacy_markers = install_model_xml_remapper(
        libero_assets_root=args.libero_assets_root,
        legacy_asset_markers=args.legacy_asset_markers,
    )

    source_root = os.path.abspath(
        os.path.expanduser(args.source_root or get_libero_path("datasets"))
    )
    output_root = os.path.abspath(
        os.path.expanduser(args.output_root or f"{source_root.rstrip(os.sep)}_replay")
    )
    camera_names = args.camera_names if args.camera_names else None

    available_tasks, missing_tasks = discover_benchmark_tasks(
        source_root=source_root,
        benchmark_names=args.benchmarks,
        task_filter=args.tasks,
    )

    print(f"[info] source-root: {source_root}")
    print(f"[info] output-root: {output_root}")
    print(f"[info] selected benchmarks: {args.benchmarks}")
    print(f"[info] selected tasks: {'all' if args.tasks is None else len(args.tasks)}")
    print(f"[info] available source files: {len(available_tasks)}")
    print(f"[info] missing source files: {len(missing_tasks)}")
    print(f"[info] robosuite root for replay xml: {robosuite_root}")
    print(f"[info] libero assets root for replay xml: {libero_assets_root}")
    print(f"[info] legacy asset markers: {list(legacy_markers)}")

    for missing in missing_tasks:
        print(f"[warning] missing source demo: {missing['source_demo_path']}")

    if len(available_tasks) == 0:
        raise FileNotFoundError("No source hdf5 found for selected benchmark/task filters")

    if args.dry_run:
        for task_info in available_tasks:
            src = task_info["source_demo_path"]
            dst = os.path.join(output_root, task_info["relative_demo_path"])
            print(f"[dry-run] {format_task(task_info)}")
            print(f"          src={src}")
            print(f"          dst={dst}")
        return

    processed = 0
    skipped = 0
    failed = 0
    total_samples = 0

    for task_info in available_tasks:
        src_path = task_info["source_demo_path"]
        dst_path = os.path.join(output_root, task_info["relative_demo_path"])
        dst_file = Path(dst_path)
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists():
            if args.skip_existing:
                skipped += 1
                print(f"[skip] {dst_path}")
                continue
            if args.overwrite:
                dst_file.unlink()
            else:
                raise FileExistsError(
                    f"Target file exists: {dst_path}. Use --skip-existing or --overwrite."
                )

        print(f"[replay] {format_task(task_info)}")
        try:
            summary = reconstruct_dataset_file(
                source_hdf5_path=src_path,
                output_hdf5_path=dst_path,
                camera_names=camera_names,
                use_depth=args.use_depth,
                no_proprio=args.no_proprio,
                divergence_threshold=args.state_error_threshold,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
            )
            validation = validate_reconstructed_file(src_path, dst_path)
        except Exception as exc:
            failed += 1
            print(f"[error] replay failed: {exc}")
            continue

        if not validation["ok"]:
            failed += 1
            print(f"[error] validation failed: {dst_path}")
            for error in validation["errors"]:
                print(f"  - {error}")
            print_verify_commands(src_path, dst_path)
            continue

        consistency = validate_action_state_consistency(
            source_hdf5_path=src_path,
            rebuilt_hdf5_path=dst_path,
            atol=args.value_check_atol,
        )
        if not consistency["ok"]:
            failed += 1
            print(f"[error] action/state consistency failed: {dst_path}")
            for error in consistency["errors"]:
                print(f"  - {error}")
            print_verify_commands(src_path, dst_path)
            continue

        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"[warning] {warning}")

        diverged_steps = sum(ep["num_diverged_steps"] for ep in summary["episodes"])
        diverged_ratio = (
            float(diverged_steps) / float(summary["total_samples"])
            if summary["total_samples"] > 0
            else 0.0
        )
        if diverged_ratio > args.max_diverged_ratio:
            level = "[error]" if args.enforce_divergence_check else "[warning]"
            print(
                f"{level} replay divergence high: "
                f"ratio={diverged_ratio:.4f} > threshold={args.max_diverged_ratio:.4f}"
            )
            if args.enforce_divergence_check:
                failed += 1
                print_verify_commands(src_path, dst_path)
                continue

        processed += 1
        total_samples += summary["total_samples"]
        max_err = (
            max(ep["max_state_error"] for ep in summary["episodes"])
            if summary["episodes"]
            else 0.0
        )
        diverged = sum(
            ep["num_diverged_steps"] > 0 for ep in summary["episodes"]
        )
        print(
            "[ok] demos={num_demos}, transitions={num_samples}, max_state_error={max_err:.6f}, "
            "diverged_episodes={diverged}, diverged_ratio={diverged_ratio:.4f}".format(
                num_demos=summary["num_demos"],
                num_samples=summary["total_samples"],
                max_err=max_err,
                diverged=diverged,
                diverged_ratio=diverged_ratio,
            )
        )
        print_verify_commands(src_path, dst_path)

    print("========================================")
    print(
        "[done] processed={processed}, skipped={skipped}, failed={failed}, "
        "total_transitions={total_samples}".format(
            processed=processed,
            skipped=skipped,
            failed=failed,
            total_samples=total_samples,
        )
    )
    print("[note] step3/step4 are deferred in this script.")

    if failed > 0:
        raise RuntimeError(f"Reconstruction finished with {failed} failed files")


if __name__ == "__main__":
    main()
