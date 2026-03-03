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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
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
DEFAULT_CAMERA_HEIGHT = 128
DEFAULT_CAMERA_WIDTH = 128
DEFAULT_NO_PROPRIO = False
DEFAULT_STATE_ERROR_THRESHOLD = 0.01
DEFAULT_VALUE_CHECK_ATOL = 1e-6
DEFAULT_TRAJECTORY_CAMERA_COUNT = 10
DEFAULT_TRAJECTORY_INTERP_SAMPLES = 300


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


def _parse_camera_offset_file(offset_file):
    with open(offset_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) != 3:
        raise ValueError(
            f"Offset file must contain exactly 3 non-empty lines, got {len(lines)}: {offset_file}"
        )

    sequences = []
    for line in lines:
        try:
            values = [float(x) for x in line.split()]
        except ValueError as exc:
            raise ValueError(f"Offset file contains non-numeric value: {offset_file}") from exc
        sequences.append(values)

    lengths = [len(seq) for seq in sequences]
    for i, length in enumerate(lengths):
        if length < 2 or length > 25:
            raise ValueError(
                f"Offset line {i + 1} length must be in [2, 25], got {length}"
            )

    for i, seq in enumerate(sequences):
        if abs(seq[0]) > 1e-9:
            raise ValueError(
                f"Offset line {i + 1} must start with 0, got {seq[0]} in {offset_file}"
            )

    d_phi = np.asarray(sequences[0], dtype=np.float64)
    d_theta = np.asarray(sequences[1], dtype=np.float64)
    d_r = np.asarray(sequences[2], dtype=np.float64)
    return d_phi, d_theta, d_r


def _trajectory_camera_names(base_camera_name, num_cameras):
    return [f"{base_camera_name}_traj_{idx:02d}" for idx in range(num_cameras)]


def _parse_vector_attr(attr_value, dim, attr_name):
    if attr_value is None:
        raise ValueError(f"Missing camera attribute '{attr_name}' in XML")
    values = np.asarray([float(x) for x in attr_value.split()], dtype=np.float64)
    if values.shape[0] != dim:
        raise ValueError(
            f"Camera attribute '{attr_name}' must have {dim} numbers, got {values.shape[0]}"
        )
    return values


def _interpolate_offset_sequence(values, target_count):
    if len(values) == target_count:
        return np.asarray(values, dtype=np.float64)

    key_t = np.linspace(0.0, 1.0, len(values))
    dense_t = np.linspace(0.0, 1.0, target_count)
    spline = CubicSpline(key_t, values, bc_type="natural")
    return np.asarray(spline(dense_t), dtype=np.float64)


def _uniform_sample_positions(positions, num_samples):
    if positions.shape[0] == num_samples:
        return positions

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return np.repeat(positions[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total_length, num_samples)
    sampled = np.stack(
        [np.interp(targets, cumulative, positions[:, axis]) for axis in range(3)],
        axis=1,
    )
    return sampled


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm


def _lookat_quat_wxyz(camera_pos, target_pos):
    forward = _normalize(target_pos - camera_pos)
    z_axis = -forward

    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) <= 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up_hint, z_axis)

    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rot = np.column_stack([x_axis, y_axis, z_axis])

    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _camera_center_from_pose(base_pos, base_quat_wxyz):
    quat_xyzw = np.array(
        [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]],
        dtype=np.float64,
    )
    base_forward = Rotation.from_quat(quat_xyzw).apply(np.array([0.0, 0.0, -1.0]))
    base_radius = float(np.linalg.norm(base_pos))
    if base_radius <= 1e-12:
        base_radius = 1.0
    return base_pos + base_forward * base_radius


def _generate_trajectory_camera_specs(
    base_pos,
    base_quat_wxyz,
    d_phi,
    d_theta,
    d_r,
    num_cameras,
    interp_samples,
    camera_names,
    camera_fovy=None,
):
    center = _camera_center_from_pose(base_pos, base_quat_wxyz)
    rel0 = base_pos - center
    r0 = float(np.linalg.norm(rel0))
    if r0 <= 1e-12:
        raise ValueError("Base camera radius is too small to build spherical trajectory")

    phi0 = float(np.arctan2(rel0[1], rel0[0]))
    theta0 = float(np.arctan2(rel0[2], np.hypot(rel0[0], rel0[1])))

    interp_count = max(int(interp_samples), len(d_phi))
    d_phi_i = _interpolate_offset_sequence(d_phi, interp_count)
    d_theta_i = _interpolate_offset_sequence(d_theta, interp_count)
    d_r_i = _interpolate_offset_sequence(d_r, interp_count)

    phi = phi0 + np.deg2rad(d_phi_i)
    theta = theta0 + np.deg2rad(d_theta_i)
    radius = np.maximum(r0 + d_r_i, 0.05)

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.sin(theta)
    dense_positions = center + np.stack([x, y, z], axis=1)

    sampled_positions = _uniform_sample_positions(dense_positions, num_cameras)

    specs = []
    for name, pos in zip(camera_names, sampled_positions):
        quat = _lookat_quat_wxyz(pos, center)
        spec = {
            "name": name,
            "pos": pos,
            "quat": quat,
            "mode": "fixed",
        }
        if camera_fovy is not None:
            spec["fovy"] = camera_fovy
        specs.append(spec)
    return specs


def _inject_trajectory_cameras(xml_str, trajectory_config):
    if trajectory_config is None:
        return xml_str

    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML missing <worldbody>, cannot inject trajectory cameras")

    base_name = trajectory_config["base_camera_name"]
    base_camera_elem = None
    for cam in root.iter("camera"):
        if cam.get("name") == base_name:
            base_camera_elem = cam
            break
    if base_camera_elem is None:
        raise ValueError(f"Base camera '{base_name}' not found in XML")

    base_pos = _parse_vector_attr(base_camera_elem.get("pos"), 3, "pos")
    base_quat = _parse_vector_attr(base_camera_elem.get("quat", "1 0 0 0"), 4, "quat")
    camera_fovy = base_camera_elem.get("fovy")

    specs = _generate_trajectory_camera_specs(
        base_pos=base_pos,
        base_quat_wxyz=base_quat,
        d_phi=trajectory_config["d_phi"],
        d_theta=trajectory_config["d_theta"],
        d_r=trajectory_config["d_r"],
        num_cameras=trajectory_config["num_cameras"],
        interp_samples=trajectory_config["interp_samples"],
        camera_names=trajectory_config["camera_names"],
        camera_fovy=camera_fovy,
    )

    for spec in specs:
        attrs = {
            "name": spec["name"],
            "mode": spec["mode"],
            "pos": " ".join(f"{v:.6f}" for v in spec["pos"]),
            "quat": " ".join(f"{v:.6f}" for v in spec["quat"]),
        }
        if "fovy" in spec and spec["fovy"] is not None:
            attrs["fovy"] = spec["fovy"]
        ET.SubElement(worldbody, "camera", attrs)

    return ET.tostring(root, encoding="utf8").decode("utf8")


def install_model_xml_remapper(
    libero_assets_root=None,
    legacy_asset_markers=None,
    trajectory_config=None,
):
    legacy_asset_markers = tuple(legacy_asset_markers or DEFAULT_LEGACY_ASSET_MARKERS)
    robosuite_root = _resolve_robosuite_root()
    libero_assets_root = _resolve_libero_assets_root(libero_assets_root)
    original_postprocess = replay_utils.libero_utils.postprocess_model_xml

    def patched_postprocess(xml_str, cameras_dict=None):
        if cameras_dict is None:
            cameras_dict = {}
        xml = original_postprocess(xml_str, cameras_dict)
        xml = _rewrite_model_xml_paths(
            xml_str=xml,
            robosuite_root=robosuite_root,
            libero_assets_root=libero_assets_root,
            legacy_markers=legacy_asset_markers,
        )
        return _inject_trajectory_cameras(xml, trajectory_config)

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
        "--camera-names",
        nargs="+",
        default=None,
        help=(
            "Camera names used during replay rendering. "
            "Default follows source env_args camera_names."
        ),
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
        "--camera-offset-file",
        type=str,
        default=None,
        help=(
            "Optional txt file with 3 offset lines (d_phi, d_theta, d_r). "
            "When provided, trajectory cameras are generated from this file."
        ),
    )
    parser.add_argument(
        "--camera-base-name",
        type=str,
        default="agentview",
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
    if args.trajectory_camera_count < 1:
        raise ValueError("--trajectory-camera-count must be >= 1")
    if args.trajectory_interp_samples < 2:
        raise ValueError("--trajectory-interp-samples must be >= 2")

    camera_names = args.camera_names if args.camera_names else None
    trajectory_config = None
    trajectory_camera_names = []

    if args.camera_offset_file:
        offset_file = os.path.abspath(os.path.expanduser(args.camera_offset_file))
        d_phi, d_theta, d_r = _parse_camera_offset_file(offset_file)
        trajectory_camera_names = _trajectory_camera_names(
            args.camera_base_name,
            args.trajectory_camera_count,
        )
        trajectory_config = {
            "base_camera_name": args.camera_base_name,
            "d_phi": d_phi,
            "d_theta": d_theta,
            "d_r": d_r,
            "num_cameras": args.trajectory_camera_count,
            "interp_samples": args.trajectory_interp_samples,
            "camera_names": trajectory_camera_names,
        }

        if camera_names is None:
            camera_names = list(trajectory_camera_names)
        else:
            # Keep existing requested cameras while appending generated trajectory cameras.
            camera_names = list(camera_names) + list(trajectory_camera_names)

    robosuite_root, libero_assets_root, legacy_markers = install_model_xml_remapper(
        libero_assets_root=None,
        legacy_asset_markers=DEFAULT_LEGACY_ASSET_MARKERS,
        trajectory_config=trajectory_config,
    )

    source_root = os.path.abspath(
        os.path.expanduser(args.source_root or get_libero_path("datasets"))
    )
    output_root = os.path.abspath(
        os.path.expanduser(args.output_root or f"{source_root.rstrip(os.sep)}_replay")
    )

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
    print(
        "[info] replay settings: "
        f"camera_names={camera_names if camera_names is not None else 'source_default'}, "
        f"camera_height={args.camera_height}, "
        f"camera_width={args.camera_width}, "
        f"no_proprio={DEFAULT_NO_PROPRIO}"
    )
    if trajectory_config is not None:
        print(f"[info] trajectory offset file: {os.path.abspath(os.path.expanduser(args.camera_offset_file))}")
        print(f"[info] trajectory base camera: {args.camera_base_name}")
        print(f"[info] trajectory camera count: {args.trajectory_camera_count}")
        print(f"[info] trajectory interpolation samples: {args.trajectory_interp_samples}")
        print(f"[info] generated trajectory cameras: {trajectory_camera_names}")

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
    failed = 0
    total_samples = 0

    for task_info in available_tasks:
        src_path = task_info["source_demo_path"]
        dst_path = os.path.join(output_root, task_info["relative_demo_path"])
        dst_file = Path(dst_path)
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists():
            if args.overwrite:
                dst_file.unlink()
            else:
                raise FileExistsError(
                    f"Target file exists: {dst_path}. Use --overwrite."
                )

        print(f"[replay] {format_task(task_info)}")
        try:
            summary = reconstruct_dataset_file(
                source_hdf5_path=src_path,
                output_hdf5_path=dst_path,
                camera_names=camera_names,
                no_proprio=DEFAULT_NO_PROPRIO,
                divergence_threshold=DEFAULT_STATE_ERROR_THRESHOLD,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
            )
            validation = validate_reconstructed_file(src_path, dst_path)
            if trajectory_config is not None and validation["errors"]:
                ignored_obs_errors = []
                kept_errors = []
                for error in validation["errors"]:
                    if "/obs missing keys in rebuilt" in error or "/obs/" in error:
                        ignored_obs_errors.append(error)
                    else:
                        kept_errors.append(error)
                if ignored_obs_errors:
                    print(
                        "[warning] ignored source-vs-rebuilt obs-key mismatch errors "
                        "because trajectory cameras intentionally change observation keys."
                    )
                validation["errors"] = kept_errors
                validation["ok"] = len(kept_errors) == 0
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
            atol=DEFAULT_VALUE_CHECK_ATOL,
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
        "[done] processed={processed}, failed={failed}, "
        "total_transitions={total_samples}".format(
            processed=processed,
            failed=failed,
            total_samples=total_samples,
        )
    )
    print("[note] step3/step4 are deferred in this script.")

    if failed > 0:
        raise RuntimeError(f"Reconstruction finished with {failed} failed files")


if __name__ == "__main__":
    main()
