"""Camera injection and XML remapping helpers for multiview replay."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET

import init_path  # noqa: F401
import numpy as np
import replay_dataset_utils as replay_utils
import robosuite
from libero.libero import get_libero_path
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from .config import (
    DEFAULT_LEGACY_ASSET_MARKERS,
    OperationCameraConfig,
    TrajectoryCameraConfig,
    dedupe_keep_order,
)

_ORIGINAL_POSTPROCESS_MODEL_XML = replay_utils.libero_utils.postprocess_model_xml


@dataclass(frozen=True)
class CameraSpec:
    """Description of a camera node to be injected into the MuJoCo XML."""

    name: str
    pos: np.ndarray
    quat: np.ndarray
    mode: str = "fixed"
    fovy: Optional[str] = None


def _resolve_robosuite_root() -> Optional[Path]:
    """Resolves the active robosuite package root."""

    candidates = []
    package_file = getattr(robosuite, "__file__", None)
    if package_file:
        candidates.append(Path(package_file).resolve().parent)
    for package_path in list(getattr(robosuite, "__path__", [])):
        candidates.append(Path(package_path).resolve())

    deduped_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if (candidate / "models").exists():
            return candidate
    return deduped_candidates[0] if deduped_candidates else None


def _resolve_libero_assets_root(user_path: Optional[str] = None) -> Path:
    """Resolves the LIBERO assets root used when repairing XML paths."""

    if user_path:
        return Path(os.path.abspath(os.path.expanduser(user_path)))
    try:
        configured_path = get_libero_path("assets")
        return Path(os.path.abspath(os.path.expanduser(configured_path)))
    except Exception:
        return Path(__file__).resolve().parents[1] / "libero" / "libero" / "assets"


def _iter_asset_elements(root: ET.Element) -> list[ET.Element]:
    """Returns mesh and texture nodes from a MuJoCo XML root."""

    asset = root.find("asset")
    if asset is None:
        return []
    return asset.findall("mesh") + asset.findall("texture")


def _rewrite_model_xml_paths(
    xml_str: str,
    robosuite_root: Optional[Path],
    libero_assets_root: Path,
    legacy_markers: Iterable[str],
) -> str:
    """Rewrites legacy asset paths in the serialized MuJoCo XML."""

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
                normalized_marker = marker.rstrip("/") + "/"
                if normalized_marker not in normalized_path:
                    continue
                suffix = normalized_path.rsplit(normalized_marker, 1)[1]
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


def _parse_vector_attr(attr_value: Optional[str], dim: int, attr_name: str) -> np.ndarray:
    """Parses a whitespace-separated XML attribute into a float vector."""

    if attr_value is None:
        raise ValueError(f"Missing camera attribute '{attr_name}' in XML")
    values = np.asarray([float(value) for value in attr_value.split()], dtype=np.float64)
    if values.shape[0] != dim:
        raise ValueError(
            f"Camera attribute '{attr_name}' must have {dim} numbers, got {values.shape[0]}"
        )
    return values


def _camera_pose_from_element(camera_elem: ET.Element) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Returns ``(pos, quat, fovy)`` from a camera XML element."""

    pos = _parse_vector_attr(camera_elem.get("pos"), 3, "pos")
    quat = _parse_vector_attr(camera_elem.get("quat", "1 0 0 0"), 4, "quat")
    return pos, quat, camera_elem.get("fovy")


def _interpolate_offset_sequence(values: np.ndarray, target_count: int) -> np.ndarray:
    """Interpolates an offset sequence to the desired sample count."""

    if len(values) == target_count:
        return np.asarray(values, dtype=np.float64)

    key_t = np.linspace(0.0, 1.0, len(values))
    dense_t = np.linspace(0.0, 1.0, target_count)
    spline = CubicSpline(key_t, values, bc_type="natural")
    return np.asarray(spline(dense_t), dtype=np.float64)


def _uniform_sample_positions(positions: np.ndarray, num_samples: int) -> np.ndarray:
    """Samples positions uniformly along a polyline."""

    if positions.shape[0] == num_samples:
        return positions

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return np.repeat(positions[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total_length, num_samples)
    return np.stack(
        [np.interp(targets, cumulative, positions[:, axis]) for axis in range(3)],
        axis=1,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizes a vector and raises if the vector length is zero."""

    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm


def _rotate_xy(vec: np.ndarray, degrees: float) -> np.ndarray:
    """Rotates a vector around the world z-axis in the x-y plane."""

    radians = np.deg2rad(degrees)
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)
    return np.array(
        [
            cos_v * vec[0] - sin_v * vec[1],
            sin_v * vec[0] + cos_v * vec[1],
            vec[2],
        ],
        dtype=np.float64,
    )


def _lookat_quat_wxyz(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Constructs a MuJoCo wxyz quaternion for a look-at camera pose."""

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


def _pitch_target_up(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    degrees: float,
) -> np.ndarray:
    """Rotates a camera target upward around the camera-local right axis."""

    direction = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        camera_pos, dtype=np.float64
    )
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-12:
        raise ValueError("Cannot pitch a zero-length camera direction")

    direction = direction / distance
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right_axis = np.cross(direction, up_hint)
    if np.linalg.norm(right_axis) <= 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_axis = np.cross(direction, up_hint)
    right_axis = _normalize(right_axis)

    pitched_direction = Rotation.from_rotvec(
        np.deg2rad(degrees) * right_axis
    ).apply(direction)
    return np.asarray(camera_pos, dtype=np.float64) + pitched_direction * distance


def _advance_along_view(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Moves a camera forward along its current view direction."""

    return np.asarray(camera_pos, dtype=np.float64) + _normalize(
        np.asarray(target_pos, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    ) * float(distance)


def _camera_center_from_pose(base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Approximates the look-at center from a camera world pose."""

    quat_xyzw = np.array(
        [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]],
        dtype=np.float64,
    )
    base_forward = Rotation.from_quat(quat_xyzw).apply(np.array([0.0, 0.0, -1.0]))
    base_radius = float(np.linalg.norm(base_pos))
    if base_radius <= 1e-12:
        base_radius = 1.0
    return base_pos + base_forward * base_radius


def _find_camera_element(
    camera_map: dict[str, ET.Element],
    candidate_names: Iterable[str],
) -> Optional[ET.Element]:
    """Finds the first existing camera element among candidate names."""

    for name in candidate_names:
        camera_elem = camera_map.get(name)
        if camera_elem is not None:
            return camera_elem
    return None


def _build_fixed_camera_spec(
    name: str,
    pos: np.ndarray,
    target_pos: np.ndarray,
    camera_fovy: Optional[str] = None,
) -> CameraSpec:
    """Builds a fixed camera spec that looks at ``target_pos``."""

    return CameraSpec(
        name=name,
        pos=np.asarray(pos, dtype=np.float64),
        quat=_lookat_quat_wxyz(np.asarray(pos, dtype=np.float64), target_pos),
        fovy=camera_fovy,
    )


def _append_camera_specs(
    worldbody: ET.Element,
    specs: Iterable[CameraSpec],
    existing_names: Optional[Iterable[str]] = None,
) -> None:
    """Appends new camera elements to the worldbody while deduplicating by name."""

    existing_name_set = set(existing_names or [])
    for spec in specs:
        if spec.name in existing_name_set:
            continue
        attrs = {
            "name": spec.name,
            "mode": spec.mode,
            "pos": " ".join(f"{value:.6f}" for value in spec.pos),
            "quat": " ".join(f"{value:.6f}" for value in spec.quat),
        }
        if spec.fovy is not None:
            attrs["fovy"] = spec.fovy
        ET.SubElement(worldbody, "camera", attrs)
        existing_name_set.add(spec.name)


def _generate_trajectory_camera_specs(
    config: TrajectoryCameraConfig,
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    camera_fovy: Optional[str] = None,
) -> list[CameraSpec]:
    """Generates trajectory camera poses from the configured offsets."""

    center = _camera_center_from_pose(base_pos, base_quat_wxyz)
    rel0 = base_pos - center
    radius0 = float(np.linalg.norm(rel0))
    if radius0 <= 1e-12:
        raise ValueError("Base camera radius is too small to build spherical trajectory")

    phi0 = float(np.arctan2(rel0[1], rel0[0]))
    theta0 = float(np.arctan2(rel0[2], np.hypot(rel0[0], rel0[1])))

    interp_count = max(int(config.interp_samples), len(config.d_phi))
    d_phi = _interpolate_offset_sequence(config.d_phi, interp_count)
    d_theta = _interpolate_offset_sequence(config.d_theta, interp_count)
    d_r = _interpolate_offset_sequence(config.d_r, interp_count)

    phi = phi0 + np.deg2rad(d_phi)
    theta = theta0 + np.deg2rad(d_theta)
    radius = np.maximum(radius0 + d_r, 0.05)

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.sin(theta)
    dense_positions = center + np.stack([x, y, z], axis=1)
    sampled_positions = _uniform_sample_positions(dense_positions, config.num_cameras)

    specs = []
    for name, pos in zip(config.camera_names, sampled_positions):
        specs.append(
            CameraSpec(
                name=name,
                pos=pos,
                quat=_lookat_quat_wxyz(pos, center),
                fovy=camera_fovy,
            )
        )
    return specs


def _generate_operation_camera_specs(
    root: ET.Element,
    config: OperationCameraConfig,
) -> list[CameraSpec]:
    """Generates the fixed operation cameras around the inferred scene center."""

    camera_map = {
        camera.get("name"): camera
        for camera in root.iter("camera")
        if camera.get("name") is not None
    }

    center_camera = _find_camera_element(
        camera_map,
        dedupe_keep_order(
            [
                "frontview",
                config.base_camera_name,
                "agentview",
                "birdview",
                "sideview",
            ]
        ),
    )
    if center_camera is None:
        raise ValueError(
            "Cannot build operation cameras because no reference camera was found in XML"
        )

    center_pos, center_quat, _ = _camera_pose_from_element(center_camera)
    center = _camera_center_from_pose(center_pos, center_quat)

    fovy_camera = _find_camera_element(
        camera_map,
        dedupe_keep_order(
            [
                "frontview",
                "birdview",
                "sideview",
                config.base_camera_name,
                "agentview",
            ]
        ),
    )
    camera_fovy = fovy_camera.get("fovy") if fovy_camera is not None else None

    front_camera = _find_camera_element(
        camera_map,
        dedupe_keep_order(
            [
                "frontview",
                config.base_camera_name,
                "agentview",
            ]
        ),
    )
    if front_camera is not None:
        front_pos, _, _ = _camera_pose_from_element(front_camera)
        front_rel = front_pos - center
    else:
        front_rel = np.array([1.0, 0.0, 1.2], dtype=np.float64)

    if np.linalg.norm(front_rel[:2]) <= 1e-8:
        front_rel = np.array(
            [max(float(np.linalg.norm(front_rel)), 0.8), 0.0, max(front_rel[2], 1.0)],
            dtype=np.float64,
        )
    horizontal_radius = max(np.linalg.norm(front_rel[:2]), 0.8)
    front_dir = _normalize(np.array([front_rel[0], front_rel[1], 0.0], dtype=np.float64))
    back_dir = -front_dir

    side_camera = _find_camera_element(camera_map, ["sideview"])
    # LIBERO's legacy `sideview` aligns with the view that users expect to see
    # as `rightview` in the generated multiview dataset. Use it as the right-side
    # reference and derive the left-side camera by mirroring it across the
    # front-view plane so variable names and exported camera names stay aligned.
    if side_camera is not None:
        side_pos, _, _ = _camera_pose_from_element(side_camera)
        right_rel = side_pos - center
    else:
        right_rel = _rotate_xy(front_rel, 90.0)

    top_camera = _find_camera_element(camera_map, ["birdview"])
    if top_camera is not None:
        top_pos, _, _ = _camera_pose_from_element(top_camera)
        top_rel = top_pos - center
    else:
        horizontal_radius = max(
            np.linalg.norm(front_rel[:2]),
            np.linalg.norm(right_rel[:2]),
            0.8,
        )
        top_rel = np.array(
            [
                -0.2 * horizontal_radius,
                0.0,
                max(abs(front_rel[2]) + horizontal_radius * 1.1, 1.8),
            ],
            dtype=np.float64,
        )

    left_rel = np.array([right_rel[0], -right_rel[1], right_rel[2]], dtype=np.float64)
    if np.linalg.norm(left_rel[:2]) <= 1e-8:
        left_rel = _rotate_xy(front_rel, -90.0)

    back_rel = _rotate_xy(front_rel, 180.0)

    # Apply small, task-agnostic view corrections for the generated operation cameras:
    # rotate the side cameras slightly further toward the front workspace, move the top
    # camera closer to the agent/front direction, and pull the back camera forward so it
    # stays outside walls in tighter scenes.
    side_target = center + front_dir * (0.32 * horizontal_radius)
    side_target[2] -= 0.08 * horizontal_radius
    top_pos = center + top_rel + front_dir * (0.36 * horizontal_radius)
    top_pos[2] -= 0.30 * horizontal_radius
    left_pos = center + left_rel
    left_pos[2] += 0.26 * horizontal_radius
    right_pos = center + right_rel
    right_pos[2] += 0.26 * horizontal_radius
    back_pos = center + back_rel - back_dir * (0.95 * horizontal_radius)
    back_pos[2] += 0.42 * horizontal_radius
    top_target = center.copy()
    top_target[2] -= 0.60 * horizontal_radius
    top_pos = top_pos + _normalize(top_target - top_pos) * (0.22 * horizontal_radius)
    left_target = _pitch_target_up(left_pos, side_target, 0.0)
    right_target = _pitch_target_up(right_pos, side_target, 0.0)
    left_pos = _advance_along_view(left_pos, left_target, 0.18 * horizontal_radius)
    right_pos = _advance_along_view(right_pos, right_target, 0.18 * horizontal_radius)
    back_target = center.copy()
    back_target[2] += 0.40 * horizontal_radius
    back_target = _pitch_target_up(back_pos, back_target, 15.0)

    return [
        _build_fixed_camera_spec(
            name=config.camera_names["top"],
            pos=top_pos,
            target_pos=top_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["left"],
            pos=left_pos,
            target_pos=left_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["right"],
            pos=right_pos,
            target_pos=right_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["back"],
            pos=back_pos,
            target_pos=back_target,
            camera_fovy=camera_fovy,
        ),
    ]


def _inject_operation_cameras(
    xml_str: str,
    config: Optional[OperationCameraConfig],
) -> str:
    """Injects the generated operation cameras into the XML string."""

    if config is None:
        return xml_str

    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML missing <worldbody>, cannot inject operation cameras")

    specs = _generate_operation_camera_specs(root, config)
    existing_names = {camera.get("name") for camera in root.iter("camera") if camera.get("name")}
    _append_camera_specs(worldbody, specs, existing_names=existing_names)
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _inject_trajectory_cameras(
    xml_str: str,
    config: Optional[TrajectoryCameraConfig],
) -> str:
    """Injects the generated trajectory cameras into the XML string."""

    if config is None:
        return xml_str

    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML missing <worldbody>, cannot inject trajectory cameras")

    camera_map = {
        camera.get("name"): camera
        for camera in root.iter("camera")
        if camera.get("name") is not None
    }
    base_camera_elem = camera_map.get(config.base_camera_name)
    if base_camera_elem is None:
        raise ValueError(f"Base camera '{config.base_camera_name}' not found in XML")

    base_pos = _parse_vector_attr(base_camera_elem.get("pos"), 3, "pos")
    base_quat = _parse_vector_attr(base_camera_elem.get("quat", "1 0 0 0"), 4, "quat")
    specs = _generate_trajectory_camera_specs(
        config=config,
        base_pos=base_pos,
        base_quat_wxyz=base_quat,
        camera_fovy=base_camera_elem.get("fovy"),
    )

    existing_names = {camera.get("name") for camera in root.iter("camera") if camera.get("name")}
    _append_camera_specs(worldbody, specs, existing_names=existing_names)
    return ET.tostring(root, encoding="utf8").decode("utf8")


def install_model_xml_remapper(
    libero_assets_root: Optional[str] = None,
    legacy_asset_markers: Optional[Iterable[str]] = None,
    operation_config: Optional[OperationCameraConfig] = None,
    trajectory_config: Optional[TrajectoryCameraConfig] = None,
) -> tuple[Optional[Path], Path, tuple[str, ...]]:
    """Installs the XML remapper used during dataset replay.

    Args:
        libero_assets_root: Optional override for the LIBERO assets root.
        legacy_asset_markers: Optional asset path markers used to rewrite
            historical dataset XML paths.
        operation_config: Configuration for the generated operation cameras.
        trajectory_config: Configuration for the generated trajectory cameras.

    Returns:
        A tuple of ``(robosuite_root, libero_assets_root, legacy_asset_markers)``.
    """

    resolved_markers = tuple(legacy_asset_markers or DEFAULT_LEGACY_ASSET_MARKERS)
    robosuite_root = _resolve_robosuite_root()
    resolved_assets_root = _resolve_libero_assets_root(libero_assets_root)

    def patched_postprocess(xml_str: str, cameras_dict: Optional[dict] = None) -> str:
        xml = _ORIGINAL_POSTPROCESS_MODEL_XML(xml_str, cameras_dict or {})
        xml = _rewrite_model_xml_paths(
            xml_str=xml,
            robosuite_root=robosuite_root,
            libero_assets_root=resolved_assets_root,
            legacy_markers=resolved_markers,
        )
        xml = _inject_operation_cameras(xml, operation_config)
        return _inject_trajectory_cameras(xml, trajectory_config)

    replay_utils.libero_utils.postprocess_model_xml = patched_postprocess
    return robosuite_root, resolved_assets_root, resolved_markers
