import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T

import init_path
from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import TASK_MAPPING
import libero.libero.utils.utils as libero_utils


DEFAULT_BENCHMARKS = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]
DEFAULT_CAMERA_NAMES = ["robot0_eye_in_hand", "agentview"]


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def safe_env_reset(env, max_attempts=100):
    for _ in range(max_attempts):
        try:
            env.reset()
            return
        except Exception:
            continue
    raise RuntimeError(f"Failed to reset env after {max_attempts} attempts")


def sorted_demo_keys(data_group):
    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def camera_name_to_obs_key(camera_name):
    if camera_name == "robot0_eye_in_hand":
        return "eye_in_hand_rgb"
    return f"{camera_name}_rgb"


def correct_image_orientation(image):
    # MuJoCo offscreen render uses bottom-left as origin, flip to top-left image convention.
    return np.flip(image, axis=0).copy()


def discover_benchmark_tasks(source_root, benchmark_names=None, task_filter=None):
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    benchmark_names = benchmark_names or DEFAULT_BENCHMARKS
    task_filter = set(task_filter) if task_filter else None

    available = []
    missing = []
    for benchmark_name in benchmark_names:
        benchmark_key = benchmark_name.lower()
        if benchmark_key not in benchmark_dict:
            raise ValueError(
                f"Unsupported benchmark '{benchmark_name}', available: {sorted(benchmark_dict.keys())}"
            )
        benchmark_instance = benchmark_dict[benchmark_key]()
        for task_idx in range(benchmark_instance.get_num_tasks()):
            task = benchmark_instance.get_task(task_idx)
            if task_filter is not None and task.name not in task_filter:
                continue

            rel_path = benchmark_instance.get_task_demonstration(task_idx)
            src_path = os.path.join(source_root, rel_path)
            info = {
                "benchmark_name": benchmark_instance.name,
                "task_idx": task_idx,
                "task_name": task.name,
                "relative_demo_path": rel_path,
                "source_demo_path": src_path,
            }
            if os.path.exists(src_path):
                available.append(info)
            else:
                missing.append(info)

    return available, missing


def _fallback_env_args(data_group):
    env_info = {}
    if "env_info" in data_group.attrs:
        env_info = json.loads(decode_attr(data_group.attrs["env_info"]))

    problem_name = None
    if "problem_info" in data_group.attrs:
        problem_info = json.loads(decode_attr(data_group.attrs["problem_info"]))
        problem_name = problem_info.get("problem_name")

    bddl_file = decode_attr(data_group.attrs.get("bddl_file_name", ""))
    env_name = decode_attr(data_group.attrs.get("env_name", data_group.attrs.get("env", "")))
    return {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": bddl_file,
        "env_kwargs": env_info,
    }


def load_env_args(data_group):
    if "env_args" in data_group.attrs:
        return json.loads(decode_attr(data_group.attrs["env_args"]))
    return _fallback_env_args(data_group)


def build_replay_env(
    data_group,
    camera_names=None,
    camera_height=128,
    camera_width=128,
):
    env_args = load_env_args(data_group)
    env_kwargs = copy.deepcopy(env_args.get("env_kwargs", {}))

    problem_name = env_args.get("problem_name")
    if not problem_name and "problem_info" in data_group.attrs:
        problem_info = json.loads(decode_attr(data_group.attrs["problem_info"]))
        problem_name = problem_info.get("problem_name")
    if not problem_name:
        raise ValueError("Cannot resolve problem_name from source dataset attributes")

    bddl_file_name = decode_attr(
        data_group.attrs.get("bddl_file_name", env_args.get("bddl_file", ""))
    )
    if bddl_file_name:
        env_kwargs["bddl_file_name"] = bddl_file_name

    if camera_names is None:
        camera_names = env_kwargs.get("camera_names", DEFAULT_CAMERA_NAMES)

    libero_utils.update_env_kwargs(
        env_kwargs,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=list(camera_names),
        reward_shaping=True,
        control_freq=20,
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        camera_segmentations=None,
    )

    env = TASK_MAPPING[problem_name](**env_kwargs)

    rebuilt_env_args = {
        "type": 1,
        "env_name": env_args.get("env_name", decode_attr(data_group.attrs.get("env", ""))),
        "problem_name": problem_name,
        "bddl_file": bddl_file_name,
        "env_kwargs": env_kwargs,
    }
    return env, rebuilt_env_args


def _extract_proprio(obs):
    proprio = {}
    if "robot0_gripper_qpos" in obs:
        proprio["gripper_states"] = obs["robot0_gripper_qpos"]
    if "robot0_joint_pos" in obs:
        proprio["joint_states"] = obs["robot0_joint_pos"]
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        ee_state = np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
        proprio["ee_states"] = ee_state
    return proprio


def replay_demo_episode(
    env,
    source_episode_group,
    camera_names=None,
    no_proprio=False,
    divergence_threshold=0.01,
    camera_height=128,
    camera_width=128,
):
    if camera_names is None:
        camera_names = DEFAULT_CAMERA_NAMES

    source_states = np.array(source_episode_group["states"][()])
    actions = np.array(source_episode_group["actions"][()])
    model_xml = decode_attr(source_episode_group.attrs["model_file"])

    if len(source_states) == 0:
        raise ValueError("Episode has empty states")
    if len(actions) == 0:
        raise ValueError("Episode has empty actions")

    # We reload from XML directly so custom trajectory cameras only need to exist in this XML.
    model_xml = libero_utils.postprocess_model_xml(model_xml, {})
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(source_states[0])
    env.sim.forward()
    model_xml = env.sim.model.get_xml()

    obs_arrays = {}
    for camera_name in camera_names:
        obs_arrays[camera_name_to_obs_key(camera_name)] = []

    proprio_arrays = {}
    robot_states = []
    replay_states = []
    max_state_error = 0.0
    num_diverged_steps = 0

    def capture_current_step(obs):
        for camera_name in camera_names:
            img_key = f"{camera_name}_image"
            if img_key in obs:
                image = obs[img_key]
            else:
                # For dynamically injected cameras, render directly from sim if no observable exists.
                image = env.sim.render(
                    camera_name=camera_name,
                    width=int(camera_width),
                    height=int(camera_height),
                    depth=False,
                )
            corrected_image = correct_image_orientation(image)
            obs_arrays[camera_name_to_obs_key(camera_name)].append(corrected_image)

        if not no_proprio:
            step_proprio = _extract_proprio(obs)
            for key, value in step_proprio.items():
                if key not in proprio_arrays:
                    proprio_arrays[key] = []
                proprio_arrays[key].append(value)

        robot_states.append(env.get_robot_state_vector(obs))

    obs = env._get_observations(force_update=True)
    for step_idx, action in enumerate(actions):
        # Record the observation before applying the action so image[i] aligns with action[i].
        capture_current_step(obs)

        obs, _, _, _ = env.step(action)
        state_playback = env.sim.get_state().flatten()
        replay_states.append(state_playback)

        if step_idx < len(source_states) - 1:
            err = float(np.linalg.norm(source_states[step_idx + 1] - state_playback))
            max_state_error = max(max_state_error, err)
            if err > divergence_threshold:
                num_diverged_steps += 1

    obs_data = {}
    for key, values in obs_arrays.items():
        obs_data[key] = np.stack(values, axis=0)
    if not no_proprio:
        for key, values in proprio_arrays.items():
            obs_data[key] = np.stack(values, axis=0)
        if "ee_states" in obs_data:
            obs_data["ee_pos"] = obs_data["ee_states"][:, :3]
            obs_data["ee_ori"] = obs_data["ee_states"][:, 3:]

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

    return {
        "actions": actions,
        "states": source_states,
        "robot_states": np.stack(robot_states, axis=0),
        "obs_data": obs_data,
        "rewards": rewards,
        "dones": dones,
        "num_samples": int(len(actions)),
        "model_file": model_xml,
        "init_state": source_states[0],
        "replay_states": np.stack(replay_states, axis=0),
        "max_state_error": max_state_error,
        "num_diverged_steps": num_diverged_steps,
    }


def write_episode_to_hdf5(target_data_group, demo_key, source_episode_group, replay_episode):
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


def reconstruct_dataset_file(
    source_hdf5_path,
    output_hdf5_path,
    camera_names=None,
    no_proprio=False,
    divergence_threshold=0.01,
    camera_height=128,
    camera_width=128,
):
    out_path = Path(output_hdf5_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    episode_summaries = []

    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(out_path, "w") as target_file:
        for attr_key, attr_val in source_file.attrs.items():
            target_file.attrs[attr_key] = attr_val

        if "mask" in source_file:
            source_file.copy("mask", target_file)

        source_data_group = source_file["data"]
        target_data_group = target_file.create_group("data")
        for attr_key, attr_val in source_data_group.attrs.items():
            target_data_group.attrs[attr_key] = attr_val

        env, rebuilt_env_args = build_replay_env(
            source_data_group,
            camera_names=None,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        try:
            demo_keys = sorted_demo_keys(source_data_group)
            for demo_key in demo_keys:
                source_ep_group = source_data_group[demo_key]
                replay_ep = replay_demo_episode(
                    env=env,
                    source_episode_group=source_ep_group,
                    camera_names=camera_names,
                    no_proprio=no_proprio,
                    divergence_threshold=divergence_threshold,
                    camera_height=camera_height,
                    camera_width=camera_width,
                )
                write_episode_to_hdf5(target_data_group, demo_key, source_ep_group, replay_ep)
                total_samples += replay_ep["num_samples"]
                episode_summaries.append(
                    {
                        "demo_key": demo_key,
                        "num_samples": replay_ep["num_samples"],
                        "max_state_error": replay_ep["max_state_error"],
                        "num_diverged_steps": replay_ep["num_diverged_steps"],
                    }
                )
        finally:
            env.close()

        target_data_group.attrs["env_args"] = json.dumps(rebuilt_env_args)
        target_data_group.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
        target_data_group.attrs["num_demos"] = len(demo_keys)
        target_data_group.attrs["total"] = total_samples

    return {
        "source_hdf5_path": source_hdf5_path,
        "output_hdf5_path": str(out_path),
        "num_demos": len(episode_summaries),
        "total_samples": total_samples,
        "episodes": episode_summaries,
    }


def validate_reconstructed_file(source_hdf5_path, rebuilt_hdf5_path):
    errors = []
    warnings = []

    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(rebuilt_hdf5_path, "r") as rebuilt_file:
        if "data" not in source_file:
            errors.append("Source missing data group")
            return {"ok": False, "errors": errors, "warnings": warnings}
        if "data" not in rebuilt_file:
            errors.append("Rebuilt missing data group")
            return {"ok": False, "errors": errors, "warnings": warnings}

        source_data = source_file["data"]
        rebuilt_data = rebuilt_file["data"]
        src_demo_keys = sorted_demo_keys(source_data)
        dst_demo_keys = sorted_demo_keys(rebuilt_data)

        if len(src_demo_keys) != len(dst_demo_keys):
            errors.append(
                f"Episode count mismatch: source={len(src_demo_keys)}, rebuilt={len(dst_demo_keys)}"
            )
            return {"ok": False, "errors": errors, "warnings": warnings}

        src_action_min = np.inf
        src_action_max = -np.inf
        dst_action_min = np.inf
        dst_action_max = -np.inf

        for demo_key in src_demo_keys:
            if demo_key not in rebuilt_data:
                errors.append(f"Missing demo group in rebuilt: {demo_key}")
                continue

            src_ep = source_data[demo_key]
            dst_ep = rebuilt_data[demo_key]

            for required_key in ["actions", "states", "robot_states", "rewards", "dones", "obs"]:
                if required_key not in src_ep:
                    errors.append(f"Source {demo_key} missing key '{required_key}'")
                if required_key not in dst_ep:
                    errors.append(f"Rebuilt {demo_key} missing key '{required_key}'")

            if "actions" in src_ep and "actions" in dst_ep:
                if src_ep["actions"].shape != dst_ep["actions"].shape:
                    errors.append(
                        f"{demo_key}/actions shape mismatch {src_ep['actions'].shape} vs {dst_ep['actions'].shape}"
                    )
                src_actions = src_ep["actions"][()]
                dst_actions = dst_ep["actions"][()]
                src_action_min = min(src_action_min, float(np.min(src_actions)))
                src_action_max = max(src_action_max, float(np.max(src_actions)))
                dst_action_min = min(dst_action_min, float(np.min(dst_actions)))
                dst_action_max = max(dst_action_max, float(np.max(dst_actions)))

            if "states" in src_ep and "states" in dst_ep:
                if src_ep["states"].shape != dst_ep["states"].shape:
                    errors.append(
                        f"{demo_key}/states shape mismatch {src_ep['states'].shape} vs {dst_ep['states'].shape}"
                    )

            if "obs" in src_ep and "obs" in dst_ep:
                src_obs_keys = sorted(src_ep["obs"].keys())
                dst_obs_keys = sorted(dst_ep["obs"].keys())
                missing_obs = sorted(set(src_obs_keys) - set(dst_obs_keys))
                if missing_obs:
                    errors.append(f"{demo_key}/obs missing keys in rebuilt: {missing_obs}")
                for obs_key in src_obs_keys:
                    if obs_key not in dst_ep["obs"]:
                        continue
                    if src_ep["obs"][obs_key].shape != dst_ep["obs"][obs_key].shape:
                        errors.append(
                            f"{demo_key}/obs/{obs_key} shape mismatch {src_ep['obs'][obs_key].shape} vs {dst_ep['obs'][obs_key].shape}"
                        )

            src_num = int(src_ep.attrs.get("num_samples", -1))
            dst_num = int(dst_ep.attrs.get("num_samples", -1))
            if src_num != dst_num:
                errors.append(f"{demo_key} num_samples mismatch {src_num} vs {dst_num}")

        if src_action_min != np.inf and dst_action_min != np.inf:
            if not np.isclose(src_action_min, dst_action_min) or not np.isclose(
                src_action_max, dst_action_max
            ):
                warnings.append(
                    "Action range differs: "
                    f"source=[{src_action_min:.6f}, {src_action_max:.6f}] "
                    f"rebuilt=[{dst_action_min:.6f}, {dst_action_max:.6f}]"
                )

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}
