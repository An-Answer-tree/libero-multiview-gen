"""Visualize all RGB camera views of one episode as per-frame grids."""

import argparse
import math
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

PREFERRED_OBS_ORDER = [
    "eye_in_hand_rgb",
    "agentview_rgb",
    "operation_topview_rgb",
    "operation_leftview_rgb",
    "operation_rightview_rgb",
    "operation_backview_rgb",
]


def sorted_demo_keys(data_group):
    """Returns episode keys sorted by numeric suffix."""

    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(demo_keys, key=lambda key: int(key.split("_")[1]))


def _discover_rgb_obs_keys(obs_group):
    keys = [key for key in obs_group.keys() if key.endswith("_rgb")]
    preferred = [key for key in PREFERRED_OBS_ORDER if key in keys]
    remaining = sorted(key for key in keys if key not in preferred)
    return preferred + remaining


def _make_canvas(frame_images, labels, cell_w, cell_h, cols, pad, header_h):
    rows = int(math.ceil(len(frame_images) / float(cols)))
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = rows * (cell_h + header_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for idx, (image, label) in enumerate(zip(frame_images, labels)):
        row = idx // cols
        col = idx % cols
        x = pad + col * cell_w
        y = pad + row * (cell_h + header_h)
        draw.text((x, y), label, fill=(240, 240, 240))
        tile = Image.fromarray(image.astype(np.uint8), mode="RGB")
        canvas.paste(tile, (x, y + header_h))

    return np.asarray(canvas, dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize all RGB camera views of one episode as per-frame grids."
    )
    parser.add_argument("--dataset", required=True, help="Path to replayed hdf5 dataset.")
    parser.add_argument(
        "--demo-key",
        default=None,
        help="Episode key such as demo_0. Defaults to the first episode.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save frame grids and the summary video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS used for the exported MP4 video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = os.path.abspath(os.path.expanduser(args.dataset))
    output_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        demo_key = args.demo_key or sorted_demo_keys(data_group)[0]
        if demo_key not in data_group:
            raise KeyError(f"Demo key not found: {demo_key}")

        obs_group = data_group[demo_key]["obs"]
        obs_keys = _discover_rgb_obs_keys(obs_group)
        if not obs_keys:
            raise ValueError(f"No *_rgb observations found in {dataset_path}:{demo_key}/obs")

        num_frames = min(obs_group[key].shape[0] for key in obs_keys)
        if args.max_frames is not None:
            num_frames = min(num_frames, int(args.max_frames))

        sample = np.asarray(obs_group[obs_keys[0]][0])
        cell_h, cell_w = int(sample.shape[0]), int(sample.shape[1])
        cols = int(math.ceil(math.sqrt(len(obs_keys))))
        pad = 16
        header_h = 24

        video_path = output_dir / f"{demo_key}_multiview.mp4"
        with imageio.get_writer(video_path, fps=int(args.fps)) as writer:
            for frame_idx in range(num_frames):
                frame_images = [np.asarray(obs_group[key][frame_idx]) for key in obs_keys]
                grid = _make_canvas(
                    frame_images=frame_images,
                    labels=obs_keys,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    cols=cols,
                    pad=pad,
                    header_h=header_h,
                )
                Image.fromarray(grid).save(frames_dir / f"frame_{frame_idx:05d}.png")
                writer.append_data(grid)

    print(f"[ok] dataset={dataset_path}")
    print(f"[ok] demo_key={demo_key}")
    print(f"[ok] obs_keys={obs_keys}")
    print(f"[ok] frames={num_frames}")
    print(f"[ok] video={video_path}")
    print(f"[ok] frames_dir={frames_dir}")


if __name__ == "__main__":
    main()
