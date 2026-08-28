# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert an IsaacLab SO-101 HDF5 dataset into LeRobot format.

Adapted from the OMX converter (../OMX/isaaclab2lerobot.py). Differences:
  * 6 SO-101 joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper) instead of OMX's joint1..joint5 + gripper_joint_1.
  * Gripper floor restoration handles SO-101's two-space recording: the
    recorded ``actions`` are in WIRE space (pre env remap) while
    ``obs/joint_pos`` are in JOINT space (post env remap). See GRIPPER section.
"""

import math
import h5py
import numpy as np
import argparse
import av
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.utils import (
    validate_frame,
    validate_episode_buffer,
    write_episode,
    write_episode_stats,
    write_info,
)

# SO-101 joint order (matches the leader /leader/joint_trajectory message and
# the recorded action/observation columns).
SO101_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# ---------------------------------------------------------------------------
# Gripper floor restoration (mirror OMY's clamp-artifact correction).
#
# so101_sdk.py floors the gripper COMMAND at radians(SO101_GRIPPER_MIN_OPEN_PCT)
# before it is returned by get_action() and recorded, so any value at/below the
# floor means the operator drove the leader gripper fully closed. We restore
# those to the true fully-closed value in EACH column's own space:
#   * actions      -> WIRE space (pre env remap):  fully closed = radians(0) = 0
#   * obs/joint_pos -> JOINT space (post env remap, scale 1.1 / offset
#                      radians(-10)): fully closed = radians(-10)
#
# NOTE: the obs threshold/closed value depend on the env gripper remap
# (SO101 pick_place env_cfg.json actions.*.gripper_action). Keep these in sync
# if that remap changes.
# ---------------------------------------------------------------------------
GRIPPER_IDX = 5
_EPS = 1e-3

_GRIPPER_MIN_OPEN_PCT = 15.0           # matches so101_sdk SO101_GRIPPER_MIN_OPEN_PCT
_GRIPPER_SCALE = 1.1                   # matches env gripper_action scale
_GRIPPER_OFFSET = math.radians(-10.0)  # matches env gripper_action offset

# action (wire) space
_ACTION_GRIPPER_FLOOR = math.radians(_GRIPPER_MIN_OPEN_PCT)   # ~0.2618
_ACTION_GRIPPER_CLOSED = 0.0
# obs (joint) space: the wire floor passed through the env remap
_STATE_GRIPPER_FLOOR = _GRIPPER_SCALE * _ACTION_GRIPPER_FLOOR + _GRIPPER_OFFSET  # ~0.0175
_STATE_GRIPPER_CLOSED = math.radians(-10.0)                   # -0.1745

CAMERA_CONFIG = {
    "cam_wrist": {"height": 480, "width": 640},
    "cam_top": {"height": 480, "width": 640},
}


def get_env_features(fps: int, camera_config=CAMERA_CONFIG):
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(SO101_JOINT_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(SO101_JOINT_NAMES),
        }
    }

    for cam_name, cfg in camera_config.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [cfg["height"], cfg["width"], 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": cfg["height"],
                "video.width": cfg["width"],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return features


class LeRobotH264Dataset(LeRobotDataset):
    """LeRobotDataset subclass that writes h264 video directly with PyAV.

    Bypasses lerobot's PNG-intermediate pipeline:
      - add_frame_without_write_image : stores raw numpy frames in memory
      - save_episode_without_write_image : writes h264 MP4 via PyAV directly,
        then handles parquet and metadata through standard lerobot internals.
    """

    def add_frame_without_write_image(self, frame: dict, task: str) -> None:
        """Buffer a frame without writing images to disk as PNG."""
        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp", frame_index / self.fps)
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key, value in frame.items():
            if key not in self.episode_buffer:
                self.episode_buffer[key] = [value]
            else:
                self.episode_buffer[key].append(value)

        self.episode_buffer["task"].append(task)
        self.episode_buffer["size"] += 1

    def save_episode_without_write_image(self) -> None:
        """Save episode: write h264 directly with PyAV, skip PNG pipeline."""
        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(t) for t in tasks]
        )

        # Stack non-video features into numpy arrays for parquet
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"]:
                continue
            if ft["dtype"] in ["image", "video"]:
                continue
            if key in episode_buffer and isinstance(episode_buffer[key], list):
                episode_buffer[key] = np.stack(episode_buffer[key])

        # Save state/action data to parquet (video keys are ignored by _save_episode_table)
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = self._compute_episode_stats(episode_buffer)

        # Write h264 videos directly with PyAV
        video_count = 0
        for key, ft in self.features.items():
            if ft["dtype"] != "video" or key not in episode_buffer:
                continue
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            self._write_video_h264(episode_buffer[key], video_path)
            video_count += 1
            self.meta.info["features"][key]["info"] = {
                "video.height": ft["shape"][0],
                "video.width": ft["shape"][1],
                "video.channels": ft["shape"][2],
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
            }

        # Update metadata
        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1
        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length
        self.meta.info["total_videos"] += video_count
        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}

        write_info(self.meta.info, self.meta.root)
        write_episode(
            {"episode_index": episode_index, "tasks": episode_tasks, "length": episode_length},
            self.meta.root,
        )
        write_episode_stats(episode_index, ep_stats, self.meta.root)

        self.episode_buffer = None

    def _write_video_h264(
        self, frames: list, video_path: Path, crf: int = 23, g: int = 2
    ) -> None:
        """Encode a list of RGB numpy frames to h264 MP4 using PyAV."""
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(video_path), "w") as output:
            stream = output.add_stream("libx264", self.fps)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "g": str(g)}
            for frame_arr in frames:
                av_frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
                av_frame = av_frame.reformat(format="yuv420p")
                for packet in stream.encode(av_frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)

    def _compute_episode_stats(self, episode_buffer: dict) -> dict:
        """Compute per-episode statistics, handling video as numpy arrays."""
        ep_stats = {}
        for key, data in episode_buffer.items():
            if key not in self.features:
                continue
            dtype = self.features[key]["dtype"]
            if dtype == "string":
                continue
            if dtype in ["image", "video"]:
                sampled = self._sample_images_from_arrays(data)
                stats = get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True)
                ep_stats[key] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in stats.items()
                }
            else:
                arr = data if isinstance(data, np.ndarray) else np.array(data)
                ep_stats[key] = get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1)
        return ep_stats

    def _sample_images_from_arrays(self, image_array: list) -> np.ndarray:
        """Sample frames from a list of (H,W,C) numpy arrays → (N,C,H,W)."""
        n = len(image_array)
        num_samples = max(100, min(int(n ** 0.75), 10_000))
        num_samples = min(num_samples, n)
        indices = np.round(np.linspace(0, n - 1, num_samples)).astype(int)
        return np.stack([np.transpose(image_array[i], (2, 0, 1)) for i in indices])


def process_data(
    dataset: LeRobotH264Dataset, task: str, demo_group: h5py.Group,
    demo_name: str, frame_skip: int, first_frame_dir: Path | None = None
) -> bool:
    """
    Process a single demonstration group from the HDF5 dataset
    and add it into the LeRobot dataset.
    """
    try:
        # Load entire data arrays from HDF5
        actions = np.array(demo_group['actions'], dtype=np.float32)
        joint_pos = np.array(demo_group['obs/joint_pos'], dtype=np.float32)
        cam_wrist_images = np.array(demo_group['obs/cam_wrist'], dtype=np.uint8)
        cam_top_images = np.array(demo_group['obs/cam_top'], dtype=np.uint8)
    except KeyError:
        print(f"Demo {demo_name} is not valid, skipping...")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has insufficient frames ({actions.shape[0]}), skipping...")
        return False

    # Ensure actions and joint positions are 2D arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, 6)
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, 6)

    # Correct gripper floor artifact: so101_sdk floors the gripper command before
    # recording, so floored values represent "operator fully closed". Restore
    # them per column space (see the GRIPPER section at the top of this file):
    #   actions are WIRE space, obs/joint_pos are JOINT space (post env remap).
    actions[:, GRIPPER_IDX] = np.where(
        actions[:, GRIPPER_IDX] <= _ACTION_GRIPPER_FLOOR + _EPS,
        _ACTION_GRIPPER_CLOSED, actions[:, GRIPPER_IDX])
    joint_pos[:, GRIPPER_IDX] = np.where(
        joint_pos[:, GRIPPER_IDX] <= _STATE_GRIPPER_FLOOR + _EPS,
        _STATE_GRIPPER_CLOSED, joint_pos[:, GRIPPER_IDX])

    total_state_frames = actions.shape[0]

    # Process each frame
    first_frame_saved = False
    for frame_index in tqdm(range(total_state_frames), desc=f"Processing demo {demo_name}"):
        if frame_index < frame_skip:
            continue
        if not first_frame_saved and first_frame_dir is not None:
            first_frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cam_top_images[frame_index]).save(
                first_frame_dir / f"{demo_name}_cam_top.png"
            )
            first_frame_saved = True
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.cam_wrist": cam_wrist_images[frame_index],
            "observation.images.cam_top": cam_top_images[frame_index],
        }
        dataset.add_frame_without_write_image(frame=frame, task=task)

    return True


def convert_isaaclab_to_lerobot(
    task: str, repo_id: str, robot_type: str, dataset_file: str,
    fps: int, push_to_hub: bool = False, frame_skip: int = 3, root: str = "./datasets/lerobot/so101_data",
    first_frame_dir: str | None = None,
):
    """
    Convert an IsaacLab HDF5 dataset into LeRobot dataset format.
    """
    hdf5_files = [dataset_file]
    now_episode_index = 0

    # Create a new LeRobot dataset using the h264-direct subclass
    dataset = LeRobotH264Dataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=get_env_features(fps),
        root=root,
    )

    # Process each HDF5 dataset file
    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]

                # Skip unsuccessful demonstrations
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} not successful, skipping...")
                    continue

                valid = process_data(
                    dataset, task, demo_group, demo_name, frame_skip,
                    first_frame_dir=Path(first_frame_dir) if first_frame_dir else None,
                )

                if valid:
                    now_episode_index += 1
                    dataset.save_episode_without_write_image()
                    print(f"Saved episode {now_episode_index} successfully")

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab SO-101 dataset to LeRobot format")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., SO101_Pickup)")
    parser.add_argument("--robot_type", type=str, default="SO101", help="Robot type (default: SO101)")
    parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Path to dataset HDF5 file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for dataset (default: 30)")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to HuggingFace Hub")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip rate (default: 2)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_repo_id = f"./datasets/lerobot/{timestamp}"
    parser.add_argument("--repo_id", type=str, default=default_repo_id, help=f"Repo ID (default: {default_repo_id})")
    parser.add_argument("--root", type=str, default=None, help="Output directory for the dataset (default: same as --repo_id)")
    parser.add_argument("--first_frame_dir", type=str, default=None, help="Directory to save first frame PNGs of each episode (default: <root>/first_frames)")

    args = parser.parse_args()
    root = args.root if args.root is not None else args.repo_id
    first_frame_dir = args.first_frame_dir if args.first_frame_dir is not None else f"{root}/first_frames"

    convert_isaaclab_to_lerobot(
        task=args.task,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_file=args.dataset_file,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        frame_skip=args.frame_skip,
        root=root,
        first_frame_dir=first_frame_dir,
    )
