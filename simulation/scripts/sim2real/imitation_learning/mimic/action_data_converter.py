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
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert recorded demonstration actions between IK and joint space."""

import argparse
import json
import multiprocessing
import os
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# Root of the `simulation/` tree, used to locate each robot's env_cfg.json.
_SIM_ROOT = Path(__file__).resolve().parents[4]
_ENV_CFG_BASE = _SIM_ROOT / "source" / "robotis_lab" / "robotis_lab" / "real_world_tasks" / "manager_based"


def _default_env_cfg_path(robot: str) -> Path:
    """Default env_cfg.json location for a robot (e.g. .../manager_based/OMX/env_cfg.json)."""
    return _ENV_CFG_BASE / robot.upper() / "env_cfg.json"

# Per-robot joint layout and default offsets.
# The joint_pos_target observation is ordered as arm_joints + [gripper_joint] for both robots,
# so the offset vector applied during ik->joint conversion follows the same order.
# Note: the gripper offset lives under actions.record.gripper_action.offset (a scalar),
# NOT inside the arm_action.offset dict.
_ROBOT_CONFIGS = {
    "omx": {
        "arm_joints": ["joint1", "joint2", "joint3", "joint4", "joint5"],
        "gripper_joint": "gripper_joint_1",
        "default_offset": {
            "joint1": 0.0,
            "joint2": 0.02572,
            "joint3": 0.15,
            "joint4": 0.0015,
            "joint5": 0.0,
            "gripper_joint_1": 0.0,
        },
    },
    "so101": {
        "arm_joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        "gripper_joint": "gripper",
        "default_offset": {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.0145,
            "elbow_flex": -0.03,
            "wrist_flex": -0.0122,
            "wrist_roll": 0.0,
            "gripper": -0.17453292519943295,
        },
    },
}


def _joint_pos_target_order(robot_cfg: dict) -> list:
    """Full joint order (arm joints followed by the gripper joint) for a robot."""
    return list(robot_cfg["arm_joints"]) + [robot_cfg["gripper_joint"]]


def _parse_offset_from_env_cfg(env_cfg_path: Path, robot_cfg: dict) -> list:
    """Read the arm + gripper joint offset from env_cfg.json, falling back to defaults for missing keys."""
    try:
        cfg = json.loads(env_cfg_path.read_text())
        record_cfg = cfg.get("actions", {}).get("record", {})
        arm_offset = record_cfg.get("arm_action", {}).get("offset", {})
        gripper_offset = record_cfg.get("gripper_action", {}).get("offset", None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read env_cfg.json ({e}), using default offsets.")
        arm_offset = {}
        gripper_offset = None

    default_offset = robot_cfg["default_offset"]
    result = []
    for j in robot_cfg["arm_joints"]:
        result.append(arm_offset.get(j, default_offset.get(j, 0.0)))

    gripper_joint = robot_cfg["gripper_joint"]
    if gripper_offset is None:
        gripper_offset = default_offset.get(gripper_joint, 0.0)
    result.append(gripper_offset)
    return result

def convert_joint_to_ik(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint actions to IK (EEF state + gripper)."""
    try:
        eef_state = ep_data.data["obs"]["ee_frame_state"]
        joint_actions = ep_data.data["actions"]

        gripper_action = joint_actions[:, -1:]
        new_actions = torch.cat([eef_state, gripper_action], dim=1)

        ep_data.data["actions"] = new_actions
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert joint to IK: {str(e)}")

def convert_ik_to_joint(ep_data: EpisodeData, offset: torch.Tensor | None = None) -> EpisodeData:
    """Convert IK actions to joint targets, optionally removing the action offset."""
    try:
        joint_targets = ep_data.data["obs"]["joint_pos_target"].clone()
        if offset is not None:
            joint_targets = joint_targets - offset.to(joint_targets.device)
        ep_data.data["actions"] = joint_targets
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert IK to joint: {str(e)}")

ACTION_CONVERTERS = {
    "ik": convert_joint_to_ik,
    "joint": convert_ik_to_joint,
}


def process_dataset(input_file: str, output_file: str, action_type: str, offset: list[float] | None = None) -> None:
    """Process dataset episodes and convert actions to the desired type."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input dataset file does not exist: {input_file}")

    offset_tensor = torch.tensor(offset, dtype=torch.float32) if offset is not None else None

    base_converter = ACTION_CONVERTERS[action_type]
    if action_type == "joint" and offset_tensor is not None:
        converter = lambda ep: base_converter(ep, offset=offset_tensor)
    else:
        converter = base_converter

    input_handler = HDF5DatasetFileHandler()
    output_handler = HDF5DatasetFileHandler()

    input_handler.open(input_file)
    output_handler.create(output_file)

    try:
        episode_names = list(input_handler.get_episode_names())
        skipped_episodes = []
        
        for name in tqdm(episode_names, desc="Processing episodes"):
            try:
                ep_data = input_handler.load_episode(name, device="cpu")

                if ep_data.success is not None and not ep_data.success:
                    continue

                processed = deepcopy(ep_data)
                processed = converter(processed)
                output_handler.write_episode(processed)
                
            except Exception as e:
                skipped_episodes.append((name, str(e)))
                print(f"\nWarning: Skipping episode '{name}' due to error: {str(e)}")
                continue
        
        if skipped_episodes:
            print(f"\n\nSummary: Skipped {len(skipped_episodes)} episode(s) due to errors:")
            for ep_name, error_msg in skipped_episodes:
                print(f"  - {ep_name}: {error_msg}")

    finally:
        input_handler.close()
        output_handler.flush()
        output_handler.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert recorded demonstration actions between IK and joint space."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./datasets/annotated_dataset.hdf5",
        help="Path to input dataset file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/processed_annotated_dataset.hdf5",
        help="Path to save processed dataset file."
    )
    parser.add_argument(
        "--robot",
        choices=list(_ROBOT_CONFIGS.keys()),
        default="omx",
        help="Robot whose joint layout/offset to use (default: omx)."
    )
    parser.add_argument(
        "--action_type",
        choices=["ik", "joint"],
        required=True,
        help="Target action representation: 'ik' or 'joint'."
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Joint offset values to subtract from joint_pos_target when action_type=joint. "
            "Order follows the selected --robot (arm joints then gripper joint). "
            "If omitted, values are read automatically from --env_cfg_path. "
            "OMX example: --offset 0.0 0.02572 0.15 0.0015 0.0 0.0"
        ),
    )
    parser.add_argument(
        "--env_cfg_path",
        type=str,
        default=None,
        help=(
            "Path to env_cfg.json used to auto-read the joint offset "
            "when --offset is not given and action_type=joint. "
            "If omitted, defaults to the selected --robot's env_cfg.json under "
            "source/robotis_lab/.../manager_based/<ROBOT>/env_cfg.json."
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()

    robot_cfg = _ROBOT_CONFIGS[args.robot]
    joint_order = _joint_pos_target_order(robot_cfg)

    offset = args.offset
    if offset is None and args.action_type == "joint":
        env_cfg_path = Path(args.env_cfg_path) if args.env_cfg_path else _default_env_cfg_path(args.robot)
        if not env_cfg_path.exists():
            raise FileNotFoundError(
                f"env_cfg_path not found: {env_cfg_path}\n"
                "Please provide --offset manually or fix --env_cfg_path."
            )
        offset = _parse_offset_from_env_cfg(env_cfg_path, robot_cfg)
        print(f"Auto-loaded joint offset for '{args.robot}' from {env_cfg_path.name}:")
        for name, val in zip(joint_order, offset):
            print(f"  {name}: {val}")
    elif offset is not None and len(offset) != len(joint_order):
        raise ValueError(
            f"--offset expects {len(joint_order)} values for robot '{args.robot}' "
            f"({', '.join(joint_order)}), but got {len(offset)}."
        )

    process_dataset(args.input_file, args.output_file, args.action_type, offset)

if __name__ == "__main__":
    main()
