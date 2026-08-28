# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SO101 pick-place observations.

Mirrors the OMX observations module but uses the SO101 single gripper joint
name (``gripper``) instead of OMX's ``gripper_joint_1``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils

# Re-export OMX implementations that are not robot-specific.
from robotis_lab.real_world_tasks.manager_based.OMX.pick_place.mdp.observations import (  # noqa: F401
    ee_frame_state,
    last_action,
    joint_pos_name,
    joint_vel_name,
    ee_frame_pos,
    ee_frame_quat,
    joint_pos_target_name,
    bottle_in_basket,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# SO101 gripper joint name (single joint, unlike OMX's mimic pair)
_SO101_GRIPPER_JOINT = "gripper"


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.02,
    gripper_close_threshold: torch.tensor = torch.tensor([0.33]),
) -> torch.Tensor:
    """Check if an object is grasped by the SO101 follower arm."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    gripper_joint_idx = robot.joint_names.index(_SO101_GRIPPER_JOINT)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        gripper_pos <= gripper_close_threshold.to(env.device),
    )
    return grasped
