# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SO101 pick-place reset/event terms.

Identical to ``omx_pick_place_events`` except for
``randomize_joint_by_gaussian_offset``: SO101 has a single gripper joint
(``gripper``) instead of OMX's mimic pair (``gripper_joint_1`` /
``gripper_joint_2``), so we only freeze the last joint.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

# Re-export the rest of the OMX events module unchanged.
from robotis_lab.real_world_tasks.manager_based.OMX.pick_place.mdp.omx_pick_place_events import (  # noqa: F401
    set_default_joint_pose,
    sample_object_poses,
    randomize_object_pose,
    randomize_scene_lighting_domelight,
    randomize_scene_lighting_spherelight,
    randomize_camera_pose,
)


def randomize_joint_by_gaussian_offset(
    env,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper pose (SO101 has a single gripper joint)
    joint_pos[:, -1:] = asset.data.default_joint_pos[env_ids, -1:]

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
