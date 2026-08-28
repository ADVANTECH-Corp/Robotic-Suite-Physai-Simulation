# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Isaac Lab Mimic wrapper for the SO101 bottle pick-place IK env.

Logic is identical to ``OMXPickPlaceMimicEnv`` — only the class name and
expected eef key in subtask configs differ (set to "SO101" by the env cfg).
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv, ManagerBasedRLEnvCfg


class SO101PickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic env wrapper for SO101 Bottle Pick-Place IK env."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_root_pos = self.scene["robot"].data.root_pos_w
        self.robot_root_quat = self.scene["robot"].data.root_quat_w

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        eef_state = self.obs_buf["policy"]["ee_frame_state"][env_ids]
        eef_pos = eef_state[:, :3]
        eef_quat = eef_state[:, 3:7]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_eef_pos, target_eef_rot = PoseUtils.unmake_pose(target_eef_pose)
        target_eef_quat = PoseUtils.quat_from_matrix(target_eef_rot)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([target_eef_pos, target_eef_quat], dim=0)
        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_eef_pos = action[:, :3]
        target_eef_quat = action[:, 3:7]
        target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)

        target_eef_pose = PoseUtils.make_pose(target_eef_pos, target_eef_rot).clone()
        return {eef_name: target_eef_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        return {eef_name: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        for term_name, term_signal in subtask_terms.items():
            signals[term_name] = term_signal[env_ids]
        return signals
