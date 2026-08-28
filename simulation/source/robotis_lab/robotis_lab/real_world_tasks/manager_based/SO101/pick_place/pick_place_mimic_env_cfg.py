# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mimic env configuration for SO101 bottle pick-place.

Mirrors the OMX mimic cfg; subtask key uses ``SO101`` so
``SO101PickPlaceMimicEnv`` picks the correct eef name.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .joint_pos_env_cfg import SO101BottlePickPlaceEnvCfg


@configclass
class SO101PickPlaceMimicEnvCfg(SO101BottlePickPlaceEnvCfg, MimicEnvCfg):
    """Configuration for SO101 pick_place task with mimic environment."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "pick_and_place_the_bottle_in_the_basket"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        subtask_configs = []
        # First subtask: Grasp the bottle
        subtask_configs.append(
            SubTaskConfig(
                object_ref="bottle",
                subtask_term_signal="grasp_bottle",
                subtask_term_offset_range=(0, 2),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp bottle",
                next_subtask_description="Place bottle in basket",
            )
        )
        # Second subtask: Place bottle in basket
        subtask_configs.append(
            SubTaskConfig(
                object_ref="basket",
                subtask_term_signal="bottle_in_basket",
                subtask_term_offset_range=(0, 2),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place bottle in basket",
                next_subtask_description="Task complete",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0001,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["SO101"] = subtask_configs
