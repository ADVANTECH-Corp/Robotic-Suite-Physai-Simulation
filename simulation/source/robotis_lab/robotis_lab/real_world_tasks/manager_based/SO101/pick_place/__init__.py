# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import gymnasium as gym

gym.register(
    id="RobotisLab-Real-Pick-Place-Bottle-SO101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SO101BottlePickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-Real-Mimic-Pick-Place-Bottle-SO101-v0",
    entry_point="robotis_lab.real_world_tasks.manager_based.SO101.pick_place.pick_place_mimic_env:SO101PickPlaceMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_mimic_env_cfg:SO101PickPlaceMimicEnvCfg",
    },
    disable_env_checker=True,
)
