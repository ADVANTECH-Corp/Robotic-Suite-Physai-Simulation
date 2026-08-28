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

"""
Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(env: ManagerBasedRLEnv, bottle_cfg: SceneEntityCfg, basket_cfg: SceneEntityCfg, distance_threshold: float = 0.10) -> torch.Tensor:
    """
    Success = bottle placed inside the basket.
    """

    bottle: RigidObject = env.scene[bottle_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    bottle_pos = bottle.data.root_pos_w
    basket_pos = basket.data.root_pos_w

    # Check if bottle is close to basket horizontally (x, y)
    horizontal_distance = torch.linalg.vector_norm(bottle_pos[:, :2] - basket_pos[:, :2], dim=1)
    horizontal_ok = horizontal_distance < distance_threshold

    # Check if bottle is within the basket vertically.
    # basket root origin is near the rim (top), basket height is 4cm.
    basket_bottom = basket_pos[:, 2] - 0.04  # 4cm below rim = basket bottom
    basket_rim = basket_pos[:, 2] + 0.01    # 1cm above rim for tolerance

    vertical_ok = (bottle_pos[:, 2] > basket_bottom) & (bottle_pos[:, 2] < basket_rim)

    done = horizontal_ok & vertical_ok

    if env.num_envs == 1:
        print(
            f"[DEBUG task_done] bottle_z={bottle_pos[0, 2].item():.4f}"
            f" | basket_z={basket_pos[0, 2].item():.4f}"
            f" | rim={basket_rim[0].item():.4f}"
            f" | h_dist={horizontal_distance[0].item():.4f} (threshold={distance_threshold})"
            f" | h_ok={horizontal_ok[0].item()} v_ok={vertical_ok[0].item()} done={done[0].item()}"
        )

    return done
