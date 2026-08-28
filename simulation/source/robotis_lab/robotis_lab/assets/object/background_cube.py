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
# This file is derived from robotis_lab and has been modified.

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


BACKGROUND_CUBE_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.1, 3.0, 2.0),  # thin, wide, tall cube as background wall
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Static object, no physics
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,  # No collision
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.5),  # Initial gray color, will be randomized
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.2, 0.0, 1.0),  # Behind the table at x=1.2
    ),
)
