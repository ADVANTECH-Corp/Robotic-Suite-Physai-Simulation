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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR

OMX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/OMX/OMX.usd",
        scale=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=[0.0, 0.48596, -0.32845],
        rot=[0.70711, 0.0, 0.0, -0.70711],
        joint_pos={
            "joint1": 0.0,
            "joint2": -2.0,
            "joint3": 1.5,
            "joint4": 1.5,
            "joint5": 0.0,
            "gripper_joint_1": 0.7,
        },
    ),
    actuators={
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=100.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-5]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=80.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_1"],
            velocity_limit_sim=10.0,
            effort_limit_sim=50.0,
            stiffness=10.0,
            damping=0.2,
        ),
    },
)

"""Configuration of OMX arm using implicit actuator models."""
OMX_OFF_SELF_COLLISION_CFG = OMX_CFG.replace(
    spawn=OMX_CFG.spawn.replace(
        articulation_props=OMX_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=False,
        )
    )
)
