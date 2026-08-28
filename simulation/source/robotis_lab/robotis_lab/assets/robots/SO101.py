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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR

# Joint names match the leader /leader/joint_trajectory message:
#   shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/SO101/so101.usd",
        scale=(1.0, 1.0, 1.0),
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
        pos=[0.0, 0.51031, -0.35671],
        rot=[0.70711, 0.0, 0.0, -0.70711],
        joint_pos={
            "shoulder_pan": -0.06828,
            "shoulder_lift": -1.70466,
            "elbow_flex": 1.39779,
            "wrist_flex": 1.37708,
            "wrist_roll": -0.10203,
            "gripper": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift"],
            velocity_limit_sim=6.0,
            effort_limit_sim=100.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex"],
            velocity_limit_sim=6.0,
            effort_limit_sim=80.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex", "wrist_roll"],
            velocity_limit_sim=6.0,
            effort_limit_sim=80.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            velocity_limit_sim=10.0,
            effort_limit_sim=50.0,
            stiffness=10.0,
            damping=0.2,
        ),
    },
)
"""Configuration of SO101 follower arm using implicit actuator models."""

SO101_OFF_SELF_COLLISION_CFG = SO101_CFG.replace(
    spawn=SO101_CFG.spawn.replace(
        articulation_props=SO101_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=False,
        )
    )
)
