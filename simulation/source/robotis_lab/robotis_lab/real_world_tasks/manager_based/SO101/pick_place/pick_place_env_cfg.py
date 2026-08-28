# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SO101 pick-place base environment configuration.

Mirrors the OMX ``pick_place_env_cfg`` but uses SO101 joint names:
  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
"""

import json
import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from robotis_lab.real_world_tasks.manager_based.OMX.pick_place.pick_place_env_cfg import _merge_with_log

from . import mdp


# SO101 joint groupings (matches the leader /leader/joint_trajectory message).
SO101_ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SO101_GRIPPER_JOINT = "gripper"
SO101_ALL_JOINTS = SO101_ARM_JOINTS + [SO101_GRIPPER_JOINT]

# End-effector body name in the SO101 USD (under so101_follower).
SO101_EE_BODY = "gripper"


# Gripper wire->sim remap.
# The leader publishes the gripper as math.radians(0..100) -> 0..1.745 rad on
# the wire (a 0-100 lerobot percentage faked into "degrees"). The sim gripper
# joint physical range is -10..100 deg = math.radians(-10)..math.radians(100)
# rad. Map the wire span (0..1.745) linearly onto the joint span so the leader's
# fully-closed (0%) reaches the -10 deg lower limit and fully-open (100%) reaches
# the 100 deg upper limit:
#   target = scale * wire + offset
#   scale  = (upper - lower) / 100 = 110 / 100 = 1.1
#   offset = radians(lower)        = radians(-10)
# NOTE: scale/offset are derived from THIS USD's gripper limits (-10..100 deg).
# Re-derive if the asset or calibration changes.
_GRIPPER_LOWER_DEG = -10.0
_GRIPPER_UPPER_DEG = 100.0
_GRIPPER_SCALE = (_GRIPPER_UPPER_DEG - _GRIPPER_LOWER_DEG) / 100.0
_GRIPPER_OFFSET = math.radians(_GRIPPER_LOWER_DEG)

_DEFAULT_AC = {
    "record": {
        "arm_action": {
            "scale": 1.0,
            "offset": {name: 0.0 for name in SO101_ARM_JOINTS},
        },
        "gripper_action": {"scale": _GRIPPER_SCALE, "offset": _GRIPPER_OFFSET},
    },
    "mimic_ik": {
        "arm_action": {
            "controller": {"ik_params": {"lambda_val": 0.05}},
            "body_offset": {"pos": [0.0, 0.0, -0.09]},
        },
        "gripper_action": {"scale": _GRIPPER_SCALE, "offset": _GRIPPER_OFFSET},
    },
}

try:
    _cfg = json.loads((Path(__file__).parents[1] / "env_cfg.json").read_text())
    _ac = _merge_with_log(_DEFAULT_AC, _cfg.get("actions", {}), "actions")
except FileNotFoundError:
    print("[env_cfg] env_cfg.json not found, using all defaults for actions")
    _ac = _DEFAULT_AC


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO101 pick and place scene."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table: AssetBaseCfg = MISSING

    cam_wrist: CameraCfg = MISSING
    cam_top: CameraCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.5, 0.5, 0.5), intensity=3000.0),
    )

    SphereLight = AssetBaseCfg(
        prim_path="/World/GroundPlane/SphereLight",
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": list(SO101_ALL_JOINTS), "asset_name": "robot"},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_name,
            params={"joint_names": list(SO101_ALL_JOINTS), "asset_name": "robot"},
        )
        cam_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist"), "data_type": "rgb", "normalize": False},
        )
        cam_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_top"), "data_type": "rgb", "normalize": False},
        )
        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": list(SO101_ALL_JOINTS), "asset_name": "robot"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_bottle = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("bottle"),
            },
        )

        bottle_in_basket = ObsTerm(
            func=mdp.bottle_in_basket,
            params={
                "bottle_cfg": SceneEntityCfg("bottle"),
                "basket_cfg": SceneEntityCfg("basket"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "bottle_cfg": SceneEntityCfg("bottle"),
            "basket_cfg": SceneEntityCfg("basket"),
            "distance_threshold": 0.1,
        },
    )

    bottle_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.4, "asset_cfg": SceneEntityCfg("bottle")},
    )


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SO101 pick and place environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 30.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def init_action_cfg(self, mode: str):
        print(f"Initializing action configuration for device: {mode}")
        if mode in ['record', 'inference']:
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=list(SO101_ARM_JOINTS),
                scale=_ac["record"]["arm_action"]["scale"],
                use_default_offset=False,
                offset=_ac["record"]["arm_action"]["offset"],
            )
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[SO101_GRIPPER_JOINT],
                scale=_ac["record"]["gripper_action"]["scale"],
                offset=_ac["record"]["gripper_action"]["offset"],
                use_default_offset=False,
            )
        elif mode in ['mimic_ik']:
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=list(SO101_ARM_JOINTS),
                body_name=SO101_EE_BODY,
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    ik_params=_ac["mimic_ik"]["arm_action"]["controller"]["ik_params"],
                    ik_method="dls",
                    use_relative_mode=False,
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=_ac["mimic_ik"]["arm_action"]["body_offset"]["pos"]
                ),
            )
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[SO101_GRIPPER_JOINT],
                scale=_ac["mimic_ik"]["gripper_action"]["scale"],
                offset=_ac["mimic_ik"]["gripper_action"]["offset"],
                use_default_offset=False,
            )
        else:
            self.actions.arm_action = None
            self.actions.gripper_action = None
