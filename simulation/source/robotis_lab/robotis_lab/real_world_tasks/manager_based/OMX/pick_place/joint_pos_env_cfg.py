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

import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from robotis_lab.real_world_tasks.manager_based.OMX.pick_place.mdp import omx_pick_place_events
from robotis_lab.real_world_tasks.manager_based.OMX.pick_place.pick_place_env_cfg import PickPlaceEnvCfg, _merge_with_log

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.OMX import OMX_CFG  # isort: skip
from robotis_lab.assets.object.robotis_omy_table import OMY_TABLE_CFG
from robotis_lab.assets.object.plastic_bottle import PLASTIC_BOTTLE_CFG
from robotis_lab.assets.object.plastic_basket import PLASTIC_BASKET_CFG

def _rpy_to_wxyz(roll: float, pitch: float, yaw: float) -> tuple:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _parse_cam_rot(rot) -> tuple:
    if isinstance(rot, dict):
        return _rpy_to_wxyz(math.radians(rot["roll"]), math.radians(rot["pitch"]), math.radians(rot["yaw"]))
    return tuple(rot)


_DEFAULT_EV = {
    "init_omx_arm_pose": {
        "default_pose": [-0.0138, -2.02178, 1.57386, 1.695, -0.01994, 0.69335, -0.69335],
    },
    "randomize_omx_joint_state": {"mean": 0.0, "std": 0.02},
    "randomize_bottle_positions": {
        "pose_range": {"x": [-0.05, 0.04], "y": [0.2, 0.2], "z": [-0.30074, -0.30074], "yaw": [-1.5707963267948966, -1.5707963267948966]},
        "min_separation": 0.1,
    },
    "randomize_basket_positions": {
        "pose_range": {"x": [-0.1848, -0.1848], "y": [0.27301, 0.27301], "z": [-0.29502, -0.29502], "roll": [-1.5707963267948966, -1.5707963267948966], "pitch": [3.141592653589793, 3.141592653589793], "yaw": [0.0, 0.0]},
        "min_separation": 0.1,
    },
    "randomize_scene_light": {
        "intensity_range": [1500.0, 1500.0],
        "color_range": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    },
    "randomize_sphere_light": {
        "intensity_range": [50000.0, 50000.0],
        "color_range": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        "pose_range": {"x": [1.11564, 1.11564], "y": [1.0323, 1.0323], "z": [2.5, 2.5]},
    },
    "randomize_top_camera": {
        "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01], "roll": [-0.01, 0.01], "pitch": [-0.01, 0.01], "yaw": [-0.01, 0.01]},
    },
}
_DEFAULT_SC = {
    "cam_wrist": {
        "height": 480, "width": 640,
        "spawn": {"focal_length": 2.79, "focus_distance": 200.0, "horizontal_aperture": 3.2, "clipping_range": [0.01, 100.0]},
        "offset": {"pos": [0.07041, 0.00104, 0.02851], "rot": [0.66823, 0.27316, -0.2574, -0.64234]},
    },
    "cam_top": {
        "height": 480, "width": 640,
        "spawn": {"focal_length": 16.9157, "focus_distance": 102.3, "horizontal_aperture": 20.955, "clipping_range": [0.01, 100.0]},
        "offset": {"pos": [-0.0203, -0.1053, 0.4543], "rot": {"roll": 23.278100642635437, "pitch": 0.8467805711243336, "yaw": -1.0833502299705997}},
    },
}

try:
    _env_cfg_path = Path(__file__).parents[1] / "env_cfg.json"
    print(f"[INFO] Loading env_cfg.json from: {_env_cfg_path}")
    _cfg = json.loads(_env_cfg_path.read_text())
    _ev = _merge_with_log(_DEFAULT_EV, _cfg.get("events", {}), "events")
    _sc = _merge_with_log(_DEFAULT_SC, _cfg.get("scene", {}), "scene")
except FileNotFoundError:
    print("[env_cfg] env_cfg.json not found, using all defaults for events/scene")
    _ev = _DEFAULT_EV
    _sc = _DEFAULT_SC


@configclass
class EventCfg:
    """Configuration for events."""

    init_omx_arm_pose = EventTerm(
        func=omx_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": _ev["init_omx_arm_pose"]["default_pose"],
        },
    )

    randomize_omx_joint_state = EventTerm(
        func=omx_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": _ev["randomize_omx_joint_state"]["mean"],
            "std": _ev["randomize_omx_joint_state"]["std"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_bottle_positions = EventTerm(
        func=omx_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {k: tuple(v) for k, v in _ev["randomize_bottle_positions"]["pose_range"].items()},
            "min_separation": _ev["randomize_bottle_positions"]["min_separation"],
            "asset_cfgs": [SceneEntityCfg("bottle")],
        },
    )

    randomize_basket_positions = EventTerm(
        func=omx_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {k: tuple(v) for k, v in _ev["randomize_basket_positions"]["pose_range"].items()},
            "min_separation": _ev["randomize_basket_positions"]["min_separation"],
            "asset_cfgs": [SceneEntityCfg("basket")],
        },
    )

    randomize_scene_light = EventTerm(
        func=omx_pick_place_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": tuple(_ev["randomize_scene_light"]["intensity_range"]),
            "color_range": tuple(tuple(c) for c in _ev["randomize_scene_light"]["color_range"]),
            "asset_cfg": SceneEntityCfg("light"),
        },
    )

    randomize_sphere_light = EventTerm(
        func=omx_pick_place_events.randomize_scene_lighting_spherelight,
        mode="reset",
        params={
            "intensity_range": tuple(_ev["randomize_sphere_light"]["intensity_range"]),
            "color_range": tuple(tuple(c) for c in _ev["randomize_sphere_light"]["color_range"]),
            "pose_range": {k: tuple(v) for k, v in _ev["randomize_sphere_light"]["pose_range"].items()},
            "asset_cfg": SceneEntityCfg("SphereLight"),
        },
    )

    randomize_top_camera = EventTerm(
        func=omx_pick_place_events.randomize_camera_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cam_top"),
            "pose_range": {k: tuple(v) for k, v in _ev["randomize_top_camera"]["pose_range"].items()},
            "convention": "ros",
        },
    )


@configclass
class OMXBottlePickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set OMX as robot
        self.scene.robot = OMX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set table
        self.scene.table = OMY_TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")

        self.scene.bottle = PLASTIC_BOTTLE_CFG.replace(prim_path="{ENV_REGEX_NS}/cube")

        self.scene.basket = PLASTIC_BASKET_CFG.replace(prim_path="{ENV_REGEX_NS}/box")

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        self.scene.cam_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMX/link5/cam_wrist",
            update_period=0.0,
            height=_sc["cam_wrist"]["height"],
            width=_sc["cam_wrist"]["width"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_sc["cam_wrist"]["spawn"]["focal_length"],
                focus_distance=_sc["cam_wrist"]["spawn"]["focus_distance"],
                horizontal_aperture=_sc["cam_wrist"]["spawn"]["horizontal_aperture"],
                clipping_range=tuple(_sc["cam_wrist"]["spawn"]["clipping_range"])
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(_sc["cam_wrist"]["offset"]["pos"]),
                rot=_parse_cam_rot(_sc["cam_wrist"]["offset"]["rot"]),
                convention="isaac",
            )
        )
        self.scene.cam_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Table/robotis_omy_table/camera_link/cam_top",
            update_period=0.0,
            height=_sc["cam_top"]["height"],
            width=_sc["cam_top"]["width"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_sc["cam_top"]["spawn"]["focal_length"],
                focus_distance=_sc["cam_top"]["spawn"]["focus_distance"],
                horizontal_aperture=_sc["cam_top"]["spawn"]["horizontal_aperture"],
                clipping_range=tuple(_sc["cam_top"]["spawn"]["clipping_range"])
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(_sc["cam_top"]["offset"]["pos"]),
                rot=_parse_cam_rot(_sc["cam_top"]["offset"]["rot"]),
                convention="isaac",
            )
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMX/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMX/link5",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.09, 0.0, 0.0],
                    ),
                ),
            ],
        )
