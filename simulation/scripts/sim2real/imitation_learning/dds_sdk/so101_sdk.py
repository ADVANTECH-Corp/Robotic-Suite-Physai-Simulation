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

import os
import math
import threading
import torch
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections.abc import Callable

try:
    from pynput.keyboard import Listener as KeyboardListener
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _KEYBOARD_AVAILABLE = False
from datetime import datetime

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.builtin_interfaces.msg import Time_

from robotis_dds_python.tools.topic_manager import TopicManager


# Joint names that match /leader/joint_trajectory message published by the
# SO-101 leader (see backend/open_manipulator/so101_ros2_bridge).
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Leave a gap when the leader gripper is (near) fully closed.
# The leader reports the gripper as a 0-100 percentage faked into radians
# (math.radians(pct); 0 = fully closed, 100 = open). We never forward a gripper
# command below this percentage of travel, so the sim gripper stops short of
# closing all the way. With the env gripper remap (scale 1.1, offset
# radians(-10)) the resulting sim joint angle is (1.1 * pct - 10) deg, e.g.
# 10% -> ~1 deg of opening above the -10 deg hard-closed limit. Raise this value
# to leave more space.
# NOTE: like omx_sdk's 0.12 gripper clamp, this floor is baked into the action
# returned by get_action and therefore into the recorded hdf5 dataset (the
# gripper's bottom range collapses to this value). Correct during
# dataset->lerobot conversion if the raw leader command is needed there.
SO101_GRIPPER_MIN_OPEN_PCT = 15.0
SO101_GRIPPER_WIRE_MIN = math.radians(SO101_GRIPPER_MIN_OPEN_PCT)


class SO101Sdk:
    """SO101Sdk class for DDS teleoperation and publishing robot state/images.

    Mirrors OMXSdk but is tailored to the SO-101 follower arm:
      * 6 joints (5 arm + 1 gripper)
      * /leader/joint_trajectory is in radians (SI), like the OMX convention,
        so values are forwarded to IsaacLab raw with no unit conversion. The
        degree<->radian handling lives in the so101_ros2_bridge nodes.
    """

    def __init__(self, env, mode: str):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.joint_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self.lock = threading.Lock()  # Protect shared state

        # Initialize current joint state - will be updated only when commands are received
        self.current_joint_state = {}

        # Joint names exposed by IsaacLab articulation (see SO101.py asset cfg).
        self.joint_names = list(SO101_JOINT_NAMES)
        self.exclude_joints = []

        # DDS Topic Manager
        topic_manager = TopicManager(domain_id=self.domain_id)

        # Subscribers
        self.joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="leader/joint_trajectory",
            topic_type=JointTrajectory_
        )

        # Publishers
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name="joint_states",
            topic_type=JointState_
        )
        self.top_cam_writer = topic_manager.topic_writer(
            topic_name="camera/cam_top/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )
        self.wrist_cam_writer = topic_manager.topic_writer(
            topic_name="camera/cam_wrist/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )

        # Start subscriber thread
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

        # Keyboard listener (only available when X display is present)
        if _KEYBOARD_AVAILABLE:
            self.listener = KeyboardListener(on_press=self._on_press)
            self.listener.start()
        else:
            print("[INFO] Keyboard not available, use HTTP commands instead")

        # HTTP command server (allows remote key triggering, e.g. via WebRTC browser)
        self._http_port = int(os.getenv("SO101_CMD_PORT", os.getenv("OMX_CMD_PORT", 9999)))
        self._start_http_server()

        self._keyboard_controls()

    # ----------------------
    # HTTP command server
    # ----------------------
    def _start_http_server(self):
        sdk = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                key = self.path.strip("/").lower()
                if key in ("b", "r", "n"):
                    sdk._simulate_key(key)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(f"key '{key}' triggered\n".encode())
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"unknown key. use /b /r /n\n")

            def log_message(self, format, *args):
                pass  # suppress access logs

        server = HTTPServer(("0.0.0.0", self._http_port), _Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[CMD] HTTP command server listening on port {self._http_port}")
        print(f"[CMD]   curl -X POST http://localhost:{self._http_port}/b  -> Start recording")
        print(f"[CMD]   curl -X POST http://localhost:{self._http_port}/n  -> Save episode")
        print(f"[CMD]   curl -X POST http://localhost:{self._http_port}/r  -> Reset (discard)")

    def _simulate_key(self, key: str):
        """Simulate a key press programmatically (same logic as _on_press)."""
        if self.mode == 'record':
            if key == 'b':
                self._started = True
                self._reset_state = False
            elif key == 'r':
                self._started = False
                self._reset_state = True
                self._call_callback("R")
            elif key == 'n':
                self._started = False
                self._reset_state = True
                self._call_callback("N")
        elif self.mode == 'inference':
            if key == 'b':
                self._started = True
                self._reset_state = False
            elif key == 'r':
                self._started = False
                self._reset_state = True
                self._call_callback("R")

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the robot:")
        if self.mode == 'record':
            print("[N] Save successful episode and proceed to the next one")
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start recording the current episode")
        elif self.mode == 'inference':
            print("[N] Save successful episode and proceed to the next one")
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start/Resume robot control")

    def _on_press(self, key):
        try:
            if self.mode == 'record':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
                elif key.char == 'n':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("N")
            elif self.mode == 'inference':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    # ----------------------
    # Subscriber loop
    # ----------------------
    def _subscriber_loop(self):
        """Continuously read joint trajectory commands from the leader.

        The leader publishes positions in radians (SI) on
        /leader/joint_trajectory, so we forward them raw — IsaacLab joint
        targets are already in radians (same as OMXSdk).
        """
        try:
            while self.running:
                for msg in self.joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            # The /leader/joint_trajectory wire is already in
                            # radians (SI), matching the OMX convention, so
                            # forward raw with no unit conversion (same as
                            # OMXSdk). Degree<->radian handling lives at the
                            # hardware boundary in the so101_ros2_bridge nodes.
                            self.joint_trajectory_cmd = [
                                float(joint_dict.get(name, 0.0))
                                for name in self.joint_names
                            ]
        except Exception as e:
            print("Subscriber thread exception:", e)
        finally:
            try:
                self.joint_trajectory_reader.Close()
            except:
                pass
            print("Subscriber closed")

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_joint_states(self):
        """Publish current joint states over DDS."""
        now = datetime.now()
        stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
        header = Header_(stamp=stamp, frame_id="base_link")

        obs_joint_name = self.env.scene["robot"].data.joint_names
        all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        all_velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        all_efforts = [0.0] * len(all_positions)

        # Flatten nested lists if necessary
        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]
        if isinstance(all_velocities[0], list):
            all_velocities = [v for sub in all_velocities for v in sub]

        # Get indices of the joints we care about
        indices = [obs_joint_name.index(name) for name in self.joint_names]

        positions = [all_positions[i] for i in indices]
        velocities = [all_velocities[i] for i in indices]
        efforts = [all_efforts[i] for i in indices]

        joint_state = JointState_(
            header=header,
            name=list(self.joint_names),
            position=positions,
            velocity=velocities,
            effort=efforts
        )

        try:
            self.joint_state_writer.write(joint_state)
        except Exception as e:
            print("[Writer] write error:", e)

    def _publish_camera(self, cam_name: str):
        """Publish camera image as DDS compressed image."""
        try:
            cam_data = self.env.scene[cam_name].data
            img = cam_data.output['rgb'][0].cpu().numpy()  # Convert tensor to numpy
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', img)
            jpeg_bytes = buffer.tobytes()

            now = datetime.now()
            stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
            header = Header_(stamp=stamp, frame_id="camera_frame")

            msg = CompressedImage_(header=header, format="jpeg", data=jpeg_bytes)
            if cam_name == "cam_wrist":
                self.wrist_cam_writer.write(msg)
            elif cam_name == "cam_top":
                self.top_cam_writer.write(msg)
        except Exception as e:
            print("Camera publish error:", e)

    # ----------------------
    # Action/state handling
    # ----------------------
    def _compute_action_state(self):
        """Compute current action dictionary based on keyboard input and subscriber."""
        state = {'reset': self._reset_state, 'started': self._started}
        if state['reset']:
            self._reset_state = False
            return state
        state['joint_state'] = self._get_device_state()
        return state

    def _get_device_state(self):
        """Return latest joint positions, starting with current robot state and updating with received commands."""
        with self.lock:
            # Start with current robot joint positions
            obs_joint_name = self.env.scene["robot"].data.joint_names
            all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()

            # Flatten nested lists if necessary
            if isinstance(all_positions[0], list):
                all_positions = [p for sub in all_positions for p in sub]

            # Build joint state from current robot state
            joint_state = {}
            for name in self.joint_names:
                if name in obs_joint_name:
                    idx = obs_joint_name.index(name)
                    joint_state[name] = all_positions[idx]
                else:
                    joint_state[name] = 0.0  # Fallback only if joint not found in robot

            # Update with received commands if available
            if self.joint_trajectory_cmd:
                cmd_dict = dict(zip(self.joint_names, self.joint_trajectory_cmd))
                joint_state.update(cmd_dict)
            return joint_state

    def get_action(self):
        """Return action tensor for robot control."""
        action = self._compute_action_state()
        if action['reset']:
            return {"reset": True}
        if not action['started']:
            return None

        joint_state = action['joint_state']

        # Explicitly define the order sent to simulation (must match env config)
        action_names = list(SO101_JOINT_NAMES)
        positions = [joint_state.get(name, 0.0) for name in action_names]

        # Floor the gripper so the leader's fully-closed position still leaves a
        # gap in the sim (mirrors omx_sdk's gripper clamp; see
        # SO101_GRIPPER_WIRE_MIN).
        gripper_idx = action_names.index("gripper")
        if positions[gripper_idx] < SO101_GRIPPER_WIRE_MIN:
            positions[gripper_idx] = SO101_GRIPPER_WIRE_MIN

        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        """Publish joint states and camera images."""
        self._publish_joint_states()
        self._publish_camera("cam_top")
        self._publish_camera("cam_wrist")  # cam_wrist not yet configured for SO101

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        """Stop threads and close DDS publishers/subscribers."""
        self.running = False
        self.thread.join()
        for obj in [self.joint_trajectory_reader, self.joint_state_writer,
                    self.top_cam_writer, self.wrist_cam_writer]:
            try:
                obj.Close()
            except:
                pass
        print("SO101Sdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Add callback function for a specific key."""
        self._additional_callbacks[key] = func
