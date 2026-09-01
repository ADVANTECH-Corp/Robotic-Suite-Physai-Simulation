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

"""Script to run inference with robotis_lab environments (OMX / SO101)."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inference script for robotis_lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--robot_type", type=str, default="OMX", choices=['OMX', 'SO101'], help="Type of robot to use for teleoperation.")
parser.add_argument("--num_trials", type=int, default=0, help="Stop after N episodes and report the success rate. 0 = run forever.")
parser.add_argument("--num_success", type=int, default=0, help="Stop early once N episodes have succeeded. 0 = disabled.")
parser.add_argument("--max_trials", type=int, default=0, help="Hard cap on episodes, so --num_success can never run forever. 0 = no cap.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if bool(args_cli.num_success) != bool(args_cli.max_trials):
    parser.error("--num_success and --max_trials must be set together (the cap keeps the success target from running forever).")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robotis_lab

class RateLimiter:
    """Simple class for enforcing a loop frequency."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration

def main():
    # env config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.init_action_cfg("inference")
    env_cfg.seed = args_cli.seed

    # create env
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # teleop interface
    if args_cli.robot_type == "OMX":
        from dds_sdk.omx_sdk import OMXSdk
        teleop_interface = OMXSdk(env, mode='inference')
    elif args_cli.robot_type == "SO101":
        from dds_sdk.so101_sdk import SO101Sdk
        teleop_interface = SO101Sdk(env, mode='inference')
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.robot_type}'. Supported: 'OMX', 'SO101'."
        )

    # reset env
    env.reset()
    teleop_interface.reset()
    rate_limiter = RateLimiter(args_cli.step_hz)

    # success-rate counters; env auto-resets on success / bottle drop / time_out.
    # entirely inert unless one of the eval flags is given
    eval_enabled = bool(args_cli.num_trials or args_cli.num_success or args_cli.max_trials)
    has_success_term = "success" in env.termination_manager.active_terms
    trials = 0
    successes = 0

    print("[INFO] Inference loop started. Press 'R' to reset environment.")
    should_reset_task = False
    def reset_task():
        nonlocal should_reset_task
        should_reset_task = True

    teleop_interface.add_callback("R", reset_task)

    while simulation_app.is_running():
        with torch.inference_mode():
            # Always publish observations (images and joint states)
            teleop_interface.publish_observations()
            actions = teleop_interface.get_action()

            if should_reset_task:
                print("[INFO] Reset requested.")
                should_reset_task = False
                env.reset()
                continue

            elif actions is None:
                env.render()
            else:
                if isinstance(actions, dict):
                    # Handle dictionary actions (like reset)
                    if "reset" in actions:
                        # This is a reset action, don't step the environment
                        env.render()
                        continue
                else:
                    # Handle tensor actions
                    if actions.ndim == 1:
                        actions = actions.unsqueeze(0)
                    env.step(actions)

                    if eval_enabled and env.termination_manager.dones[0].item():
                        trials += 1
                        ok = has_success_term and env.termination_manager.get_term("success")[0].item()
                        successes += int(ok)
                        print(
                            f"[EVAL] trial {trials}/{args_cli.num_trials or '-'}: "
                            f"{'SUCCESS' if ok else 'FAIL'} | "
                            f"{successes}/{trials} = {successes / trials:.1%}"
                        )
                        if (
                            (args_cli.num_trials and trials >= args_cli.num_trials)
                            or (args_cli.max_trials and trials >= args_cli.max_trials)
                            or (args_cli.num_success and successes >= args_cli.num_success)
                        ):
                            break
            if rate_limiter:
                rate_limiter.sleep(env)

    if trials:
        print(f"[EVAL] final success rate: {successes}/{trials} = {successes / trials:.1%}")
        # stopping on a success target biases k/n upward (the last trial is always a
        # success by construction); (k-1)/(n-1) is the unbiased estimate for that stop rule
        if args_cli.num_success and successes >= args_cli.num_success and trials > 1:
            print(f"[EVAL] unbiased estimate: {(successes - 1) / (trials - 1):.1%}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
