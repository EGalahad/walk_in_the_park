from typing import Optional, Dict

import gym
import numpy as np
from dm_control import composer
from dmcgym import DMCGYM
from gym import spaces
from gym.wrappers import FlattenObservation

import sim
from filter import ActionFilterWrapper
from sim.robots import A1
from sim.tasks import Run

# @torch.jit.script
# def quat_rotate(q, v):
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * \
#         torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
#             shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a + b + c


# @torch.jit.script
# def quat_rotate_inverse(q, v):
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * \
#         torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
#             shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a - b + c


def quat_rotate(quat, v, inverse=False):
    """
    rotate v using a quaternion to projected gravity vector.
    Initially quat = [1, 0, 0, 0], projected gravity = [0, 0, -1]
    """
    q_w = quat[0]
    q_vec = quat[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v, axis=-1) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c if not inverse else a - b + c


class ObsMujoco2Isaac(gym.ObservationWrapper):
    gravity_vector = np.asarray([0, 0, -1])

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "base_lin_vel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "base_ang_vel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "projected_gravity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "commands": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "dof_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
                "dof_vel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
                "actions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
            }
        )

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
        
    command_scale = np.asarray([obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel])

    default_dof_pos = np.asarray(
        [0.05, 0.7, -1.4, 0.05, 0.7, -1.4, 0.05, 0.7, -1.4, 0.05, 0.7, -1.4]
    )

    def observation(self, observation: Dict):
        """
        The observation from Mujoco is a Dict with the following keys and shape:
            'a1_description/joints_pos': 12
            'a1_description/joints_vel': 12
            'a1_description/prev_action': 12
            'a1_description/sensors_framequat': 4
            'a1_description/sensors_gyro': 3
            'a1_description/sensors_velocimeter': 3
        with total dim = 46
        The observation from Isaac is a Dict with the following keys and shape:
            'base_lin_vel': 3
            'base_ang_vel': 3
            'projected_gravity': 3
            'commands': 3/4 lin_vel_x, lin_vel_y, ang_vel_yaw and heading
            'dof_pos': 12
            'dof_vel': 12
            'actions': 12
        with total dim = 48
            class normalization:
                class obs_scales:
                    lin_vel = 2.0
                    ang_vel = 0.25
                    dof_pos = 1.0
                    dof_vel = 0.05
                    height_measurements = 5.0
                clip_observations = 100.
                clip_actions = 100.
        """
        base_lin_vel = observation["a1_description/sensors_velocimeter"]
        base_ang_vel = observation["a1_description/sensors_gyro"]
        projected_gravity = quat_rotate(
            observation["a1_description/sensors_framequat"],
            self.gravity_vector,
            inverse=True,
        )
        move_speed = 0.5
        commands = np.asarray([move_speed * 1.5, 0, 0])
        dof_pos = observation["a1_description/joints_pos"]
        dof_vel = observation["a1_description/joints_vel"]
        actions = observation["a1_description/prev_action"]
        return {
            "base_lin_vel": base_lin_vel * self.obs_scales.lin_vel,
            "base_ang_vel": base_ang_vel * self.obs_scales.ang_vel,
            "projected_gravity": projected_gravity,
            "commands": commands * self.command_scale,
            "dof_pos": (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            "dof_vel": dof_vel * self.obs_scales.dof_vel,
            "actions": actions,
        }
        # return np.concatenate(
        #     [
        #         base_lin_vel,
        #         base_ang_vel,
        #         projected_gravity,
        #         commands,
        #         dof_pos,
        #         dof_vel,
        #         actions,
        #     ]
        # )


class ClipAction(gym.ActionWrapper):
    def __init__(self, env, min_action, max_action):
        super().__init__(env)

        min_action = np.asarray(min_action)
        max_action = np.asarray(max_action)

        min_action = min_action + np.zeros(
            env.action_space.shape, dtype=env.action_space.dtype
        )

        max_action = max_action + np.zeros(
            env.action_space.shape, dtype=env.action_space.dtype
        )

        min_action = np.maximum(min_action, env.action_space.low)
        max_action = np.minimum(max_action, env.action_space.high)

        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


def make_env(
    task_name: str,
    control_frequency: int = 33,
    randomize_ground: bool = True,
    action_history: int = 1,
):
    robot = A1(action_history=action_history)
    # robot.kd = 5

    if task_name == "A1Run-v0":
        task = Run(
            robot,
            control_timestep=round(1.0 / control_frequency, 3),
            randomize_ground=randomize_ground,
        )
    else:
        raise NotImplemented

    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    env = DMCGYM(env)
    env = ObsMujoco2Isaac(env)
    env = FlattenObservation(env)

    return env


make_env.metadata = DMCGYM.metadata


def make_mujoco_env(
    env_name: str,
    control_frequency: int,
    clip_actions: bool = True,
    action_filter_high_cut: Optional[float] = -1,
    action_history: int = 1,
) -> gym.Env:
    env = make_env(
        env_name, control_frequency=control_frequency, action_history=action_history
    )

    env = gym.wrappers.TimeLimit(env, 400)

    env = gym.wrappers.ClipAction(env)

    if action_filter_high_cut is not None:
        env = ActionFilterWrapper(env, highcut=action_filter_high_cut)

    if clip_actions:
        ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)
        INIT_QPOS = sim.robots.a1.A1._INIT_QPOS
        if env.action_space.shape[0] == 12:
            env = ClipAction(env, INIT_QPOS - ACTION_OFFSET, INIT_QPOS + ACTION_OFFSET)
        else:
            env = ClipAction(
                env,
                np.concatenate([INIT_QPOS - ACTION_OFFSET, [-1.0]]),
                np.concatenate([INIT_QPOS + ACTION_OFFSET, [1.0]]),
            )

    return env
