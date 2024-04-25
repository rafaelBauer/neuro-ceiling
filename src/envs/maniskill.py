from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

import cv2

from typing import Any, Dict, Union

import gymnasium as gym

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# import gymnasium as gym
# import mani_skill2.envs  # noqa: F401
# from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv
# from mani_skill2.utils.registration import register_env
# import numpy as np
# import sapien.core as sapien
# import torch

from utils.logging import logger

from .environment import BaseEnvironment, BaseEnvironmentConfig
from utils.geometry_np import invert_homogenous_transform, quaternion_to_axis_angle
from utils.misc import invert_dict
from utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)

__ENV_NAME__: str = "NeuroCeilingEnv-v0"

default_cameras = ("hand_camera", "base_camera")

cam_name_tranlation = MappingProxyType(
    {
        "hand_camera": "wrist",
        "base_camera": "base",
        "overhead": "overhead",
        "overhead_camera_0": "overhead_0",
        "overhead_camera_1": "overhead_1",
        "overhead_camera_2": "overhead_2",
    }
)

inv_cam_name_tranlation = invert_dict(cam_name_tranlation)


@dataclass
class ManiSkillEnvironmentConfig(BaseEnvironmentConfig):
    def __init__(self):
        super().__init__("ManiSkill")
        self.headless: bool = False
        self.render_sapien: bool = True


class ManiSkillEnv(BaseEnvironment):
    __RENDER_MODE: str = "human"
    __CONTROL_MODE: str = "pd_ee_delta_pose"

    # -------------------------------------------------------------------------- #
    # Initialization
    # -------------------------------------------------------------------------- #
    def __init__(self, config: ManiSkillEnvironmentConfig) -> None:
        super().__init__(config)
        self.__HEADLESS: bool = config.headless
        self.__render_sapien: bool = config.render_sapien
        self.__env: Optional[BaseEnv] = None

    def start(self):
        kwargs = {
            # "obs_mode": self.obs_mode,
            "control_mode": self.__CONTROL_MODE,
            # "camera_cfgs": self.camera_cfgs,
            # "shader_dir": "rt" if self.CONFIG.real_depth else "ibl",
            "render_mode": self.__RENDER_MODE,
            # "render_camera_cfgs": dict(width=640, height=480)
            # "bg_name": self.bg_name,
            # "model_ids": self.model_ids,                # TODO Ask what is this
            # "max_episode_steps": self.horizon,
            # "fixed_target_link_idx": self.CONFIG.fixed_target_link_idx,   # TODO Ask what is this
            "reward_mode": "sparse",
        }

        # if kwargs["model_ids"] is None:
        #     kwargs.pop("model_ids")  # model_ids only needed for some tasks
        # if kwargs["fixed_target_link_idx"] is None:
        #     kwargs.pop("fixed_target_link_idx")  # only needed for some tasks

        # self.gym_kwargs = kwargs

        self.__env = gym.make(__ENV_NAME__, **kwargs)
        self.reset()

        # if self.seed is not None:
        #     self.__gym_env.seed(self.seed)

    # -------------------------------------------------------------------------- #
    # Observation
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    # def render(self) -> None:
    #     if not self.__HEADLESS:
    #         if self.__render_sapien:
    #             self.__env.render_human()
    #         else:
    #             obs = self.__env.render_cameras()
    #             # self.cam_win_title
    #             cv2.imshow("test", obs)
    #             cv2.waitKey(1)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self):
        super().reset()
        self.__env.reset()

    def reset_joint_pose(self) -> None:
        pass

    def close(self):
        pass

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def _step(self, action: np.ndarray, postprocess: bool = True, delay_gripper: bool = True,
              scale_action: bool = True) -> tuple[SceneObservation, float, bool, dict]:
        """
       Postprocess the action and execute it in the environment.
       Catches invalid actions and executes a zero action instead.

       Parameters
       ----------
       action : np.ndarray
           The raw action predicted by a policy.
       postprocess : bool, optional
           Whether to postprocess the action at all, by default True
       delay_gripper : bool, optional
           Whether to delay the gripper action. Usually needed for ML
           policies, by default True
       scale_action : bool, optional
           Whether to scale the action. Usually needed for ML policies,
           by default True
       invert_xy : bool, optional
           Whether to invert x and y translation. Makes it easier to teleop
           in ManiSkill because of the base camera setup, by default True

       Returns
       -------
       SceneObservation, float, bool, dict
           The observation, reward, done flag and info dict.

       Raises
       ------
       Exception
           Do not yet know how ManiSkill handles invalid actions, so raise
           an exception if it occurs in stepping the action.
       """
        prediction_is_quat = action.shape[0] == 8

        if postprocess:
            action = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
            )
        else:
            action = action

        # if self.invert_xy:
        #     # Invert x, y movement and rotation, but not gripper and z.
        #     action[:2] = -action[:2]
        #     action[3:-2] = -action[3:-2]

        # zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action[-1]])
        zero_action = np.zeros_like(action)

        # if np.isnan(action).any():
        #     logger.warning("NaN action, skipping")
        #     action = zero_action

        # NOTE: if stepping fails might be bcs postprocess_action is deactivated
        # should be used now as it also converts quats predicted by the GMM to

        # try:
        #     next_obs, reward, done, _, info = self.__gym_env.step(action)
        # except Exception as e:
        #     logger.info("Skipping invalid action {}.".format(action))
        #
        #     logger.warning("Don't yet know how ManiSkill handles invalid actions")
        #     raise e

        next_obs, reward, done, _, info = self.__env.step(zero_action)

        # obs: SceneObservation = self.__process_observation(next_obs)
        obs = next_obs

        self.__env.render()

        return obs, reward, done, info

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    def __process_observation(self, obs: dict) -> SceneObservation:
        """
        Convert the observation dict from ManiSkill to a SceneObservation.

        Parameters
        ----------
        obs : dict
            The observation dict from ManiSkill.

        Returns
        -------
        SceneObservation
            The observation in common format as a TensorClass.
        """
        cam_obs = obs["image"]
        cam_names = cam_obs.keys()

        translated_names = [cam_name_tranlation[c] for c in cam_names]
        assert set(self.cameras).issubset(set(translated_names))

        cam_rgb = {
            cam_name_tranlation[c]: cam_obs[c]["Color"][:, :, :3].transpose((2, 0, 1))
            for c in cam_names
        }

        # Negative depth is channel 2 in the position tensor.
        # See https://insiders.vscode.dev/github/vonHartz/ManiSkill2/blob/main/mani_skill2/sensors/depth_camera.py#L100-L101
        cam_depth = {
            cam_name_tranlation[c]: -cam_obs[c]["Position"][:, :, 2] for c in cam_names
        }

        # NOTE channel 0 is mesh-wise, channel 1 is actor-wise, see
        # https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#visualize-segmentation
        cam_mask = {
            cam_name_tranlation[c]: cam_obs[c]["Segmentation"][:, :, 0]
            for c in cam_names
        }

        # Invert extrinsics for consistency with RLBench, Franka. cam2world vs world2cam.
        cam_ext = {
            cam_name_tranlation[c]: invert_homogenous_transform(
                obs["camera_param"][c]["extrinsic_cv"]
            )
            for c in cam_names
        }

        cam_int = {
            cam_name_tranlation[c]: obs["camera_param"][c]["intrinsic_cv"]
            for c in cam_names
        }

        ee_pose = torch.Tensor(obs["extra"]["tcp_pose"])
        object_poses = dict_to_tensordict(
            {
                k: torch.Tensor(v)
                for k, v in obs["extra"].items()
                if k.endswith("pose") and k != "tcp_pose"
            }
        )

        joint_pos = torch.Tensor(obs["agent"]["qpos"])
        joint_vel = torch.Tensor(obs["agent"]["qvel"])

        if joint_pos.shape == torch.Size([7]):
            # For tasks with excavator attached, there's no additional joints
            finger_pose = torch.empty(0)
            finger_vel = torch.empty(0)
        else:
            # NOTE: the last two dims are the individual fingers, but they are
            # forced to be identical.
            # NOTE: switched from using split([7, 2]) (ie enforce 7 joints) to
            # assuming that the last two joints are the fingers and the rest are
            # the arm joints, as mobile manipulation envs seem to have 8 joints.
            joint_pos, finger_pose = joint_pos[:-2], joint_pos[-2:]
            joint_vel, finger_vel = joint_vel[:-2], joint_vel[-2:]

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.cameras)}
            | {
                c: SingleCamObservation(
                    **{
                        "rgb": torch.Tensor(cam_rgb[c]),
                        "depth": torch.Tensor(cam_depth[c]),
                        "mask": torch.Tensor(cam_mask[c].astype(np.uint8)).to(
                            torch.uint8
                        ),
                        "extr": torch.Tensor(cam_ext[c]),
                        "intr": torch.Tensor(cam_int[c]),
                    },
                    batch_size=empty_batchsize,
                )
                for c in self.cameras
            }
        )

        obs = SceneObservation(
            cameras=multicam_obs,
            ee_pose=ee_pose,
            object_poses=object_poses,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gripper_state=finger_pose,
            batch_size=empty_batchsize,
        )

        return obs


"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by what agents/actors are
loaded, how agents/actors are randomly initialized during env resets, how goals are randomized and parameterized in observations, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the poses of all actors, articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do. If followed correctly you can easily build a
task that can simulate on the CPU and be parallelized on the GPU without having to manage GPU memory and parallelization apart from some
code that need to be written in batched mode (e.g. reward, success conditions)

For a minimal implementation of a simple task, check out
mani_skill /envs/tasks/push_cube.py which is annotated with comments to explain how it is implemented
"""


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env(__ENV_NAME__, max_episode_steps=200)
class NeuroCeilingEnv(BaseEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------
    - how is it randomized?
    - how is that randomized?

    Success Conditions
    ------------------
    - what is done to check if this task is solved?

    Visualization: link to a video/gif of the task being solved
    """

    # here you can define a list of robots that this task is built to support and be solved by. This is so that
    # users won't be permitted to use robots not predefined here. If SUPPORTED_ROBOTS is not defined then users can do anything
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    # if you want to say you support multiple robots you can use SUPPORTED_ROBOTS = [["panda", "panda"], ["panda", "fetch"]] etc.

    # to help with programming, you can assert what type of agents are supported like below, and any shared properties of self.agent
    # become available to typecheckers and auto-completion. E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: Union[Panda, Fetch]

    # if you want to do typing for multi-agent setups, use this below and specify what possible tuples of robots are permitted by typing
    # this will then populate agent.agents (list of the instantiated agents) with the right typing
    # agent: MultiAgent[Union[Tuple[Panda, Panda], Tuple[Panda, Panda, Panda]]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2 ** 25, max_rigid_patch_count=2 ** 18
            )
        )

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: dict):
        # this code loads the agent into the current scene. You can usually ignore this function by deleting it or calling the inherited
        # BaseEnv._load_agent function
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        # here you add various objects like actors and articulations. If your task was to push a ball, you may add a dynamic sphere object on the ground
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(self._scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeA")
        self.cubeB = actors.build_cube(self._scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeB")
        self.cubeC = actors.build_cube(self._scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeC")

        # self.goal_site = actors.build_sphere(
        #     self._scene,
        #     radius=self.goal_thresh,
        #     color=[0, 1, 0, 1],
        #     name="goal_site",
        #     body_type="kinematic",
        #     add_collision=False,
        # )
        # self._hidden_objects.append(self.goal_site)

    @property
    def _default_sensor_configs(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )  # sapien_utils.look_at is a utility to get the pose of a camera that looks at a target

        # to see what all the sensors capture in the environment for observations, run env.render_sensors() which returns an rgb array you can visualize
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _setup_sensors(self, options: dict):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors(options)

    def _load_lighting(self, options: dict):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._load_lighting(options)

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cubeA.set_pose(Pose.create_from_pq(xyz, qs))
            #
            # goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def evaluate(self):
        # this function is used primarily to determine success and failure of a task, both of which are optional. If a dictionary is returned
        # containing "success": bool array indicating if the env is in success state or not, that is used as the terminated variable returned by
        # self.step. Likewise if it contains "fail": bool array indicating the opposite (failure state or not) the same occurs. If both are given
        # then a logical OR is taken so terminated = success | fail. If neither are given, terminated is always all False.
        #
        # You may also include additional keys which will populate the info object returned by self.step and that will be given to
        # `_get_obs_extra` and `_compute_dense_reward`. Note that as everything is batched, you must return a batched array of
        # `self.num_envs` booleans (or 0/1 values) for success an dfail as done in the example below
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        # should return an dict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state". Moreover, you have access to the info object
        # which is generated by the `evaluate` function above
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of of self.num_envs rewards as done in the example below.
        # Moreover, you have access to the info object which is generated by the `evaluate` function above
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    def set_state_dict(self, state):
        # this function complements get_state and sets any non simulation state related data correctly so the environment behaves
        # the exact same in terms of output rewards, observations, success etc. should you reset state to a given state and take the same actions
        self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)

# @register_env(__ENV_NAME__, max_episode_steps=200)
# class NeuroCeilingEnv(StationaryManipulationEnv):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
#
#     def evaluate(self, **kwargs) -> dict:
#         is_obj_placed: bool = True
#         is_robot_static = True
#         return dict(
#             is_obj_placed=is_obj_placed,
#             is_robot_static=is_robot_static,
#             success=is_obj_placed and is_robot_static,
#         )
#
#     # Protected methods
#     def _load_actors(self):
#         self._add_ground(render=self.bg_name is None)
#
#         self.box_half_size = np.float32([0.02] * 3)
#         self.cubeA = self._build_cube(self.box_half_size, color=(1, 0, 0), name="cubeA", static=False)
#         self.cubeB = self._build_cube(self.box_half_size, color=(0, 1, 0), name="cubeB", static=False)
#         self.cubeC = self._build_cube(self.box_half_size, color=(0, 0, 1), name="cubeC", static=False)
#
#     def _initialize_actors(self):
#         """Initialize the poses of actors."""
#         z = self.box_half_size[2]
#         cube_a_pose = sapien.Pose([0, 0, z], [0, 0, 0, 1])
#         cube_b_pose = sapien.Pose([0, 0.2, z], [0, 0, 0, 1])
#         cube_c_pose = sapien.Pose([0, -0.2, z], [0, 0, 0, 1])
#
#         self.cubeA.set_pose(cube_a_pose)
#         self.cubeB.set_pose(cube_b_pose)
#         self.cubeC.set_pose(cube_c_pose)


# class ManiSkillEnv(BaseEnvironment):
#     def __init__(self, config: ManiSkillEnvironmentConfig) -> None:
#         super().__init__(config)
#
#         # NOTE: removed ms_config dict. Just put additional kwargs into the
#         # config dict and treat them here and in launch_simulation_env.
#
#         # ManiSkill controllers have action space normalized to [-1,1].
#         # Max speed is a bit fast for teleop, so scale down.
#         self._delta_pos_scale = 0.25
#         self._delta_angle_scale = 0.5
#
#         self.cameras = config.cameras
#         self.cameras_ms = [inv_cam_name_tranlation[c] for c in self.cameras]
#
#         image_size = (self.image_height, self.image_width)
#
#         self.camera_cfgs = {
#             "width": image_size[1],
#             "height": image_size[0],
#             "use_stereo_depth": config.real_depth,
#             "add_segmentation": True,
#             # NOTE: these are examples of how to pass camera params.
#             # Should specify these in the config file.
#             # "overhead": {  # can pass specific params per cam as well
#             #     'p': [0.2, 0, 0.2],
#             #     # Quaternions are [w, x, y, z]
#             #     'q': [7.7486e-07, -0.194001, 7.7486e-07, 0.981001]
#             # },
#             # "base_camera": {
#             #     'p': [0.2, 0, 0.2],
#             #     'q': [0, 0.194, 0, -0.981]  # Quaternions are [w, x, y, z]
#             # }
#         }
#
#         self.extra_cams = []
#
#         for c, pq in config.camera_pose.items():
#             ms_name = inv_cam_name_tranlation[c]
#             if self.camera_cfgs.get(ms_name) is None:
#                 self.camera_cfgs[ms_name] = {}
#             if ms_name not in default_cameras:
#                 self.extra_cams.append(ms_name)
#             self.camera_cfgs[ms_name]["p"] = pq[:3]
#             self.camera_cfgs[ms_name]["q"] = pq[3:]
#
#         self.task_name = config.task
#         self.headless = config.headless_env
#
#         self.gym_env = None
#
#         self.render_sapien = config.render_sapien
#         self.bg_name = config.background
#         self.model_ids = config.model_ids
#
#         self.obs_mode = config.obs_mode
#         self.action_mode = config.action_mode
#
#         self.invert_xy = config.invert_xy
#
#         self.seed = config.seed
#
#         if config.static_env:
#             raise NotImplementedError
#
#         # NOTE: would like to make the horizon configurable, but didn't figure
#         # it out how to make this work with the Maniskill env registry. TODO
#         # self.horizon = -1
#
#         # if self.model_ids is None:
#         #     self.model_ids = []
#
#         if not self.render_sapien and not self.headless:
#             self.cam_win_title = "Observation"
#             self.camera_rgb_window = cv2.namedWindow(
#                 self.cam_win_title, cv2.WINDOW_AUTOSIZE
#             )
#
#         self._patch_register_cameras()
#         self.launch_simulation_env(config)
#
#         self._pin_model = self._create_pin_model()
#
#     def _create_pin_model(self) -> sapien.core.PinocchioModel:
#         return self.agent.controller.articulation.create_pinocchio_model()
#
#     @property
#     def _arm_controller(self) -> mani_skill2.agents.controllers.PDJointPosController:
#         return self.agent.controller.controllers["arm"]
#
#     @property
#     def _ee_link_idx(self) -> int:
#         return self._arm_controller.ee_link_idx  # type: ignore
#
#     @property
#     def _q_mask(self) -> np.ndarray:
#         return self._arm_controller.qmask  # type: ignore
#
#     @property
#     def _move_group(self) -> str:
#         return self.robot.get_links()[self._ee_link_idx].get_name()
#
#     @property
#     def camera_names(self):
#         return tuple(
#             cam_name_tranlation[c] for c in self.gym_env.env._camera_cfgs.keys()
#         )
#
#     @property
#     def _urdf_path(self):
#         return self.agent._get_urdf_path()
#
#     @property
#     def _srdf_path(self):
#         return self.agent._get_srdf_path()
#
#     @property
#     def agent(self):
#         return self.gym_env.agent
#
#     @property
#     def robot(self):
#         return self.agent.robot
#
#     def get_solution_sequence(self):
#         return self.gym_env.env.get_solution_sequence()
#
#     def _patch_register_cameras(self):
#         from mani_skill2.sensors.camera import CameraConfig
#         from mani_skill2.utils.sapien_utils import look_at
#
#         # from sapien.core import Pose as SapienPose
#
#         envs = [
#             mani_skill2.envs.pick_and_place.pick_clutter.PickClutterEnv,
#             mani_skill2.envs.pick_and_place.pick_cube.PickCubeEnv,
#             mani_skill2.envs.pick_and_place.pick_cube.LiftCubeEnv,
#             mani_skill2.envs.pick_and_place.pick_clutter.PickClutterYCBEnv,
#             mani_skill2.envs.pick_and_place.stack_cube.StackCubeEnv,
#             mani_skill2.envs.pick_and_place.pick_single.PickSingleEGADEnv,
#             mani_skill2.envs.pick_and_place.pick_single.PickSingleYCBEnv,
#             # mani_skill2.envs.assembly.assembling_kits.AssemblingKitsEnv,
#             # TODO: for some reason, these two break upon patching
#             # mani_skill2.envs.assembly.peg_insertion_side.PegInsertionSideEnv,
#             # mani_skill2.envs.assembly.plug_charger.PlugChargerEnv
#         ]
#
#         if self.task_name in ["PegInsertionSide-v0", "PlugCharger-v0"]:
#             logger.opt(ansi=True).warning(
#                 f"Skipping camera patching for {self.task_name}. "
#                 "<red>This disables camera customization, including the "
#                 "overhead camera.</red> See code for details."
#             )
#             if "overhead" in self.camera_cfgs:
#                 self.camera_cfgs.pop("overhead")
#
#         def _register_cameras(self):
#             cfgs = _orig_register_cameras(self)
#             if type(cfgs) is CameraConfig:
#                 cfgs = [cfgs]
#             pose = look_at([0, 0, 0], [0, 0, 0])
#             for c in self._extra_camera_names:
#                 if c == "base_camera":
#                     continue
#                 else:
#                     logger.info(f"Registering camera {c}")
#                     cfgs.append(
#                         CameraConfig(c, pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
#                     )
#             return cfgs
#
#         for env in envs:
#             _orig_register_cameras = env._register_cameras
#
#             env._extra_camera_names = self.extra_cams
#             env._register_cameras = _register_cameras
#
#     def launch_simulation_env(self, config):
#         env_name = self.task_name
#
#         kwargs = {
#             "obs_mode": self.obs_mode,
#             "control_mode": self.action_mode,
#             "camera_cfgs": self.camera_cfgs,
#             "shader_dir": "rt" if config.real_depth else "ibl",
#             # "render_camera_cfgs": dict(width=640, height=480)
#             "bg_name": self.bg_name,
#             "model_ids": self.model_ids,
#             # "max_episode_steps": self.horizon,
#             "fixed_target_link_idx": self.config.fixed_target_link_idx,
#             "reward_mode": "sparse",
#         }
#
#         if kwargs["model_ids"] is None:
#             kwargs.pop("model_ids")  # model_ids only needed for some tasks
#         if kwargs["fixed_target_link_idx"] is None:
#             kwargs.pop("fixed_target_link_idx")  # only needed for some tasks
#
#         # NOTE: full list of arguments
#         # obs_mode = None,
#         # control_mode = None,
#         # sim_freq: int = 500,
#         # control_freq: int = 20, That's what I use already.
#         # renderer: str = "sapien",
#         # renderer_kwargs: dict = None,
#         # shader_dir: str = "ibl",
#         # render_config: dict = None,
#         # enable_shadow: bool = False,
#         # camera_cfgs: dict = None,
#         # render_camera_cfgs: dict = None,
#         # bg_name: str = None,
#
#         self.gym_kwargs = kwargs
#
#         self.gym_env = gym.make(env_name, **kwargs)
#
#         if self.seed is not None:
#             self.gym_env.seed(self.seed)
#
#     def make_twin(self, control_mode: str | None = None, obs_mode: str | None = None):
#         kwargs = self.gym_kwargs.copy()
#         if control_mode is not None:
#             kwargs["control_mode"] = control_mode
#         if obs_mode is not None:
#             kwargs["obs_mode"] = obs_mode
#
#         return gym.make(self.task_name, **kwargs)
#
#     def render(self):
#         if not self.headless:
#             if self.render_sapien:
#                 self.gym_env.render_human()
#             else:
#                 obs = self.gym_env.render_cameras()
#                 cv2.imshow(self.cam_win_title, obs)
#                 cv2.waitKey(1)
#
#     def reset(self, **kwargs):
#         super().reset()
#
#         obs, _ = self.gym_env.reset(**kwargs)
#
#         obs = self.process_observation(obs)
#
#         self._pin_model = self._create_pin_model()
#
#         return obs
#
#     def reset_to_demo(self, demo):
#         reset_kwargs = demo["reset_kwargs"]
#         seed = reset_kwargs.pop("seed")
#         return self.reset(seed=seed, options=reset_kwargs)
#
#     def get_seed(self):
#         return self.gym_env.get_episode_seed()
#
#     def get_state(self):
#         return self.gym_env.get_state()
#
#     def set_state(self, state):
#         self.gym_env.set_state(state)
#
#     def _step(
#             self,
#             action: np.ndarray,
#             postprocess: bool = True,
#             delay_gripper: bool = True,
#             scale_action: bool = True,
#     ) -> tuple[SceneObservation, float, bool, dict]:
#         """
#         Postprocess the action and execute it in the environment.
#         Catches invalid actions and executes a zero action instead.
#
#         Parameters
#         ----------
#         action : np.ndarray
#             The raw action predicted by a policy.
#         postprocess : bool, optional
#             Whether to postprocess the action at all, by default True
#         delay_gripper : bool, optional
#             Whether to delay the gripper action. Usually needed for ML
#             policies, by default True
#         scale_action : bool, optional
#             Whether to scale the action. Usually needed for ML policies,
#             by default True
#         invert_xy : bool, optional
#             Whether to invert x and y translation. Makes it easier to teleop
#             in ManiSkill because of the base camera setup, by default True
#
#         Returns
#         -------
#         SceneObservation, float, bool, dict
#             The observation, reward, done flag and info dict.
#
#         Raises
#         ------
#         Exception
#             Do not yet know how ManiSkill handles invalid actions, so raise
#             an exception if it occurs in stepping the action.
#         """
#         prediction_is_quat = action.shape[0] == 8
#
#         if postprocess:
#             action = self.postprocess_action(
#                 action,
#                 scale_action=scale_action,
#                 delay_gripper=delay_gripper,
#                 prediction_is_quat=prediction_is_quat,
#             )
#         else:
#             action = action
#
#         if self.invert_xy:
#             # Invert x, y movement and rotation, but not gripper and z.
#             action[:2] = -action[:2]
#             action[3:-2] = -action[3:-2]
#
#         # zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action[-1]])
#         zero_action = np.zeros_like(action)
#
#         if np.isnan(action).any():
#             logger.warning("NaN action, skipping")
#             action = zero_action
#
#         # NOTE: if stepping fails might be bcs postprocess_action is deactivated
#         # should be used now as it also converts quats predicted by the GMM to
#
#         try:
#             next_obs, reward, done, _, info = self.gym_env.step(action)
#         except Exception as e:
#             logger.info("Skipping invalid action {}.".format(action))
#
#             logger.warning("Don't yet know how ManiSkill handles invalid actions")
#             raise e
#
#             next_obs, reward, done, _, info = self.gym_env.step(zero_action)
#
#         obs = self.process_observation(next_obs)
#
#         self.render()
#
#         return obs, reward, done, info
#
#     def close(self):
#         self.gym_env.close()
#
#     def process_observation(self, obs: dict) -> SceneObservation:
#         """
#         Convert the observation dict from ManiSkill to a SceneObservation.
#
#         Parameters
#         ----------
#         obs : dict
#             The observation dict from ManiSkill.
#
#         Returns
#         -------
#         SceneObservation
#             The observation in common format as a TensorClass.
#         """
#         cam_obs = obs["image"]
#         cam_names = cam_obs.keys()
#
#         translated_names = [cam_name_tranlation[c] for c in cam_names]
#         assert set(self.cameras).issubset(set(translated_names))
#
#         cam_rgb = {
#             cam_name_tranlation[c]: cam_obs[c]["Color"][:, :, :3].transpose((2, 0, 1))
#             for c in cam_names
#         }
#
#         # Negative depth is channel 2 in the position tensor.
#         # See https://insiders.vscode.dev/github/vonHartz/ManiSkill2/blob/main/mani_skill2/sensors/depth_camera.py#L100-L101
#         cam_depth = {
#             cam_name_tranlation[c]: -cam_obs[c]["Position"][:, :, 2] for c in cam_names
#         }
#
#         # NOTE channel 0 is mesh-wise, channel 1 is actor-wise, see
#         # https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#visualize-segmentation
#         cam_mask = {
#             cam_name_tranlation[c]: cam_obs[c]["Segmentation"][:, :, 0]
#             for c in cam_names
#         }
#
#         # Invert extrinsics for consistency with RLBench, Franka. cam2world vs world2cam.
#         cam_ext = {
#             cam_name_tranlation[c]: invert_homogenous_transform(
#                 obs["camera_param"][c]["extrinsic_cv"]
#             )
#             for c in cam_names
#         }
#
#         cam_int = {
#             cam_name_tranlation[c]: obs["camera_param"][c]["intrinsic_cv"]
#             for c in cam_names
#         }
#
#         ee_pose = torch.Tensor(obs["extra"]["tcp_pose"])
#         object_poses = dict_to_tensordict(
#             {
#                 k: torch.Tensor(v)
#                 for k, v in obs["extra"].items()
#                 if k.endswith("pose") and k != "tcp_pose"
#             }
#         )
#
#         joint_pos = torch.Tensor(obs["agent"]["qpos"])
#         joint_vel = torch.Tensor(obs["agent"]["qvel"])
#
#         if joint_pos.shape == torch.Size([7]):
#             # For tasks with excavator attached, there's no additional joints
#             finger_pose = torch.empty(0)
#             finger_vel = torch.empty(0)
#         else:
#             # NOTE: the last two dims are the individual fingers, but they are
#             # forced to be identical.
#             # NOTE: switched from using split([7, 2]) (ie enforce 7 joints) to
#             # assuming that the last two joints are the fingers and the rest are
#             # the arm joints, as mobile manipulation envs seem to have 8 joints.
#             joint_pos, finger_pose = joint_pos[:-2], joint_pos[-2:]
#             joint_vel, finger_vel = joint_vel[:-2], joint_vel[-2:]
#
#         multicam_obs = dict_to_tensordict(
#             {"_order": CameraOrder._create(self.cameras)}
#             | {
#                 c: SingleCamObservation(
#                     **{
#                         "rgb": torch.Tensor(cam_rgb[c]),
#                         "depth": torch.Tensor(cam_depth[c]),
#                         "mask": torch.Tensor(cam_mask[c].astype(np.uint8)).to(
#                             torch.uint8
#                         ),
#                         "extr": torch.Tensor(cam_ext[c]),
#                         "intr": torch.Tensor(cam_int[c]),
#                     },
#                     batch_size=empty_batchsize,
#                 )
#                 for c in self.cameras
#             }
#         )
#
#         obs = SceneObservation(
#             cameras=multicam_obs,
#             ee_pose=ee_pose,
#             object_poses=object_poses,
#             joint_pos=joint_pos,
#             joint_vel=joint_vel,
#             gripper_state=finger_pose,
#             batch_size=empty_batchsize,
#         )
#
#         return obs
#
#     def get_replayed_obs(self):
#         # To be used from extract_demo.py
#         obs = self.gym_env._episode_data[0]["o"]
#         print(obs)
#         done = self.gym_env._episode_data[0]["d"]
#         reward = self.gym_env._episode_data[0]["r"]
#         info = self.gym_env._episode_data[0]["info"]
#
#     def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
#         return quaternion_to_axis_angle(quaternion)
#
#     def get_inverse_kinematics(
#             self,
#             target_pose: np.ndarray,
#             reference_qpos: np.ndarray,
#             max_iterations: int = 100,
#     ) -> np.ndarray:
#         qpos, success, error = self._pin_model.compute_inverse_kinematics(
#             self._ee_link_idx,
#             sapien.core.Pose(target_pose[:3], target_pose[3:7]),
#             initial_qpos=reference_qpos,
#             active_qmask=self._q_mask,
#             max_iterations=max_iterations,
#         )
#
#         if not success:
#             raise ValueError(f"Failed to find IK solution: {error}")
#
#         return qpos
#
#     def get_forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
#         self._pin_model.compute_forward_kinematics(qpos)
#
#         pose = self._pin_model.get_link_pose(self._ee_link_idx)
#
#         return np.concatenate([pose.p, pose.q])
#
#     def reset_joint_pose(
#             self,
#             joint_pos=[
#                 -8.2433e-03,
#                 4.3171e-01,
#                 -2.0684e-03,
#                 -1.9697e00,
#                 -7.5249e-04,
#                 2.3248e00,
#                 8.0096e-01,
#                 0.04,
#                 0.04,
#             ],
#     ) -> None:
#         self.robot.set_qpos(joint_pos)
