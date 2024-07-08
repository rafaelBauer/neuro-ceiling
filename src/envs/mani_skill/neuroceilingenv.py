"""
At a high-level, ManiSkill tasks can minimally be defined by what agents/actors are
loaded, how agents/actors are randomly initialized during env resets, how goals are randomized and parameterized in
observations, and success conditions

Environment reset comprises running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
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

import random
from typing import Union, Any, Dict, Final

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

__ENV_NAME__: str = "NeuroCeilingEnv-v0"

from envs.taskconfig import TaskConfig
from utils.pose import RotationRepresentation
import utils.pose as pose_utils


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do
# gym.make("CustomEnv-v0")
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
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    # if you want to say you support multiple robots you can use SUPPORTED_ROBOTS = [["panda", "panda"], ["panda", "fetch"]] etc.

    # to help with programming, you can assert what type of agents are supported like below, and any shared properties of self.agent
    # become available to typecheckers and auto-completion. E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: Union[Panda, PandaWristCam, Fetch]

    # if you want to do typing for multi-agent setups, use this below and specify what possible tuples of robots are permitted by typing
    # this will then populate agent.agents (list of the instantiated agents) with the right typing
    # agent: MultiAgent[Union[Tuple[Panda, Panda], Tuple[Panda, Panda, Panda]]]
    cube_half_size = 0.02
    goal_thresh = 0.01

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        task_config=TaskConfig(_target_objects_pose={}, _initial_objects={}, _available_spots_pose={}),
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # "task_config" is a custom field added, so _load_scene and _load_scene and _initialize_episode methods can
        # create the objects at specific poses as well as be aware of the goal
        self._task_config = task_config
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_cfg=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))

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
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.actors: {str, Actor} = {}
        self.spots: {str, Actor} = {}

        # Add the cubes
        for number, (name, obj) in enumerate(self._task_config.initial_objects.items()):
            self.actors[name] = actors.build_cube(self.scene, half_size=self.cube_half_size, color=obj.color, name=name)
            spot_name = "Spot " + str(number)
            spot_color = obj.color.copy()
            spot_color[-1] = 0
            self.spots[spot_name] = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=spot_color,
                name=spot_name,
                body_type="kinematic",
                add_collision=False,
            )
            self._hidden_objects.append(self.spots[spot_name])

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
            self.table_scene.initialize(env_idx)

            objects_shuffled = list(self._task_config.initial_objects.values())
            random.shuffle(objects_shuffled)

            # robot_pose in w.r.t to the robot world frame
            robot_pose: Final[pose_utils.Pose] = pose_utils.Pose(obj=self.agent.robot.pose.sp)
            for number, (name, obj) in enumerate(self._task_config.initial_objects.items()):
                obj_pose: pose_utils.Pose = pose_utils.Pose(obj=robot_pose * objects_shuffled[number].pose)
                self.actors[name].set_pose(
                    Pose.create(obj_pose.to_tensor(rotation_representation=RotationRepresentation.QUATERNION))
                )

            for number, (name, spot) in enumerate(self._task_config.available_spots_pose.items()):
                spot_pose: pose_utils.Pose = pose_utils.Pose(obj=robot_pose * spot)
                spot_name = "Spot " + str(number)
                self.spots[spot_name].set_pose(
                    Pose.create(spot_pose.to_tensor(rotation_representation=RotationRepresentation.QUATERNION))
                )

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

        robot_pose: Final[pose_utils.Pose] = pose_utils.Pose(obj=self.agent.robot.pose.sp)
        target_positions = torch.zeros(len(self._task_config.target_objects_pose), device=self.device, dtype=bool)

        for i, (name, target_pose) in enumerate(self._task_config.target_objects_pose.items()):
            actor_pose: pose_utils.Pose = pose_utils.Pose(
                obj=(robot_pose.inv() * pose_utils.Pose(obj=self.actors[name].pose.sp))
            )
            # Make sure that the actor is close to the target pose
            target_positions[i] = target_pose.is_close(actor_pose, atol=self.goal_thresh)
        return {
            "success": target_positions.all(),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        # should return a dict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state". Moreover, you have access to the info object
        # which is generated by the `evaluate` function above

        objects = {}
        spots = {}
        robot_pose: Final[pose_utils.Pose] = pose_utils.Pose(obj=self.agent.robot.pose.sp)

        for name, actor in self.actors.items():
            actor_pose: pose_utils.Pose = pose_utils.Pose(obj=(robot_pose.inv() * pose_utils.Pose(obj=actor.pose.sp)))
            objects[name] = actor_pose.to_tensor(rotation_representation=RotationRepresentation.EULER)

        for name, spot in self.spots.items():
            spot_pose: pose_utils.Pose = pose_utils.Pose(obj=(robot_pose.inv() * pose_utils.Pose(obj=spot.pose.sp)))
            spots[name] = spot_pose.to_tensor(rotation_representation=RotationRepresentation.EULER)

        return {
            "objects": objects,
            "spots": spots,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of of self.num_envs rewards as done in the example below.
        # Moreover, you have access to the info object which is generated by the `evaluate` function above
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
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
