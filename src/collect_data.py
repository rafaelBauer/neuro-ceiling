import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Register ManiSkill2 environments in gym
import mani_skill2.envs
import sapien.core as sapien

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.envs.pick_and_place.pick_single import build_actor_ycb
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at


def plot_img(img, title=None):
    plt.figure(figsize=(10, 6))
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.show()


@register_env("NeuroCeilingEnv-v0", max_episode_steps=200, override=True)
class NeuroCeilingEnv(PickCubeEnv):
    def _load_actors(self):
        build_actor_ycb()
        # Load YCB objects
        # It is the same as in PickSingleYCB-v0, just for illustration here
        builder = self._scene.create_actor_builder()
        model_dir = ASSET_DIR / "mani_skill2_ycb/models/077_rubiks_cube"
        scale = self.cube_half_size / 0.01887479572529618
        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_collisions_from_file(
            filename=collision_file, scale=scale, density=1000
        )
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=scale)
        self.obj = builder.build(name="apple" + str(i))

        # Add a goal indicator (visual only)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

        # -------------------------------------------------------------------------- #
        # Load static scene
        # -------------------------------------------------------------------------- #
        builder = self._scene.create_actor_builder()
        path = f"{ASSET_DIR}/hab2_bench_assets/stages/Baked_sc1_staging_00.glb"
        pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        # NOTE: use nonconvex collision for static scene
        builder.add_nonconvex_collision_from_file(path, pose)
        builder.add_visual_from_file(path, pose)
        self.arena = builder.build_static()
        # Add offset so that the workspace is on the table
        offset = np.array([-2.0616, -3.1837, 0.66467 + 0.095])
        self.arena.set_pose(sapien.Pose(-offset))

    def initialize_episode(self):
        super().initialize_episode()

        # Rotate the robot for better visualization
        self.agent.robot.set_pose(
            sapien.Pose([0, -0.56, 0], [0.707, 0, 0, 0.707])
        )

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.p = cam_cfg.p + [0, 0, -0.35]
        cam_cfg.fov = 1
        return cam_cfg


env = gym.make("NeuroCeilingEnv-v0")
plot_img(env.unwrapped.render_rgb_array())
env.close()
del env
