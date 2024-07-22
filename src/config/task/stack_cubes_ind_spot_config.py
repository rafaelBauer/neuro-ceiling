from envs.taskconfig import TaskConfig

# Poses must be a pose using quaternion and w.r.t to the robot base frame
_initial_objects = {
    "Cube A": {"pose": [0.615, -0.2, 0.02, 0, 1, 0, 0]},  # Red
    "Cube B": {"pose": [0.615, 0.0, 0.02, 0, 1, 0, 0], "color": [0, 1, 0, 1]},  # Green
    "Cube C": {"pose": [0.615, 0.2, 0.02, 0, 1, 0, 0], "color": [0, 0, 1, 1]},  # Blue
}

_available_spots_pose = {
    "Spot A": {"pose": [0.615, -0.2, 0.02, 0, 1, 0, 0]},
    "Spot B": {"pose": [0.615, 0.0, 0.02, 0, 1, 0, 0]},
    "Spot C": {"pose": [0.615, 0.2, 0.02, 0, 1, 0, 0]},
}

# Doesn't matter where the objects are placed, as long as they are stacked in the correct order
_target_objects_pose = {
    "Cube A": [0.615, float("nan"), 0.02, 0, 1, 0, 0],
    "Cube B": [0.615, float("nan"), 0.1, 0, 1, 0, 0],
    "Cube C": [0.615, float("nan"), 0.06, 0, 1, 0, 0],
}

config = TaskConfig(
    _initial_objects=_initial_objects,
    _available_spots_pose=_available_spots_pose,
    _target_objects_pose=_target_objects_pose,
)
