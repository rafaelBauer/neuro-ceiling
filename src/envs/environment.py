from abc import ABC, abstractmethod
from typing import Final, final

from envs.robotactions import RobotAction
from envs.robotinfo import RobotInfo, RobotMotionInfo
from envs.taskconfig import TaskConfig
from utils.logging import log_constructor
from utils.sceneobservation import SceneObservation


class BaseEnvironmentConfig:
    """
    This is a base class for environment configuration. It is used to define the type of environment.

    Attributes
    ----------
    __ENV_TYPE : str
        The type of the environment. This is a private attribute.
        It will be set by the child class. It is used to automatically create the
        environment object.
        The configured object should be in a file named <env_type>.py, and should be named <env_type>Env.
        For example: if the env_type is 'ManiSkill', the file should be maniskill.py and the
        class should be ManiSkillEnv.

    Methods
    -------
    env_type() -> str
        This is a property method that returns __ENV_TYPE.
    """

    def __init__(self, env_type: str, task_config: TaskConfig):
        self.__ENV_TYPE: Final[str] = env_type
        self._task_config: Final[TaskConfig] = task_config

    @property
    def env_type(self) -> str:
        return self.__ENV_TYPE

    @property
    def task_config(self) -> TaskConfig:
        return self._task_config


class BaseEnvironment(ABC):
    """
    This is the base class for all environments. It defines the common interface that all environments should adhere to.
    Each environment should be configured with a BaseEnvironmentConfig instance.

    Attributes
    ----------
    CONFIG : BaseEnvironmentConfig
        The configuration for this environment.

    Methods
    -------
    reset(**kwargs) -> None
        Resets the environment to a new episode. In the BaseEnvironment, this only resets the gripper plot.

    step(action: np.ndarray) -> tuple[dict, float, bool, dict]
        Executes the action in the environment. Simple wrapper around _step, that allows us to perform extra actions
        before and after the step.

    _step(action: np.ndarray) -> tuple[dict, float, bool, dict]
        Executes the action in the environment. This method is abstract and should be implemented in child classes.

    start() -> None
        Starts the environment. In the real word would open the connections and do everything needed to allow the
        interactions. In the simulation, it creates the environment.

    stop() -> None
        Gracefully stops and closes the environment.

    reset_joint_pose() -> None
        Resets the joint pose. This method is abstract and should be implemented in child classes.
    """

    def __init__(self, config: BaseEnvironmentConfig) -> None:
        """
        Constructs all the necessary attributes for the BaseEnvironment object.

        Parameters
        ----------
        config : BaseEnvironmentConfig
            The configuration for this environment.
        """
        self.CONFIG: BaseEnvironmentConfig = config

    def reset(self, **kwargs) -> SceneObservation:
        """
        Resets the environment to a new episode. In the BaseEnvironment, this only resets the gripper plot.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments. Not used in the base class.
        """

    @final
    def step(self, action: RobotAction) -> tuple[SceneObservation, float, bool, dict]:
        """
        Executes the action in the environment. Simple wrapper around _step, that allows us to perform extra actions
        before and after the step.

        Parameters
        ----------
        action : RobotAction
           The action predicted by a policy. This should be a RobotAction object.

        Returns
        -------
        tuple[dict, float, bool, dict]
            The observation, reward, done flag and info dict.
        """
        assert isinstance(action, RobotAction), f"Action should be of type RobotAction, but got {type(action)}"
        return self._step(action)

    @abstractmethod
    def _step(self, action: RobotAction) -> tuple[SceneObservation, float, bool, dict]:
        """
        Executes the action in the environment. This method is abstract and should be implemented in child classes.

        Parameters
        ----------
        action : np.ndarray[(7,), np.float32]
           The action to execute. This should be a 7D vector consisting of the delta position (x, y, z),
           delta rotation (quaternion - (x, y, z, w)),
           and gripper action.


        Returns
        -------
        tuple[dict, float, bool, dict]
           The observation, reward, done flag, and info dictionary resulting from executing the action.
        """

        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """
        Starts the environment. In the real word would open the connections and do everything needed to allow the
        interactions. In the simulation, it creates the environment.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Gracefully stops and closes the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_joint_pose(self) -> None:
        """
        Resets the joint pose. This method is abstract and should be implemented in child classes.
        """
        raise NotImplementedError("Need to implement in child class.")

    @abstractmethod
    def get_robot_info(self) -> final(RobotInfo):
        """
        Returns the robot object of the environment.

        Returns
        -------
        RobotInfo
            The data object containing the robot information.
        """
        raise NotImplementedError("Need to implement in child class.")

    @abstractmethod
    def get_robot_motion_info(self) -> final(RobotMotionInfo):
        """
        Returns the robot object of the environment.

        Returns
        -------
        RobotInfo
            The data object containing the robot information.
        """
        raise NotImplementedError("Need to implement in child class.")
