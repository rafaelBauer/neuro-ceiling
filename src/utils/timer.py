import threading
import time
from collections.abc import Callable
from typing import Final, Optional

from utils.logging import logger


class Timer:
    """
    A Timer class that extends the threading.Thread class.

    This class is used to execute a function periodically in a separate thread.

    Attributes:
        __timer_runs (threading.Event): An event that controls the running of the timer.
        __function (Callable[[], None]): The function to be executed periodically.
        __period (float): The period in seconds between each execution of the function.
    """

    def __init__(self, function: Callable[[], None], period_s: float):
        """
        Initialize the Timer with the function to be executed and the period.

        Args:
            function (Callable[[], None]): The function to be executed periodically.
            period_s (float): The period in seconds between each execution of the function.
        """
        self.__timer_runs = threading.Event()
        self.__timer_runs.clear()
        self.__function: Final[Callable[[], None]] = function
        self.__period: Final[float] = period_s
        self.__current_thread: Optional[threading.Thread] = None
        super().__init__()

    def start(self):
        """
        Start the timer.

        This method sets the event that controls the running of the timer and starts the thread.
        """
        if not self.__timer_runs.is_set():
            self.__timer_runs.set()
            self.__current_thread = threading.Thread(target=self.run)
            self.__current_thread.start()

    def stop(self):
        """
        Stop the timer.

        This method clears the event that controls the running of the timer and joins the thread.
        """
        if self.__timer_runs.is_set():
            self.__timer_runs.clear()
            self.__current_thread.join()
            self.__current_thread = None

    def run(self):
        """
        Run the timer.

        This method is executed when the thread is started. It executes the function periodically
        as long as the timer is running. If the execution of the function takes longer than the period,
        a warning is logged and the function is executed again immediately.
        """
        while self.__timer_runs.is_set():
            start_time: float = time.time()
            self.__function()
            sleep_time = self.__period - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                logger.warning("Callback took longer than period. Not sleeping.")
