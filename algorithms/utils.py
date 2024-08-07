"""
This utility file is modified from the original file from the repository https://github.com/whoiszyc/IntelliHealer.
"""
import sys
import os
import logging
import csv
from datetime import datetime
from typing import Optional, Union
from pathlib import Path

import numpy as np
import torch


class logger:

    def __init__(self,
                 log_output_path: Optional[str] = None,
                 level: int = logging.DEBUG,
                 verbose: int = 0) -> None:
        """
        Method to return a custom logger with the given name and level
        """
        now = datetime.now()
        dt_string = now.strftime("__%Y_%m_%d_%H_%M")
        self.dt_string = dt_string
        # check if the dir is given
        if log_output_path is None:
            # if dir is not given, save results at root dir
            output_path = os.getcwd()
            event_log_output_path = output_path + "/" + "log" + dt_string + ".log"
            self.disturb_log_path = output_path + "/" + "disturb" + dt_string + ".csv"
            self.agent_recovery_rate_log_path = output_path + "/" + "agent_recovery_rate" + dt_string + ".csv"
            self.expert_recovery_rate_log_path = output_path + "/" + "expert_recovery_rate" + dt_string + ".csv"
            self.success_rate_log_path = output_path + "/" + "success_rate" + dt_string + ".csv"

        else:
            ensure_directory(log_output_path)
            event_log_output_path = log_output_path + "/" + "log" + dt_string + ".log"
            self.disturb_log_path = log_output_path + "/" + "disturb" + dt_string + ".csv"
            self.agent_recovery_rate_log_path = log_output_path + "/" + "agent_recovery_rate" + dt_string + ".csv"
            self.expert_recovery_rate_log_path = log_output_path + "/" + "expert_recovery_rate" + dt_string + ".csv"
            self.success_rate_log_path = log_output_path + "/" + "success_rate" + dt_string + ".csv"

        # create the logger
        self.event_logger = logging.getLogger(event_log_output_path)
        self.event_logger.setLevel(level)
        format_string = (
            "%(asctime)s - %(levelname)s - %(funcName)s (%(lineno)d):  %(message)s"
        )
        datefmt = "%Y-%m-%d %I:%M:%S %p"
        log_format = logging.Formatter(format_string, datefmt)

        # Creating and adding the file handler
        file_handler = logging.FileHandler(event_log_output_path, mode="a")
        file_handler.setFormatter(log_format)
        self.event_logger.addHandler(file_handler)

        if verbose == 1:
            # Creating and adding the console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            self.event_logger.addHandler(console_handler)

    def save_to_file(self, disturb: Union[list, np.ndarray],
                     agent_recovery_rate: Union[list, np.ndarray],
                     expert_recovery_rate: Union[list, np.ndarray],
                     success_rate: Union[list, np.ndarray]) -> None:
        _save_csv(self.disturb_log_path, disturb)
        _save_csv(self.agent_recovery_rate_log_path, agent_recovery_rate)
        _save_csv(self.expert_recovery_rate_log_path, expert_recovery_rate)
        _save_csv(self.success_rate_log_path, success_rate)


def _save_csv(path: str, score: Union[list, np.ndarray]) -> None:
    if not os.path.exists(path):
        with open(path, "w"):
            pass
    with open(path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(score)


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_directory(directory: Union[Path, str]) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    pass


def find_1Dtensor_value_indices(tensor: torch.Tensor,
                                value_range: int) -> dict:
    """
    Finds indices for each value in a given tensor within a specified range.

    Parameters:
    - tensor (torch.Tensor): The input tensor.
    - value_range (int): The range of values to search for in the tensor.

    Returns:
    - dict: A dictionary where keys are the values within the specified range,
            and values are lists of indices where those values occur in the tensor.
    """
    value_indices = {}
    for value in range(
            value_range):  # Iterate through the specified range of values
        indices = torch.where(
            tensor == value)[0].tolist()  # Find indices of the current value
        value_indices[value] = indices  # Store indices in the dictionary
    return value_indices


def check_cuda() -> torch.device:
    # check cuda
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # print('use_gpu ==', use_gpu)
        # print('device_ids ==', np.arange(0, torch.cuda.device_count()))
        return torch.device('cuda')
    else:
        return torch.device('cpu')
