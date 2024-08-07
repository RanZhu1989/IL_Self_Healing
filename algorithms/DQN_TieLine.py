import os
import collections
import random
from tqdm import tqdm
from typing import Tuple, Optional
import copy
import argparse

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

import selfhealing_env
from utils import logger, check_cuda, get_time


class ReplayBuffer():

    def __init__(
        self, capacity: int,
        device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, exp_data: tuple) -> None:
        self.buffer.append(exp_data)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor,
               torch.tensor]:
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(
            *mini_batch)

        obs_batch = torch.tensor(np.array(obs_batch),
                                 dtype=torch.float32,
                                 device=self.device)

        action_batch = torch.tensor(action_batch,
                                    dtype=torch.int64,
                                    device=self.device).unsqueeze(1)

        reward_batch = torch.tensor(reward_batch,
                                    dtype=torch.float32,
                                    device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch),
                                      dtype=torch.float32,
                                      device=self.device)
        done_batch = torch.tensor(done_batch,
                                  dtype=torch.float32,
                                  device=self.device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self) -> int:
        return len(self.buffer)


class DQN_Agent():
    """Vanilla DQN Agent"""

    def __init__(
        self,
        Q_func: torch.nn.Module,
        action_dim: int,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        replay_start_size: int,
        batch_size: int,
        replay_frequent: int,
        target_sync_frequent:
        int,  # The frequency of synchronizing the parameters of the two Q networks
        epsilon: float = 0.1,  # Initial epsilon
        mini_epsilon: float = 0.01,  # Minimum epsilon
        explore_decay_rate: float = 0.0001,  # Decay rate of epsilon
        gamma: float = 0.9,
        device: torch.device = torch.device("cpu")
    ) -> None:

        self.device = device
        self.action_dim = action_dim

        self.exp_counter = 0

        self.replay_buffer = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.replay_frequent = replay_frequent

        self.target_sync_frequent = target_sync_frequent
        """Two Q functions (mian_Q and target_Q) are used to stabilize the training process. 
            Since they share the same network structure, we can use copy.deepcopy to copy the main_Q to target_Q for initialization."""
        self.main_Q_func = Q_func
        self.target_Q_func = copy.deepcopy(Q_func)

        self.optimizer = optimizer

        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.gamma = gamma
        self.explore_decay_rate = explore_decay_rate
        pass

    def get_target_action(self, obs: np.ndarray) -> int:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        Q_list = self.target_Q_func(obs)
        action = torch.argmax(Q_list).item()
        return action

    def get_behavior_action(self, obs: np.ndarray) -> int:
        """Here, a simple epsilon decay is used to balance the exploration and exploitation.
            The epsilon is decayed from epsilon_init to mini_epsilon."""
        self.epsilon = max(self.mini_epsilon,
                           self.epsilon - self.explore_decay_rate)

        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.get_target_action(obs)

        return action

    """Here, we defined a function to synchronize the parameters of the main_Q and target_Q."""

    def sync_target_Q_func(self) -> None:
        for target_params, main_params in zip(self.target_Q_func.parameters(),
                                              self.main_Q_func.parameters()):
            target_params.data.copy_(main_params.data)

    def batch_Q_approximation(self, batch_obs: torch.tensor,
                              batch_action: torch.tensor,
                              batch_reward: torch.tensor,
                              batch_next_obs: torch.tensor,
                              batch_done: torch.tensor) -> None:

        batch_current_Q = torch.gather(self.main_Q_func(batch_obs), 1,
                                       batch_action).squeeze(1)
        batch_TD_target = batch_reward + (
            1 - batch_done
        ) * self.gamma * self.target_Q_func(batch_next_obs).max(1)[0]
        loss = torch.mean(F.mse_loss(batch_current_Q, batch_TD_target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Q_approximation(self, obs: np.ndarray, action: int, reward: float,
                        next_obs: np.ndarray, done: bool) -> None:

        self.exp_counter += 1
        self.replay_buffer.append((obs, action, reward, next_obs, done))

        if len(
                self.replay_buffer
        ) > self.replay_start_size and self.exp_counter % self.replay_frequent == 0:
            self.batch_Q_approximation(
                *self.replay_buffer.sample(self.batch_size))

        # Synchronize the parameters of the two Q networks every target_update_frequent steps
        if self.exp_counter % self.target_sync_frequent == 0:
            self.sync_target_Q_func()


class Q_Network(torch.nn.Module):
    """You can define your own network structure here."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(Q_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class TrainManager():

    def __init__(
        self,
        env: gym.Env,
        log_output_path: Optional[str] = None,
        episode_num: int = 1000,
        test_iterations: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        mini_epsilon: float = 0.01,
        explore_decay_rate: float = 0.0001,
        buffer_capacity: int = 2000,
        replay_start_size: int = 200,
        replay_frequent: int = 4,
        target_sync_frequent: int = 200,
        batch_size: int = 32,
        seed: int = 0,
        my_device: torch.device = torch.device('cpu')) -> None:

        self.env = env
        obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        action_dim = env.action_space.n

        self.seed = seed
        _, _ = self.env.reset(seed=self.seed)
        self.test_rng = random.Random(self.seed + 1)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device(my_device)
        Q_func = Q_Network(obs_dim, action_dim).to(self.device)
        optimizer = torch.optim.Adam(Q_func.parameters(), lr=lr)
        self.episode_num = episode_num
        self.log_output_path = log_output_path
        self.logger = logger(log_output_path=self.log_output_path)
        self.test_iterations = test_iterations

        self.buffer = ReplayBuffer(capacity=buffer_capacity,
                                   device=self.device)

        self.agent = DQN_Agent(Q_func=Q_func,
                               action_dim=action_dim,
                               optimizer=optimizer,
                               replay_buffer=self.buffer,
                               replay_start_size=replay_start_size,
                               batch_size=batch_size,
                               replay_frequent=replay_frequent,
                               target_sync_frequent=target_sync_frequent,
                               epsilon=epsilon,
                               mini_epsilon=mini_epsilon,
                               explore_decay_rate=explore_decay_rate,
                               gamma=gamma,
                               device=self.device)

        self.episode_total_rewards = np.zeros(episode_num)
        self.index_episode = 0
        pass

    def train_episode(self) -> None:
        obs, _ = self.env.reset()
        done = False
        while not done:
            action = self.agent.get_behavior_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.Q_approximation(obs, action, reward, next_obs, done)
            obs = next_obs

    def train(self) -> None:
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for e in range(self.episode_num):
                episode_reward = self.train_episode()
                # run test
                avg_test_success_rate = self.test()
                pbar.set_postfix({"test_avg_rate": avg_test_success_rate})
                pbar.update(1)

    def test(self) -> float:
        options = {
            "Specific_Disturbance": None,
            "Expert_Policy_Required": True,
            "External_RNG": self.test_rng
        }
        self.logger.event_logger.info(
            "-------------------Test Report--------------------")
        saved_disturbance_set = []  # Save data
        saved_agent_load_rate = []
        saved_expert_load_rate = []
        saved_success_rate = []

        # Test for multiple times
        for test_idx in range(self.test_iterations):
            while True:
                s0, expert_info = self.env.reset(options=options)
                if expert_info["Expert_Policy"] != None:
                    break
            test_disturbance_set = expert_info["Disturbance_Set"]
            self.logger.event_logger.info("# Test {}".format(test_idx + 1))
            self.logger.event_logger.info(
                "The testing disturbance is {}".format(
                    sorted(test_disturbance_set)))
            # Calculate goal using expert policy
            load_rate_s0 = expert_info["Recovered_Load_Rate_S0"]
            expert_load_rate = expert_info["Expert_Policy"]["Load_Rate"]
            # goal = sum[optimal load rate] - sum[load rate at s0] * total time steps
            goal = np.sum(
                expert_load_rate).item() - load_rate_s0 * len(expert_load_rate)
            self.logger.event_logger.info("Expert policy is {}".format(
                expert_info["Expert_Policy"]["TieLine_Action"].T.tolist()))
            # Calculate performance using agent policy
            agent_info = None
            agent_load_rate = []
            for step in self.env.unwrapped.exploration_seq_idx:
                s0 = np.reshape(s0,
                                (-1, self.env.unwrapped.system_data.N_Branch))
                a = self.agent.get_target_action(s0)
                self.logger.event_logger.info("Given state S[{}] = {}".format(
                    step, s0[0]))
                self.logger.event_logger.info(
                    "Agent action at S[{}] is A[{}] = {}".format(
                        step, step + 1, a))
                s, reward, _, _, agent_info = self.env.step(a)
                load_rate = agent_info["Recovered_Load_Rate"]
                agent_load_rate.append(load_rate)
                self.logger.event_logger.info("Event at S[{}] is '{}'".format(
                    step + 1, agent_info["Event_Log"]))
                self.logger.event_logger.info(
                    "Reward R[{}] is {}. Load rate at S[{}] is {}.".format(
                        step + 1, reward, step + 1, load_rate))
                s0 = s
            # Evaluate agent performance
            self.logger.event_logger.info(
                "Load rate at S0 is {}".format(load_rate_s0))
            # performance = sum[load rate] - sum[load rate at s0] * total time steps
            performance = np.sum(
                agent_load_rate).item() - load_rate_s0 * len(agent_load_rate)
            if abs(goal) > 1e-6:
                success_rate = min(performance / goal, 1)
                self.logger.event_logger.info(
                    "performance: {}; goal: {}".format(performance, goal))
            else:
                success_rate = 1  # if goal is too small, we consider it as success
                self.logger.event_logger.info(
                    "Nothing we can do to improve the load profile. (Reason: performance={}, goal={})"
                    .format(performance, goal))

            saved_disturbance_set.append(test_disturbance_set)
            saved_agent_load_rate.append(agent_load_rate)
            saved_expert_load_rate.append(expert_load_rate)
            saved_success_rate.append(success_rate)

        # Save data to file
        self.logger.save_to_file(
            disturb=saved_disturbance_set,
            agent_recovery_rate=np.array(saved_agent_load_rate),
            expert_recovery_rate=np.array(saved_expert_load_rate),
            success_rate=np.array(saved_success_rate))

        return np.mean(saved_success_rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--env_id', type=str, default='SelfHealing-v0')
    parser.add_argument('--data_file', type=str, default='Case_33BW_Data.xlsx')
    parser.add_argument('--opt_framework', type=str, default='Gurobipy')
    parser.add_argument('--solver', type=str, default='gurobi')
    parser.add_argument('--solver_display', type=bool, default=False)
    parser.add_argument('--vvo', type=bool, default=False)
    parser.add_argument('--min_disturbance', type=int, default=1)
    parser.add_argument('--max_disturbance', type=int, default=1)
    parser.add_argument('--Sb', type=int, default=100)
    parser.add_argument('--V0', type=float, default=1.05)
    parser.add_argument('--V_min', type=float, default=0.95)
    parser.add_argument('--V_max', type=float, default=1.05)
    # Public training parameters
    parser.add_argument('--result_folder_name', type=str, default='results')
    parser.add_argument('--forced_cpu', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_iterations', type=int, default=5)
    parser.add_argument('--train_epochs', type=int, default=1000)
    # DQN parameters
    parser.add_argument('--DQN_lr', type=float, default=1.0e-3)
    parser.add_argument('--DQN_gamma', type=float, default=0.95)
    parser.add_argument('--DQN_epsilon', type=float, default=0.2)
    parser.add_argument('--DQN_mini_epsilon', type=float, default=5.0e-2)
    parser.add_argument('--DQN_explore_decay_rate', type=float, default=1.0e-4)
    parser.add_argument('--DQN_buffer_capacity', type=int, default=20000)
    parser.add_argument('--DQN_replay_start_size', type=int, default=64)
    parser.add_argument('--DQN_replay_frequent', type=int, default=4)
    parser.add_argument('--DQN_target_sync_frequent', type=int, default=2)
    parser.add_argument('--DQN_batch_size', type=int, default=32)

    # parse arguments
    args = parser.parse_args()

    env = gym.make(id=args.env_id,
                   opt_framework=args.opt_framework,
                   solver=args.solver,
                   data_file=args.data_file,
                   solver_display=args.solver_display,
                   min_disturbance=args.min_disturbance,
                   max_disturbance=args.max_disturbance,
                   vvo=False,
                   Sb=args.Sb,
                   V0=args.V0,
                   V_min=args.V_min,
                   V_max=args.V_max)

    current_path = os.getcwd()
    task_name = 'DQN_TieLine'
    log_output_path = current_path + "/" + args.result_folder_name + "/" + task_name + \
                    ("_n_" + str(args.min_disturbance) + "to" + str(args.max_disturbance)
                        + "_" + get_time() + "/" )

    if args.forced_cpu:
        device = torch.device("cpu")
    else:
        device = check_cuda()

    manager = TrainManager(env=env,
                           log_output_path=log_output_path,
                           episode_num=args.train_epochs,
                           test_iterations=args.test_iterations,
                           lr=args.DQN_lr,
                           gamma=args.DQN_gamma,
                           epsilon=args.DQN_epsilon,
                           mini_epsilon=args.DQN_mini_epsilon,
                           explore_decay_rate=args.DQN_explore_decay_rate,
                           buffer_capacity=args.DQN_buffer_capacity,
                           replay_start_size=args.DQN_replay_start_size,
                           replay_frequent=args.DQN_replay_frequent,
                           target_sync_frequent=args.DQN_target_sync_frequent,
                           batch_size=args.DQN_batch_size,
                           seed=args.seed,
                           my_device=device)

    manager.logger.event_logger.info(
        f"=============== Task Info =================")
    manager.logger.event_logger.info(
        f"ENV Settings == {args.env_id}, Device == {device}, Seed == {args.seed}"
    )
    manager.logger.event_logger.info(
        f"Task == {task_name}, Training Epochs == {args.train_epochs}, Test Iterations == {args.test_iterations}"
    )
    manager.logger.event_logger.info(
        f"Opt_Framework == {args.opt_framework}, Solver == {args.solver}, system_file == {args.data_file}"
    )
    manager.logger.event_logger.info(
        f"min_disturbance == {args.min_disturbance}, max_disturbance == {args.max_disturbance}"
    )

    manager.train()
