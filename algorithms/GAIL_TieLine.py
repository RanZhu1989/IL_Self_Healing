import os
import collections
import random
from tqdm import tqdm
from typing import Tuple, Optional
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd

import selfhealing_env
from utils import logger, check_cuda, get_time
from configs import args


class Episode_Recorder():

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.reset()

    def append(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        obs = torch.tensor(np.array([obs]),
                           dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array([[action]]),
                              dtype=torch.int64).to(self.device)
        reward = torch.tensor(np.array([[reward]]),
                              dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(np.array([next_obs]),
                                dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array([[done]]),
                            dtype=torch.float32).to(self.device)
        self.trajectory["obs"] = torch.cat((self.trajectory["obs"], obs))
        self.trajectory["action"] = torch.cat(
            (self.trajectory["action"], action))
        self.trajectory["reward"] = torch.cat(
            (self.trajectory["reward"], reward))
        self.trajectory["next_obs"] = torch.cat(
            (self.trajectory["next_obs"], next_obs))
        self.trajectory["done"] = torch.cat((self.trajectory["done"], done))

    def get_trajectory(
        self
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor,
               torch.tensor, torch.tensor]:

        return self.trajectory["obs"], self.trajectory["action"], \
            self.trajectory["reward"], self.trajectory["next_obs"], self.trajectory["done"]

    def reset(self) -> None:
        """ Clear the trajectory when begin a new episode."""
        self.trajectory = {
            "obs": torch.tensor([], dtype=torch.float32).to(self.device),
            "action": torch.tensor([], dtype=torch.int64).to(self.device),
            "reward": torch.tensor([], dtype=torch.float32).to(self.device),
            "next_obs": torch.tensor([], dtype=torch.float32).to(self.device),
            "done": torch.tensor([], dtype=torch.float32).to(self.device)
        }

    def add_reward(self, reward: np.ndarray) -> None:
        self.trajectory["reward"] = torch.tensor(
            reward, dtype=torch.float32).to(self.device)


# PPO Agent
class PPO_Clip_Agent():
    """ 
    We use the PPO-Clip algorithm to act as the Policy (generator) in the GAIL framework.
    """

    def __init__(
        self,
        episode_recorder: Episode_Recorder,
        actor_network: torch.nn.Module,
        critic_network: torch.nn,
        actor_optimizer: torch.optim,
        critic_optimizer: torch.optim,
        gamma: float = 0.9,
        advantage_lambda: float = 0.95,  # Discount factor for advantage function
        clip_epsilon: float = 0.2,  # Clipping parameter
        train_iters:
        int = 10,  # Number of iterations to train the policy in each episode
        batch_size: int = 32,
        device: torch.device = torch.device("cpu")
    ) -> None:

        self.device = device
        self.episode_recorder = episode_recorder
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_epsilon = clip_epsilon
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.advantage_lambda = advantage_lambda

    def get_action(self, obs: np.ndarray) -> torch.tensor:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        dist = self.actor_network(obs)
        action_probs = torch.distributions.Categorical(dist)
        picked_action = action_probs.sample()

        return picked_action

    def calculate_log_prob(self, obs: torch.tensor,
                           action: torch.tensor) -> torch.tensor:
        dist = self.actor_network(obs)
        log_prob = torch.log(dist.gather(-1, action))

        return log_prob

    def calculate_advantage(self, td_error: torch.tensor) -> torch.tensor:
        """ The advantage function is calculated by the TD error. """
        td_error = td_error.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_error[::-1]:
            advantage = self.gamma * self.advantage_lambda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list),
                                 dtype=torch.float32).to(self.device)

        return advantage

    def train_policy(self) -> None:
        obs, action, reward, next_obs, done = self.episode_recorder.get_trajectory(
        )

        # Data preparation
        # 1. TD_target = r + gamma * V(s')
        TD_target = reward + self.gamma * self.critic_network(next_obs) * (
            1 - done)
        TD_error = TD_target - self.critic_network(obs)
        # 2. Advantage
        advantage = self.calculate_advantage(TD_error)
        # 3. Log_prob
        old_log_prob = self.calculate_log_prob(obs, action).detach(
        )  # Freeze the log_prob obtained by the current policy
        # Update the policy by batch
        for _ in range(self.train_iters):
            sample_indices = np.random.randint(low=0,
                                               high=obs.shape[0],
                                               size=self.batch_size)

            critic_loss = torch.mean(
                F.mse_loss(TD_target[sample_indices].detach(),
                           self.critic_network(obs[sample_indices])))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            log_prob = self.calculate_log_prob(obs[sample_indices],
                                               action[sample_indices])
            ratio = torch.exp(
                log_prob -
                old_log_prob[sample_indices])  # pi_theta / pi_theta_old
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon,
                                        1 + self.clip_epsilon)
            actor_loss = torch.mean(
                -torch.min(ratio * advantage[sample_indices], clipped_ratio *
                           advantage[sample_indices]))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


class Actor_Network(torch.nn.Module):

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(Actor_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


class Critic_Network(torch.nn.Module):

    def __init__(self, obs_dim: int) -> None:
        super(Critic_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Expert experience collector
class ReplayBuffer():
    """Expert experience collector"""

    def __init__(
        self, capacity: int,
        device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, exp_data: tuple) -> None:
        self.buffer.append(exp_data)

    def sample(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        if shuffle:
            self.buffer = random.sample(self.buffer, len(self.buffer))
        batch_size = min(batch_size,
                         len(self.buffer))  # in case the buffer is not enough
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, next_obs_batch, done_batch = zip(*mini_batch)
        obs_batch = torch.tensor(np.array(obs_batch),
                                 dtype=torch.float32,
                                 device=self.device)
        action_batch = torch.tensor(np.array(action_batch),
                                    dtype=torch.int64,
                                    device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch),
                                      dtype=torch.float32,
                                      device=self.device)
        done_batch = torch.tensor(np.array(done_batch),
                                  dtype=torch.float32,
                                  device=self.device)

        return obs_batch, action_batch, next_obs_batch, done_batch

    def fetch_all(
        self,
        shuffle: bool = True
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        if shuffle:
            self.buffer = random.sample(self.buffer, len(self.buffer))
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(
            *self.buffer)
        obs_batch = torch.tensor(np.array(obs_batch),
                                 dtype=torch.float32,
                                 device=self.device)
        action_batch = torch.tensor(np.array(action_batch),
                                    dtype=torch.int64,
                                    device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch),
                                      dtype=torch.float32,
                                      device=self.device)
        done_batch = torch.tensor(np.array(done_batch),
                                  dtype=torch.float32,
                                  device=self.device)

        return obs_batch, action_batch, next_obs_batch, done_batch

    def release(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


# GAIL algorithm
class GAIL():

    def __init__(self, action_dim: int, d_network: torch.nn,
                 d_optimizer: torch.optim, train_iters: int,
                 d_batch_size: int) -> None:
        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.action_dim = action_dim
        self.train_iters = train_iters
        self.d_batch_size = d_batch_size

    def train_d(self, exp_obs: torch.tensor, exp_action: torch.tensor,
                agent_obs: torch.tensor,
                agent_action: torch.tensor) -> np.ndarray:
        for _ in range(self.train_iters):
            exp_sample_indices = np.random.randint(low=0,
                                                   high=exp_obs.shape[0],
                                                   size=self.d_batch_size)
            agent_sample_indices = np.random.randint(low=0,
                                                     high=agent_obs.shape[0],
                                                     size=self.d_batch_size)

            batch_exp_action = F.one_hot(
                exp_action[exp_sample_indices].to(torch.int64),
                num_classes=self.action_dim).to(torch.float32).squeeze()
            batch_agent_action = F.one_hot(
                agent_action[agent_sample_indices].to(torch.int64),
                num_classes=self.action_dim).to(torch.float32).squeeze()
            batch_exp_prob = self.d_network(exp_obs[exp_sample_indices],
                                            batch_exp_action)
            batch_agent_prob = self.d_network(agent_obs[agent_sample_indices],
                                              batch_agent_action)
            loss = torch.nn.BCELoss()(
                batch_agent_prob,
                torch.ones_like(batch_agent_prob)) + torch.nn.BCELoss()(
                    batch_exp_prob, torch.zeros_like(batch_exp_prob))
            self.d_optimizer.zero_grad()
            loss.backward()
            self.d_optimizer.step()
            pass

        agent_action = F.one_hot(agent_action.to(torch.int64),
                                 num_classes=self.action_dim).to(
                                     torch.float32).squeeze()
        gail_reward = -torch.log(self.d_network(
            agent_obs, agent_action)).detach().cpu().numpy()

        return gail_reward


class Discriminator(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim + action_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class TrainManager():

    def __init__(
        self,
        env: gym.Env,
        log_output_path: Optional[str] = None,
        episode_num: int = 1000,
        expert_episode_used: int = 100,
        expert_sample_used: int = 50,  # use # samples to train the agent
        discriminator_train_iters=5,
        discriminator_batch_size: int = 8,
        test_iterations: int = 5,
        d_lr: float = 1e-3,
        buffer_capacity: int = 10000,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.9,
        advantage_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        on_policy_env_iters=20,
        ppo_train_iters: int = 10,
        ppo_batch_size: int = 32,
        seed: int = 0,
        device: torch.device = torch.device("cpu")) -> None:

        self.seed = seed
        self.test_rng = random.Random(self.seed + 1)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.device = device

        self.log_output_path = log_output_path
        self.logger = logger(log_output_path=self.log_output_path)
        self.expert_episode_used = expert_episode_used
        self.expert_sample_used = expert_sample_used
        self.test_iterations = test_iterations
        self.discriminator_train_iters = discriminator_train_iters
        self.discriminator_batch_size = discriminator_batch_size
        self.on_policy_env_iters = on_policy_env_iters

        self.env = env
        _ = self.env.reset(seed=self.seed)

        self.episode_num = episode_num
        self.obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.action_dim = env.action_space.n
        episode_recorder = Episode_Recorder(device=self.device)
        actor_network = Actor_Network(self.obs_dim,
                                      self.action_dim).to(self.device)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(),
                                           lr=actor_lr)
        critic_network = Critic_Network(self.obs_dim).to(self.device)
        critic_optimizer = torch.optim.Adam(critic_network.parameters(),
                                            lr=critic_lr)
        self.agent = PPO_Clip_Agent(episode_recorder=episode_recorder,
                                    actor_network=actor_network,
                                    critic_network=critic_network,
                                    actor_optimizer=actor_optimizer,
                                    critic_optimizer=critic_optimizer,
                                    gamma=gamma,
                                    advantage_lambda=advantage_lambda,
                                    clip_epsilon=clip_epsilon,
                                    train_iters=ppo_train_iters,
                                    batch_size=ppo_batch_size,
                                    device=self.device)

        self.buffer = ReplayBuffer(capacity=buffer_capacity,
                                   device=self.device)
        d_network = Discriminator(self.obs_dim,
                                  self.action_dim).to(self.device)
        d_optimizer = torch.optim.Adam(d_network.parameters(), lr=d_lr)
        self.gail = GAIL(self.action_dim, d_network, d_optimizer,
                         self.discriminator_train_iters,
                         self.discriminator_batch_size)

        self.episode_total_rewards = np.zeros(self.episode_num)
        self.index_episode = 0

    def train(self) -> None:
        # 1. Collect expert data
        self.logger.event_logger.info(
            f"=============== Expert Data Collection =================")
        options = {
            "Specific_Disturbance": None,
            "Expert_Policy_Required": True,
            "External_RNG": None
        }
        sampled_idx = 0
        with tqdm(total=self.expert_episode_used,
                  desc='Collecting Expert Data') as pbar:
            while sampled_idx < self.expert_episode_used:
                # Ensure that the env has a solution
                while True:
                    obs, info = self.env.reset(options=options)
                    if info["Expert_Policy"] != None:
                        sampled_idx += 1
                        break
                # Collect expert experience
                X = info["Expert_Policy"]["Branch_Obs"]  # s0-s4
                Y = info["Expert_Policy"]["TieLine_Action"]  # a1-a5
                step = 0
                done = False
                while not done:
                    action = Y[step].item()
                    next_obs, _, terminated, truncated, _ = self.env.step(
                        action)
                    done = terminated or truncated
                    self.buffer.append(
                        (obs, action, next_obs, done))  # append data to buffer
                    obs = next_obs
                    step += 1

                pbar.update(1)

        self.exp_obs, self.exp_action, self.exp_next_obs, self.exp_done = self.buffer.sample(
            batch_size=self.expert_sample_used, shuffle=True)

        # 2. Train the target PPO agent via interacting with the environment

        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for e in range(self.episode_num):
                self.logger.event_logger.info(
                    f"=============== Episode {e+1:d} of {self.episode_num:d} ================="
                )
                self.train_episode()
                avg_test_success_rate = self.test()
                pbar.set_postfix(
                    {"Avg. Success rate": (avg_test_success_rate)})
                pbar.update(1)

    def train_episode(self) -> float:
        options = {
            "Specific_Disturbance": None,
            "Expert_Policy_Required": True,
            "External_RNG": None
        }
        self.agent.episode_recorder.reset()
        obs = None
        for _ in range(self.on_policy_env_iters):
            # Start new episodes and collect the trajectories via target policy
            while True:
                obs, info = self.env.reset(options=options)
                if info["Expert_Policy"] != None:
                    break
            while True:
                action = self.agent.get_action(obs).item()
                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = terminated or truncated
                self.agent.episode_recorder.append(obs, action, reward,
                                                   next_obs, done)
                obs = next_obs
                if done:
                    break
        # Get the trajectory
        agent_obs, agent_action, _, _, _ = self.agent.episode_recorder.get_trajectory(
        )

        reward = self.gail.train_d(exp_obs=self.exp_obs,
                                   exp_action=self.exp_action,
                                   agent_obs=agent_obs,
                                   agent_action=agent_action)

        # Update the reward
        self.agent.episode_recorder.add_reward(reward)

        # Train the target PPO agent
        self.agent.train_policy()

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
                a = self.agent.get_action(s0)
                a = int(a)
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

        return np.array(saved_success_rate).mean()


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
    # GAIL parameters
    parser.add_argument('--used_exp_episodes', type=int, default=300)
    parser.add_argument('--used_exp_samples', type=int, default=500)
    parser.add_argument('--GAIL_d_iters', type=int, default=5)
    parser.add_argument('--GAIL_d_batch_size', type=int, default=32)
    parser.add_argument('--GAIL_d_lr', type=float, default=5.0e-3)
    # PPO (actor)
    parser.add_argument('--on_policy_env_iters', type=int, default=20)
    parser.add_argument('--PPO_actor_lr', type=float, default=1.0e-2)
    parser.add_argument('--PPO_critic_lr', type=float, default=5.0e-3)
    parser.add_argument('--PPO_gamma', type=float, default=0.98)
    parser.add_argument('--PPO_advantage_lambda', type=float, default=0.95)
    parser.add_argument('--PPO_clip_epsilon', type=float, default=0.2)
    parser.add_argument('--PPO_train_iters', type=int, default=5)
    parser.add_argument('--PPO_batch_size', type=int, default=32)
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
    task_name = 'GAIL_TieLine'
    log_output_path = current_path + "/" + args.result_folder_name + "/" + task_name + \
                    ("_n_" + str(args.min_disturbance) + "to" + str(args.max_disturbance)
                        + "_" + get_time() + "/" )

    tensorboard_path = log_output_path + "tensorboard/"

    if args.forced_cpu:
        device = torch.device("cpu")
    else:
        device = check_cuda()

    manager = TrainManager(env=env,
                           log_output_path=log_output_path,
                           test_iterations=args.test_iterations,
                           expert_episode_used=args.used_exp_episodes,
                           expert_sample_used=args.used_exp_samples,
                           discriminator_train_iters=args.GAIL_d_iters,
                           discriminator_batch_size=args.GAIL_d_batch_size,
                           episode_num=args.train_epochs,
                           on_policy_env_iters=args.on_policy_env_iters,
                           d_lr=args.GAIL_d_lr,
                           actor_lr=args.PPO_actor_lr,
                           critic_lr=args.PPO_critic_lr,
                           gamma=args.PPO_gamma,
                           advantage_lambda=args.PPO_advantage_lambda,
                           clip_epsilon=args.PPO_clip_epsilon,
                           ppo_train_iters=args.PPO_train_iters,
                           ppo_batch_size=args.PPO_batch_size,
                           seed=args.seed,
                           device=device)

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
