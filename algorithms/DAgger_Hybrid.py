import os
import collections
import random
from typing import Tuple, Optional
import argparse

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

import selfhealing_env
from utils import logger, check_cuda, get_time, find_1Dtensor_value_indices


class DAgger_Agent():
    # Dataset Aggregation (DAgger) Agent
    def __init__(
        self,
        discrete_input_dim: int,
        discrete_output_dim: int,
        continuous_input_dim: int,
        continuous_output_dim: int,
        continuous_action_bias: torch.tensor,
        continuous_action_scale: torch.tensor,
        discrete_policy_network: torch.nn.Module,
        continuous_policy_networks: list,
        discrete_optimizer: torch.optim.Optimizer,
        continuous_optimizers: list,
        device: torch.device = torch.device("cpu")
    ) -> None:

        self.discrete_input_dim = discrete_input_dim
        self.discrete_output_dim = discrete_output_dim
        self.continuous_input_dim = continuous_input_dim
        self.continuous_output_dim = continuous_output_dim
        self.continuous_action_bias = continuous_action_bias
        self.continuous_action_scale = continuous_action_scale
        self.discrete_policy_network = discrete_policy_network
        self.continuous_policy_networks = continuous_policy_networks
        self.discrete_optimizer = discrete_optimizer
        self.continuous_optimizers = continuous_optimizers
        self.device = device

    def predict_discrete(self, obs: np.ndarray) -> torch.tensor:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        dist = self.discrete_policy_network(obs)
        picked_action = torch.argmax(dist)

        return picked_action

    def predict_continuous(self, obs: np.ndarray,
                           upper_idx: int) -> torch.tensor:

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        out = self.continuous_policy_networks[upper_idx](obs)
        action = out * self.continuous_action_scale + self.continuous_action_bias

        return action

    def train_discrete(self, obs: torch.tensor, action: torch.tensor) -> None:

        log_prob = torch.log(
            self.discrete_policy_network(obs).gather(1, action))
        loss = torch.mean(-log_prob)
        self.discrete_optimizer.zero_grad()
        loss.backward()
        self.discrete_optimizer.step()

    def train_continuous(self, obs: torch.tensor, action: torch.tensor,
                         upper_idx: int) -> None:
        action_hat = self.continuous_policy_networks[upper_idx](obs)
        loss = F.mse_loss(action_hat, action)
        self.continuous_optimizers[upper_idx].zero_grad()
        loss.backward()
        self.continuous_optimizers[upper_idx].step()


class Discrete_Policy_Network(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(Discrete_Policy_Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)


class Continuous_Policy_Network(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(Continuous_Policy_Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, output_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.tanh(self.fc5(x))


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
        batch_size = min(batch_size,
                         len(self.buffer))  # in case the buffer is not enough
        if shuffle:
            self.buffer = random.sample(self.buffer, len(self.buffer))
        mini_batch = random.sample(self.buffer, batch_size)
        Xd_batch, Yd_batch, Xc_batch, Yc_batch = zip(
            *mini_batch
        )  # discrete_obs, discrete_action, continuous_obs, continuous_action
        Xd_batch = torch.tensor(np.array(Xd_batch),
                                dtype=torch.float32,
                                device=self.device)
        Yd_batch = torch.tensor(np.array(Yd_batch),
                                dtype=torch.int64,
                                device=self.device)
        Xc_batch = torch.tensor(np.array(Xc_batch),
                                dtype=torch.float32,
                                device=self.device)
        Yc_batch = torch.tensor(np.array(Yc_batch),
                                dtype=torch.float32,
                                device=self.device)

        return Xd_batch, Yd_batch, Xc_batch, Yc_batch

    def fetch_all(
        self,
        shuffle: bool = True
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        if shuffle:
            self.buffer = random.sample(self.buffer, len(self.buffer))
        Xd_batch, Yd_batch, Xc_batch, Yc_batch = zip(*self.buffer)
        Xd_batch = torch.tensor(np.array(Xd_batch),
                                dtype=torch.float32,
                                device=self.device)
        Yd_batch = torch.tensor(np.array(Yd_batch),
                                dtype=torch.int64,
                                device=self.device)
        Xc_batch = torch.tensor(np.array(Xc_batch),
                                dtype=torch.float32,
                                device=self.device)
        Yc_batch = torch.tensor(np.array(Yc_batch),
                                dtype=torch.float32,
                                device=self.device)

        return Xd_batch, Yd_batch, Xc_batch, Yc_batch

    def release(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class TrainManager():

    def __init__(self,
                 env: gym.Env,
                 log_output_path: Optional[str] = None,
                 lr: float = 0.001,
                 train_iterations: int = 10,
                 episode_num: int = 200,
                 buffer_capacity: int = 1000000,
                 batch_size: int = 32,
                 test_iterations: int = 5,
                 device: torch.device = torch.device("cpu"),
                 seed: Optional[int] = None) -> None:

        self.env = env
        self.discrete_obs_dim = gym.spaces.utils.flatdim(
            env.observation_space["X_branch"])
        self.discrete_action_dim = gym.spaces.utils.flatdim(
            env.action_space["Tieline"])
        self.continuous_obs_dim = gym.spaces.utils.flatdim(
            env.observation_space["PF"]) + gym.spaces.utils.flatdim(
                env.observation_space["QF"])
        self.continuous_action_dim = gym.spaces.utils.flatdim(
            env.action_space["Varcon"])
        self.device = device

        action_upper_bound = self.env.action_space["Varcon"].high
        action_lower_bound = self.env.action_space["Varcon"].low
        action_bias = (action_upper_bound + action_lower_bound) / 2.0
        action_bias = torch.tensor(action_bias,
                                   dtype=torch.float32).to(self.device)
        action_scale = (action_upper_bound - action_lower_bound) / 2.0
        action_scale = torch.tensor(action_scale,
                                    dtype=torch.float32).to(self.device)

        self.seed = seed
        self.test_rng = random.Random(self.seed)  # Seed for testing
        _, _ = env.reset(seed=self.seed)  # Seed for training
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        discrete_policy_network = Discrete_Policy_Network(
            self.discrete_obs_dim, self.discrete_action_dim).to(self.device)
        discrete_optimizer = torch.optim.Adam(
            discrete_policy_network.parameters(), lr=lr)
        continuous_policy_networks = [
            Continuous_Policy_Network(self.continuous_obs_dim,
                                      self.continuous_action_dim).to(
                                          self.device)
            for _ in range(self.discrete_action_dim)
        ]
        continuous_optimizers = [
            torch.optim.Adam(net.parameters(), lr=lr)
            for net in continuous_policy_networks
        ]
        self.train_iterations = train_iterations
        self.episode_num = episode_num
        self.batch_size = batch_size
        self.log_output_path = log_output_path
        self.test_iterations = test_iterations

        self.agent = DAgger_Agent(
            discrete_input_dim=self.discrete_obs_dim,
            discrete_output_dim=self.discrete_action_dim,
            continuous_input_dim=self.continuous_obs_dim,
            continuous_output_dim=self.continuous_action_dim,
            continuous_action_bias=action_bias,
            continuous_action_scale=action_scale,
            discrete_policy_network=discrete_policy_network,
            continuous_policy_networks=continuous_policy_networks,
            discrete_optimizer=discrete_optimizer,
            continuous_optimizers=continuous_optimizers,
            device=self.device)

        self.buffer = ReplayBuffer(capacity=buffer_capacity,
                                   device=self.device)

        self.logger = logger(log_output_path=self.log_output_path)
        pass

    def train(self) -> None:
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for idx_episode in range(self.episode_num):
                self.logger.event_logger.info(
                    f"=============== Episode {idx_episode+1:d} of {self.episode_num:d} ================="
                )
                self.train_episode()
                test_avg_rate = self.test()
                pbar.set_postfix({"Test Success Rate": test_avg_rate})
                pbar.update(1)
                pass

    def train_episode(self) -> None:
        options = {
            "Specific_Disturbance": None,
            "Expert_Policy_Required": True,
            "External_RNG": None
        }
        # Ensure that the env has a solution
        while True:
            _, info = self.env.reset(options=options)
            if info["Expert_Policy"] != None:
                break
        # Collect expert experience
        Xd = info["Expert_Policy"]["Branch_Obs"]  # s0-s4
        Yd = info["Expert_Policy"]["TieLine_Action"]  # a1-a5
        Xc = np.vstack((info["Expert_Policy"]["PF"],
                        info["Expert_Policy"]["QF"]))  # s5-s6
        Yc = info["Expert_Policy"]["Q_svc"]
        data = list(zip(Xd.T, Yd, Xc.T, Yc.T))
        for item in data:
            self.buffer.append(item)  # append data to buffer

        for _ in range(self.train_iterations):
            sample_indices = np.random.randint(
                low=0, high=len(self.buffer),
                size=self.batch_size)  # sample data
            Xd_all, Yd_all, Xc_all, Yc_all = self.buffer.fetch_all(
                shuffle=True)  # fetch all data
            Xd_batch = Xd_all[sample_indices]
            Yd_batch = Yd_all[sample_indices]
            Xc_batch = Xc_all[sample_indices]
            Yc_batch = Yc_all[sample_indices]
            # Discrete policy learning
            self.agent.train_discrete(Xd_batch, Yd_batch)
            # Continuous policy learning
            indices = find_1Dtensor_value_indices(Yd_batch,
                                                  self.discrete_action_dim)
            for idx in range(self.discrete_action_dim):
                self.agent.train_continuous(Xc_batch[indices[idx], :],
                                            Yc_batch[indices[idx], :], idx)

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
                s_top_0 = np.reshape(
                    s0["X_branch"],
                    (-1, self.env.unwrapped.system_data.N_Branch))
                a_tieline = self.agent.predict_discrete(s_top_0)
                a_tieline = int(a_tieline.item())
                s_pqf_0 = np.concatenate((s0["PF"], s0["QF"]), axis=0)
                a_svc = self.agent.predict_continuous(s_pqf_0, a_tieline)
                a_svc = a_svc.to("cpu").detach().numpy()
                self.logger.event_logger.info("Given state S[{}] = {}".format(
                    step, s_top_0[0]))
                self.logger.event_logger.info(
                    "Agent action at S[{}] is Ad[{}] = {}, Ac[{}] = {}".format(
                        step, step + 1, a_tieline, step + 1, a_svc))
                a = {"Tieline": a_tieline, "Varcon": a_svc}
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
                success_rate = performance / goal
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
    parser.add_argument('--train_epochs', type=int, default=5000)
    # DAgger learning parameters
    parser.add_argument('--r', type=float, default=1.0e-3)
    parser.add_argument('--update_iters', type=int, default=5)
    # parse arguments
    args = parser.parse_args()

    env = gym.make(id=args.env_id,
                   opt_framework=args.opt_framework,
                   solver=args.solver,
                   data_file=args.data_file,
                   solver_display=args.solver_display,
                   min_disturbance=args.min_disturbance,
                   max_disturbance=args.max_disturbance,
                   vvo=True,
                   Sb=args.Sb,
                   V0=args.V0,
                   V_min=args.V_min,
                   V_max=args.V_max)

    current_path = os.getcwd()
    task_name = 'DAgger_Hybrid'
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
                           train_iterations=args.update_iters,
                           lr=args.lr,
                           test_iterations=args.test_iterations,
                           device=device,
                           seed=args.seed)

    manager.train()
