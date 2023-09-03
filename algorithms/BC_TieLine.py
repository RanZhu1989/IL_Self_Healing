import os
import time
import collections
import random
from tqdm import tqdm
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import selfhealing_env
from utils import logger

class BC_Agent():
    """Behavioral Cloning Agent"""
    def __init__(self,
        input_dim:int,
        output_dim:int,
        policy_network:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        device:torch.device = torch.device("cpu")
    ) -> None:
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.device = device
        
    def predict(self, obs:np.ndarray)->torch.tensor:
        
        obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
        dist = self.policy_network(obs)
        # action_probs = torch.distributions.Categorical(dist)
        # picked_action = action_probs.sample()
        picked_action = torch.argmax(dist)

        return picked_action
    
    def learn(self, 
        obs:torch.tensor, 
        action:torch.tensor
    ) -> None:
        
        log_prob = torch.log(self.policy_network(obs).gather(1, action))
        loss = torch.mean(-log_prob)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
class Policy_Network(torch.nn.Module):
    def __init__(
        self, input_dim:int, output_dim:int
    ) -> None:
        super(Policy_Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_dim)
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)


class ReplayBuffer():
    """Expert experience collector"""
    def __init__(
        self, 
        capacity:int, 
        device:torch.device = torch.device("cpu")
    ) -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)
        
    def append(self,exp_data:tuple) -> None:
        self.buffer.append(exp_data)
        
    def sample(self,batch_size:int) -> Tuple[torch.tensor,torch.tensor]:
        batch_size = min(batch_size, len(self.buffer)) # in case the buffer is not enough        
        mini_batch = random.sample(self.buffer,batch_size)
        obs_batch, action_batch = zip(*mini_batch)
        obs_batch = torch.tensor(np.array(obs_batch),dtype=torch.float32,device=self.device)
        action_batch = torch.tensor(np.array(action_batch),dtype=torch.int64,device=self.device) 
          
        return obs_batch, action_batch
    
    def fetch_all(self) -> Tuple[torch.tensor,torch.tensor]:
        obs_batch, action_batch = zip(*self.buffer)
        obs_batch = torch.tensor(np.array(obs_batch),dtype=torch.float32,device=self.device)
        action_batch = torch.tensor(np.array(action_batch),dtype=torch.int64,device=self.device) 
          
        return obs_batch, action_batch
    
    def release(self) -> None:
        self.buffer.clear()
        
    def __len__(self) -> int:
        return len(self.buffer)


class TrainManager():
    def __init__(
        self,
        env:gym.Env,
        log_output_path:Optional[str]=None,
        lr:float = 0.001,
        train_iterations:int = 10,
        episode_num:int = 200,
        buffer_capacity:int = 1000000,
        batch_size:int = 32,
        test_iterations:int = 5,
        device:torch.device = torch.device("cpu"),
        seed:Optional[int] = None
    ) -> None:
        
        self.env = env
        self.obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.action_dim = env.action_space.n
        
        self.seed = seed
        self.test_rng = random.Random(self.seed) # Seed for testing
        _,_ = env.reset(seed=self.seed) # Seed for training
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        self.device = device
        policy_network = Policy_Network(self.obs_dim, self.action_dim).to(self.device)
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
        self.train_iterations = train_iterations
        self.episode_num = episode_num
        self.batch_size = batch_size
        self.log_output_path = log_output_path
        self.test_iterations = test_iterations
        
        self.agent = BC_Agent(
            input_dim = self.obs_dim, output_dim = self.action_dim,
            policy_network = policy_network, optimizer = optimizer
        )
        
        self.buffer = ReplayBuffer(capacity=buffer_capacity,device=self.device)
        
        self.logger = logger(log_output_path=self.log_output_path)
        pass
        
    def train(self) -> None:
        tic = time.perf_counter()
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for idx_episode in range(self.episode_num):
                self.logger.event_logger.info(
                    f"=============== Episode {idx_episode+1:d} of {self.episode_num:d} ================="
                )
                self.train_episode()
                self.test()
                toc = time.perf_counter()
                pbar.set_postfix({"Training time": f"{toc - tic:0.2f} seconds"})
                pbar.update(1)
                          
    def train_episode(self) -> None:
        options = {
            "Specific_Disturbance":None, 
            "Expert_Policy_Required":True, 
            "External_RNG":None
        }
        # Ensure that the env has a solution
        while True:
            _, info = self.env.reset(options=options)
            if info["Expert_Policy"] != None:
                break
        # Collect expert experience
        X = info["Expert_Policy"]["Branch_Obs"] # s0-s4
        Y = info["Expert_Policy"]["TieLine_Action"] # a1-a5
        data = list(zip(X.T,Y))
        for item in data:
            self.buffer.append(item) # append data to buffer
        
        for _ in range(self.train_iterations):
            # obs_batch, action_batch = self.buffer.sample(batch_size=self.batch_size) # sample data
            obs_batch, action_batch = self.buffer.fetch_all() # fetch all data
            self.agent.learn(obs_batch,action_batch)
       
    def test(self) -> None:
        options = {
            "Specific_Disturbance":None,
            "Expert_Policy_Required":True,
            "External_RNG":self.test_rng
        }
        self.logger.event_logger.info("-------------------Test Report--------------------")
        saved_disturbance_set = [] # Save data
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
            self.logger.event_logger.info("# Test {}".format(test_idx+1))
            self.logger.event_logger.info("The testing disturbance is {}".format(sorted(test_disturbance_set)))
            # Calculate goal using expert policy
            load_rate_s0 = expert_info["Recovered_Load_Rate_S0"]
            expert_load_rate = expert_info["Expert_Policy"]["Load_Rate"]
            # goal = sum[optimal load rate] - sum[load rate at s0] * total time steps
            goal = np.sum(expert_load_rate).item() - load_rate_s0 * len(expert_load_rate) 
            self.logger.event_logger.info(
                "Expert policy is {}".format(expert_info["Expert_Policy"]["TieLine_Action"].T.tolist())
            )
            # Calculate performance using agent policy           
            agent_info = None
            agent_load_rate = []
            for step in self.env.exploration_seq_idx:
                s0 = np.reshape(s0, (-1, self.env.system_data.N_Branch))
                a = self.agent.predict(s0)
                a = int(a.item())
                self.logger.event_logger.info("Given state S[{}] = {}".format(step, s0[0]))
                self.logger.event_logger.info("Agent action at S[{}] is A[{}] = {}".format(step, step+1, a))
                s, reward, _, _, agent_info = self.env.step(a)
                load_rate = agent_info["Recovered_Load_Rate"]
                agent_load_rate.append(load_rate)
                self.logger.event_logger.info("Event at S[{}] is '{}'".format(step+1, agent_info["Event_Log"]))
                self.logger.event_logger.info(
                    "Reward R[{}] is {}. Load rate at S[{}] is {}.".format(step+1, reward,step+1, load_rate)
                )
                s0 = s
            # Evaluate agent performance
            self.logger.event_logger.info("Load rate at S0 is {}".format(load_rate_s0))
            # performance = sum[load rate] - sum[load rate at s0] * total time steps
            performance = np.sum(agent_load_rate).item() - load_rate_s0*len(agent_load_rate)
            if abs(goal) > 1e-6:
                success_rate = performance/goal
                self.logger.event_logger.info("performance: {}; goal: {}".format(performance, goal)) 
            else:
                success_rate = 1 # if goal is too small, we consider it as success
                self.logger.event_logger.info(
                    "Nothing we can do to improve the load profile. (Reason: performance={}, goal={})".format(performance,goal)
                )
            
            saved_disturbance_set.append(test_disturbance_set)
            saved_agent_load_rate.append(agent_load_rate)
            saved_expert_load_rate.append(expert_load_rate)
            saved_success_rate.append(success_rate)
        
        # Save data to file
        self.logger.save_to_file(
            disturb=saved_disturbance_set, 
            agent_recovery_rate=np.array(saved_agent_load_rate), 
            expert_recovery_rate=np.array(saved_expert_load_rate), 
            success_rate=np.array(saved_success_rate)
        )
        pass


if __name__ == "__main__":
    current_path = os.getcwd()
    log_output_path = current_path + "/results/BC_TieLine/n_1/"
    tensorboard_path = log_output_path + "tensorboard/"

    env = gym.make(
        "SelfHealing-v0",
        opt_framework="JuMP",
        solver="cplex",
        data_file="Case_33BW_Data.xlsx",
        solver_display=False,
        min_disturbance=1,
        max_disturbance=1,
        vvo=False,
        Sb=100,
        V0=1.05,
        V_min=0.95,
        V_max=1.05
    )
    
    manager = TrainManager(
        env=env,
        log_output_path = log_output_path,
        episode_num=200,
        train_iterations=100,
        lr=1e-3,
        test_iterations=5, 
        device=torch.device("cpu"),
        seed=0
    )
    
    manager.train()
    