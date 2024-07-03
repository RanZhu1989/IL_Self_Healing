import os
import collections
import random
from tqdm import tqdm
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd

import selfhealing_env
## --In case of import error when you have to use python-jl to run the code, please use the following import statement--
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --------------------------------------------------------------------------------------------------------------
from utils import logger, check_cuda, get_time
from configs import args

# PPO Agent
class PPO_Clip_Agent():
    """ 
    We use the PPO-Clip algorithm to act as the Policy (generator) in the GAIL framework.
    """
    def __init__(
        self,
        episode_recorder:object,
        actor_network:torch.nn, 
        critic_network:torch.nn,
        actor_optimizer:torch.optim,
        critic_optimizer:torch.optim,
        gamma:float = 0.9,
        advantage_lambda:float = 0.95, # Discount factor for advantage function
        clip_epsilon:float = 0.2, # Clipping parameter
        train_iters:int = 10, # Number of iterations to train the policy in each episode
        device:torch.device = torch.device("cpu")
    ) -> None:
        
        self.device = device
        self.episode_recorder = episode_recorder
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer              
        self.clip_epsilon = clip_epsilon
        self.train_iters = train_iters
        self.gamma = gamma
        self.advantage_lambda = advantage_lambda
        
    def get_action(self, obs:np.ndarray) -> torch.tensor:
        obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
        dist = self.actor_network(obs)
        action_probs = torch.distributions.Categorical(dist)
        picked_action = action_probs.sample()
                       
        return picked_action
    
    def calculate_log_prob(
        self, 
        obs:torch.tensor, 
        action:torch.tensor
    ) -> torch.tensor:
        dist = self.actor_network(obs)
        log_prob = torch.log(dist.gather(1, action))
        
        return log_prob
        
    def calculate_advantage(self, td_error:torch.tensor) -> torch.tensor:
        """ The advantage function is calculated by the TD error. """
        td_error = td_error.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_error[::-1]:
            advantage = self.gamma * self.advantage_lambda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list), dtype = torch.float32).to(self.device)
        
        return advantage
    
    def train_policy(self) -> None:
        obs, action, reward, next_obs , done, = self.episode_recorder.get_trajectory()
        TD_target = reward + self.gamma * self.critic_network(next_obs) * (1 - done)
        TD_error = TD_target - self.critic_network(obs)
        advantage = self.calculate_advantage(TD_error)
        
        old_log_prob = self.calculate_log_prob(obs,action).detach() # Freeze the log_prob obtained by the current policy
        
        for _ in range(self.train_iters):
            critic_loss = torch.mean(F.mse_loss(TD_target.detach(), self.critic_network(obs)))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            log_prob = self.calculate_log_prob(obs, action)
            ratio = torch.exp(log_prob - old_log_prob) # pi_theta / pi_theta_old
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            actor_loss = torch.mean(-torch.min(ratio * advantage, clipped_ratio * advantage))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            
class Actor_Network(torch.nn.Module):

    def __init__(
        self,
        obs_dim:int,
        action_dim:int
    ) -> None:
        super(Actor_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim,64)
        self.fc2 = torch.nn.Linear(64,action_dim)
                    
    def forward(self,x:torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim = -1)
                
        return x
  
    
class Critic_Network(torch.nn.Module):

    def __init__(self,obs_dim:int) -> None:
        super(Critic_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim,64)
        self.fc2 = torch.nn.Linear(64,1)
                    
    def forward(self,x:torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
                
        return x


class Episode_Recorder():
    def __init__(
        self, 
        device:torch.device = torch.device("cpu")
    ) -> None:
        self.device = device
        self.reset()
        
    def append(
        self, 
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        obs = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array([[action]]), dtype=torch.int64).to(self.device)
        next_obs = torch.tensor(np.array([next_obs]), dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array([[done]]), dtype=torch.float32).to(self.device)
        self.trajectory["obs"] = torch.cat((self.trajectory["obs"], obs))
        self.trajectory["action"] = torch.cat((self.trajectory["action"], action))
        self.trajectory["next_obs"] = torch.cat((self.trajectory["next_obs"], next_obs))
        self.trajectory["done"] = torch.cat((self.trajectory["done"], done))
        
    def get_trajectory(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        
        return self.trajectory["obs"], self.trajectory["action"], \
            self.trajectory["reward"], self.trajectory["next_obs"], self.trajectory["done"]
                
    def reset(self) -> None:
        """ Clear the trajectory when begin a new episode."""
        self.trajectory = {
            "obs": torch.tensor([], dtype = torch.float32).to(self.device),
            "action": torch.tensor([], dtype = torch.int64).to(self.device),
            "reward": torch.tensor([], dtype = torch.float32).to(self.device),
            "next_obs": torch.tensor([], dtype = torch.float32).to(self.device),
            "done": torch.tensor([], dtype = torch.float32).to(self.device)
        }
        
    def add_reward(self, reward:np.ndarray) -> None:
        self.trajectory["reward"] = torch.tensor(reward, dtype=torch.float32).to(self.device)


# Expert experience collector
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
        
    def sample(self,
               batch_size:int,
               shuffle:bool = True
               ) -> Tuple[torch.tensor,torch.tensor]:
        batch_size = min(batch_size, len(self.buffer)) # in case the buffer is not enough
        if shuffle:
            self.buffer = random.sample(self.buffer,len(self.buffer))        
        mini_batch = random.sample(self.buffer,batch_size)
        obs_batch, action_batch = zip(*mini_batch)
        obs_batch = torch.tensor(np.array(obs_batch),dtype=torch.float32,device=self.device)
        action_batch = torch.tensor(np.array(action_batch),dtype=torch.int64,device=self.device) 
          
        return obs_batch, action_batch
    
    def fetch_all(self,
                  shuffle:bool = True
                  ) -> Tuple[torch.tensor,torch.tensor]:
        if shuffle:
            self.buffer = random.sample(self.buffer,len(self.buffer))
        obs_batch, action_batch = zip(*self.buffer)
        obs_batch = torch.tensor(np.array(obs_batch),dtype=torch.float32,device=self.device)
        action_batch = torch.tensor(np.array(action_batch),dtype=torch.int64,device=self.device) 
          
        return obs_batch, action_batch
    
    
    def release(self) -> None:
        self.buffer.clear()
        
    def __len__(self) -> int:
        return len(self.buffer)


# GAIL algorithm
class GAIL():
    def __init__(
        self,
        action_dim:int,
        d_network:torch.nn,
        d_optimizer:torch.optim,
        train_iters:int
    ) -> None:
        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.action_dim = action_dim
        self.train_iters = train_iters
        
    def train_d(self, 
                exp_obs:torch.tensor,
                exp_action:torch.tensor,
                agent_obs:torch.tensor,
                agent_action:torch.tensor
                ):

        exp_action = F.one_hot(exp_action.to(torch.int64), num_classes=self.action_dim).to(torch.float32).squeeze()
        agent_action = F.one_hot(agent_action.to(torch.int64), num_classes=self.action_dim).to(torch.float32).squeeze()
        for _ in range(self.train_iters):
            exp_prob = self.d_network(exp_obs, exp_action)
            agent_prob = self.d_network(agent_obs, agent_action)
            loss = torch.nn.BCELoss()(
                agent_prob, torch.ones_like(agent_prob)) + torch.nn.BCELoss()(
                    exp_prob, torch.zeros_like(exp_prob))
                
            self.d_optimizer.zero_grad()
            loss.backward()
            self.d_optimizer.step()
            pass
        
        gail_reward = -torch.log(self.d_network(agent_obs, agent_action)).detach().cpu().numpy()
        
        return gail_reward
        
    
class Discriminator(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim + action_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))
    

class TrainManager():
    def __init__(
        self,
        env:gym.Env,
        log_output_path:Optional[str]=None,
        episode_num:int = 1000,
        expert_sample_used:int = 50, # use # samples to train the agent
        discriminator_train_iters = 5,
        test_iterations:int = 5,
        d_lr:float = 1e-3,
        buffer_capacity:int = 10000,
        actor_lr:float = 1e-4,
        critic_lr:float = 1e-3,
        gamma:float = 0.9,
        advantage_lambda:float = 0.95,
        clip_epsilon:float = 0.2,
        ppo_train_iters:int = 10, 
        seed:int = 0,
        device:torch.device = torch.device("cpu") 
    ) -> None:
        
        self.seed = seed 
        self.test_rng = random.Random(self.seed+1)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.device = device
        
        self.log_output_path = log_output_path
        self.logger = logger(log_output_path=self.log_output_path)
        self.expert_sample_used = expert_sample_used
        self.test_iterations = test_iterations
        self.discriminator_train_iters = discriminator_train_iters
        
        self.env = env
        _,_ = self.env.reset(seed=self.seed)
        
        self.episode_num = episode_num
        obs_dim = gym.spaces.utils.flatdim(env.observation_space) 
        action_dim = env.action_space.n
        episode_recorder = Episode_Recorder(device = self.device)
        actor_network = Actor_Network(obs_dim,action_dim).to(self.device)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(),lr=actor_lr)
        critic_network = Critic_Network(obs_dim).to(self.device)
        critic_optimizer = torch.optim.Adam(critic_network.parameters(),lr=critic_lr)
        self.agent = PPO_Clip_Agent(
            episode_recorder = episode_recorder,
            actor_network = actor_network, 
            critic_network = critic_network,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer,
            gamma = gamma,
            advantage_lambda = advantage_lambda,
            clip_epsilon = clip_epsilon,
            train_iters = ppo_train_iters,
            device = self.device
        )
        
        self.buffer = ReplayBuffer(capacity=buffer_capacity,device=self.device)
        d_network = Discriminator(obs_dim, action_dim).to(self.device)
        d_optimizer = torch.optim.Adam(d_network.parameters(), lr=d_lr)
        self.gail = GAIL(action_dim, d_network, d_optimizer, self.discriminator_train_iters)
        
        self.episode_total_rewards = np.zeros(self.episode_num)
        self.index_episode = 0
        
    def train(self) -> None:
        # 1. Collect expert data
        self.logger.event_logger.info(
                    f"=============== Expert Data Collection ================="
                )
        options = {
            "Specific_Disturbance":None, 
            "Expert_Policy_Required":True, 
            "External_RNG":None
            }
        sampled_idx = 0
        with tqdm(total=self.expert_sample_used, desc='Collecting Expert Data') as pbar:
            while sampled_idx < self.expert_sample_used:
                # Ensure that the env has a solution
                while True:
                    _, info = self.env.reset(options=options)
                    if info["Expert_Policy"] != None:
                        sampled_idx += 1
                        break
                # Collect expert experience
                X = info["Expert_Policy"]["Branch_Obs"] # s0-s4
                Y = info["Expert_Policy"]["TieLine_Action"] # a1-a5
                data = list(zip(X.T,Y))
                for item in data:
                    self.buffer.append(item) # append data to buffer
                pbar.update(1)
        
        # 2. Train the target PPO agent via interacting with the environment
        
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for e in range(self.episode_num):
                self.logger.event_logger.info(
                    f"=============== Episode {e+1:d} of {self.episode_num:d} ================="
                )
                episode_reward = self.train_episode()
                self.test()
                pbar.set_postfix({"Return": (episode_reward)})
                pbar.update(1)
    
    def train_episode(self) -> float:
        # Start a new episode
        options = {
            "Specific_Disturbance":None, 
            "Expert_Policy_Required":True, 
            "External_RNG":None
            }
        self.agent.episode_recorder.reset()
        obs = None
        while True:
            obs, info = self.env.reset(options=options)
            if info["Expert_Policy"] != None:
                break
            
        total_reward = 0
        while True:
            action = self.agent.get_action(obs).item() 
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.episode_recorder.append(obs, action, next_obs, done)
            total_reward += reward 
            obs = next_obs              
            if done:
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                break
        # Get the trajectory
        agent_obs, agent_action, _, _, _ = self.agent.episode_recorder.get_trajectory()
        
        # Set the temporary reward to None
        reward = None            

        # Train the discriminator to get the reward
        # obs_batch, action_batch = self.buffer.sample(batch_size=self.batch_size) # sample data
        obs_batch, action_batch = self.buffer.fetch_all(shuffle=True) # fetch all data
        reward = self.gail.train_d(
                        exp_obs = obs_batch,
                        exp_action = action_batch,
                        agent_obs = agent_obs,
                        agent_action = agent_action
                    )
        
        # Update the reward
        self.agent.episode_recorder.add_reward(reward)
        
        # Train the target PPO agent
        self.agent.train_policy()
        
        return total_reward
        
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
                a = self.agent.get_action(s0)
                a = int(a)
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
                success_rate = min(performance/goal,1)
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
    
    def plotting(self,smoothing_window:int = 100) -> None:
        """ Plot the episode reward over time. """
        fig = plt.figure(figsize=(10,5))
        plt.plot(self.episode_total_rewards,label="Episode Reward")
        # Use rolling mean to smooth the curve
        rewards_smoothed = pd.Series(self.episode_total_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed,label="Episode Reward (Smoothed)")
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title("Episode Reward over Time")
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    env = gym.make(
        id=args.env_id,
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
        V_max=args.V_max
    )
    
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
    
    manager = TrainManager(
        env = env,
        log_output_path = log_output_path,
        test_iterations=args.test_iterations,
        expert_sample_used = args.IL_used_samples, 
        discriminator_train_iters=args.GAIL_d_iters,
        episode_num = args.train_epochs,
        d_lr = args.GAIL_d_lr,
        actor_lr = args.PPO_actor_lr,
        critic_lr = args.PPO_critic_lr,
        gamma = args.PPO_gamma,
        advantage_lambda = args.PPO_advantage_lambda,
        clip_epsilon = args.PPO_clip_epsilon,
        ppo_train_iters = args.PPO_train_iters,
        seed = args.seed,
        device = device
    )
    
    manager.logger.event_logger.info(
            f"=============== Task Info =================")
    manager.logger.event_logger.info(
        f"ENV Settings == {args.env_id}, Device == {device}, Seed == {args.seed}")
    manager.logger.event_logger.info(
        f"Task == {task_name}, Training Epochs == {args.train_epochs}, Test Iterations == {args.test_iterations}")
    manager.logger.event_logger.info(
        f"Opt_Framework == {args.opt_framework}, Solver == {args.solver}, system_file == {args.data_file}")
    manager.logger.event_logger.info(
        f"min_disturbance == {args.min_disturbance}, max_disturbance == {args.max_disturbance}")
    
    manager.train()
    manager.plotting(smoothing_window = 10)   