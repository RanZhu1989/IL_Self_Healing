import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import selfhealing_env
import time
import os
import random
from typing import Tuple, Optional, List, Dict, Any
from utils import logger
from tqdm import tqdm

class PPO_Clip_Agent():
    """ Thanks to the discrete actions, we can simply represent the action by a number. """
    def __init__(self,
                 episode_recorder:object,
                 actor_network:torch.nn, 
                 critic_network:torch.nn,
                 actor_optimizer:torch.optim,
                 critic_optimizer:torch.optim,
                 gamma:float = 0.9,
                 advantage_lambda:float = 0.95,
                 clip_epsilon:float = 0.2, 
                 train_iters:int = 10, 
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
        dist = self.actor_network(obs) # NN_outputs -softmax-> Action distribution
        # dist = F.softmax(dist, dim = 0) 
        action_probs = torch.distributions.Categorical(dist)
        picked_action = action_probs.sample()
                       
        return picked_action
    
    def calculate_log_prob(self, obs:torch.tensor, action:torch.tensor) -> torch.tensor:
        # dist = F.softmax(self.actor_network(obs), dim=1)
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
            clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
            actor_loss = torch.mean(-torch.min(ratio * advantage, clipped_ratio * advantage))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
class Actor_Network(torch.nn.Module):

    def __init__(self,obs_dim:int,action_dim:int) -> None:
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
    
    def __init__(self, device:torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.reset()
        
    def append(self, 
               obs: np.ndarray,
               action: int,
               reward:float,
               next_obs: np.ndarray,
               done: bool
               ) -> None:
        obs = torch.tensor(np.array([obs]), 
                           dtype = torch.float32).to(self.device)
        action = torch.tensor(np.array([[action]]), 
                              dtype = torch.int64).to(self.device)
        reward = torch.tensor(np.array([[reward]]), 
                              dtype = torch.float32).to(self.device)
        next_obs = torch.tensor(np.array([next_obs]), 
                                dtype = torch.float32).to(self.device)
        done = torch.tensor(np.array([[done]]), 
                            dtype = torch.float32).to(self.device)
        self.trajectory["obs"] = torch.cat((self.trajectory["obs"], obs))
        self.trajectory["action"] = torch.cat((self.trajectory["action"], action))
        self.trajectory["reward"] = torch.cat((self.trajectory["reward"], reward))
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

class TrainManager():
    
    def __init__(self,
                 env:gym.Env,
                 log_output_path:Optional[str]=None,
                 episode_num:int = 1000,
                 test_iterations:int = 5,
                 actor_lr:float = 1e-4,
                 critic_lr:float = 1e-3,
                 gamma:float = 0.9,
                 advantage_lambda:float = 0.95,
                 clip_epsilon:float = 0.2,
                 train_iters:int = 10, 
                 seed = 0,
                 my_device = "cpu"
                 ) -> None:
        
        self.seed = seed 
        self.test_rng = random.Random(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.device = torch.device(my_device)
        
        self.log_output_path = log_output_path
        self.logger = logger(log_output_path=self.log_output_path)
        self.test_iterations = test_iterations
        
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
        self.agent = PPO_Clip_Agent(episode_recorder = episode_recorder,
                                    actor_network = actor_network, 
                                    critic_network = critic_network,
                                    actor_optimizer = actor_optimizer,
                                    critic_optimizer = critic_optimizer,
                                    gamma = gamma,
                                    advantage_lambda = advantage_lambda,
                                    clip_epsilon = clip_epsilon,
                                    train_iters = train_iters,
                                    device = self.device
                                    )
                                            
        self.episode_total_rewards = np.zeros(self.episode_num)
        self.index_episode = 0
        
    def train_episode(self) -> float:
        total_reward = 0
        self.agent.episode_recorder.reset()
        obs,_ = self.env.reset() 
        while True:
            action = self.agent.get_action(obs).item() 
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.agent.episode_recorder.append(obs, action, reward, next_obs, done)
            total_reward += reward 
            obs = next_obs              
            if done:
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                break
        
        self.agent.train_policy()
        
        return total_reward
    
    def train(self) -> None:
        tic = time.perf_counter()  # start clock
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for e in range(self.episode_num):
                episode_reward = self.train_episode()
                self.test()
                toc = time.perf_counter()
                pbar.set_postfix({"Training time": f"{toc - tic:0.4f} seconds", "Return": (episode_reward)})
                pbar.update(1)
            
    def test(self):
        # reset env and using trained policy
        test_options = {"Specific_Disturbance":None,
                        "Expert_Policy_Required":True,
                        "External_RNG":self.test_rng
                        }
        # 找到一个有解的场景
        self.logger.event_logger.info("-------------------Test Report--------------------")
        # 多次测试
        saved_disturbance_set = []
        saved_agent_load_rate = []
        saved_expert_load_rate = []
        saved_success_rate = []
        
        for test_idx in range(self.test_iterations):            
            while True:
                s0, expert_info = self.env.reset(options=test_options)
                if expert_info["Expert_Policy"] != None:
                    break
            test_disturbance_set = expert_info["Disturbance_Set"]
            # ================ calculate Benchmark value to normalize the restoration as ratio ==================
            self.logger.event_logger.info("# Test {}".format(test_idx+1))
            self.logger.event_logger.info("The testing disturbance is {}".format(sorted(test_disturbance_set)))
            # 计算goal
            load_rate_s0 = expert_info["Recovered_Load_Rate_S0"]
            expert_load_rate = expert_info["Expert_Policy"]["Load_Rate"]
            goal = np.sum(expert_load_rate).item() - load_rate_s0*len(expert_load_rate) 
            self.logger.event_logger.info("Expert policy is {}".format(expert_info["Expert_Policy"]["TieLine_Action"].T.tolist())) # 记录专家策略
                        
            # ============== run agent using trained policy approximator ==================
            # 用智能体在这个env中输出动作，记录负荷恢复量
            agent_info = None
            agent_load_rate = []
            for step in self.env.exploration_seq_idx:
                s0 = np.reshape(s0, (-1, self.env.system_data.N_Branch)) # 将37维列向量转换为1*37的矩阵，-1是自动计算行数
                a = self.agent.get_action(s0)  # this action is one-hot encoded
                a = int(a) # 将tensor转换为int
                self.logger.event_logger.info("Given state S[{}] = {}".format(step, s0[0]))
                self.logger.event_logger.info("Agent action at S[{}] is A[{}] = {}".format(step, step+1, a))
                s, reward, _, _, agent_info = self.env.step(a) # 此时动作其实是一个只有单个元素的np.ndarray
                load_rate = agent_info["Recovered_Load_Rate"]
                agent_load_rate.append(load_rate) # 保存当前load rate
                self.logger.event_logger.info("Event at S[{}] is '{}'".format(step+1, agent_info["Event_Log"]))
                self.logger.event_logger.info("Reward R[{}] is {}. Load rate at S[{}] is {}.".format(step+1, reward,step+1, load_rate))
                # update state
                s0 = s
                            
            # ================ evaluate success =====================
            self.logger.event_logger.info("Load rate at S0 is {}".format(load_rate_s0)) # 记录初始负荷恢复量
            # performance = agent_load_rate[-1] - load_rate_s0 # 智能体策略回复提升量(归一化)
            performance = np.sum(agent_load_rate).item() - load_rate_s0*len(agent_load_rate)
            success_rate = performance/goal
            if abs(goal) > 1e-4:
                self.logger.event_logger.info("performance: {}; goal: {}".format(performance, goal)) 
            else:
                self.logger.event_logger.info("Nothing we can do to improve the load profile. (Reason: performance={}, goal={})".format(performance,goal)) # 差异极小，系统没救了，没有自愈的必要
            
            # ================ save data =====================
            saved_disturbance_set.append(test_disturbance_set)
            saved_agent_load_rate.append(agent_load_rate)
            saved_expert_load_rate.append(expert_load_rate)
            saved_success_rate.append(success_rate)
            
        self.logger.save_to_file(disturb=saved_disturbance_set, 
                                 agent_recovery_rate=np.array(saved_agent_load_rate), 
                                 expert_recovery_rate=np.array(saved_expert_load_rate), 
                                 success_rate=np.array(saved_success_rate))
                        
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
    
    current_path = os.getcwd()
    log_output_path = current_path + "/results/PPO_TieLine/n_1/"
    tensorboard_path = log_output_path + "tensorboard/"
    
    env = gym.make("SelfHealing-v0",
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
    
    Manger = TrainManager(env = env,
                          log_output_path = log_output_path,
                          test_iterations=5,
                        episode_num = 200,
                        actor_lr = 1e-4,
                        critic_lr = 1e-2,
                        gamma = 0.90,
                        advantage_lambda = 0.95,
                        clip_epsilon = 0.2,
                        train_iters = 10, 
                        seed = 0,
                        my_device = "cpu" 
                        )
    Manger.train()
    # Manger.plotting(smoothing_window = 10)    