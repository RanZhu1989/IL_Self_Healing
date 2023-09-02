import time
import os
import collections
import gymnasium as gym
import selfhealing_env
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from utils import logger
from tqdm import tqdm

class Agent():
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 policy_network:torch.nn.Module,
                 optimizer:torch.optim.Optimizer,
                 device:torch.device = torch.device("cpu")
    ):
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
    
    def learn(self, obs:torch.tensor, action:torch.tensor) -> None:
        log_prob = torch.log(self.policy_network(obs).gather(1, action))
        loss = torch.mean(-log_prob)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
class Policy_Network(torch.nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int
                 ):
        super(Policy_Network, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)


class ReplayBuffer():
    def __init__(self,capacity:int,device:torch.device = torch.device("cpu")) -> None:
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
    def __init__(self,
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
    ):
        self.env = env
        self.device = device
        self.train_iterations = train_iterations
        self.episode_num = episode_num
        self.seed = seed
        self.test_rng = random.Random(self.seed)
        
        _,_ = env.reset(seed=self.seed) # 设置训练环境的随机种子
        
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.batch_size = batch_size
        self.log_output_path = log_output_path
        self.test_iterations = test_iterations
        
        self.obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.action_dim = env.action_space.n
        policy_network = Policy_Network(self.obs_dim, self.action_dim).to(self.device)
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
        
        self.agent = Agent(input_dim = self.obs_dim,
                           output_dim = self.action_dim,
                           policy_network = policy_network,
                           optimizer = optimizer
                           )
        self.buffer = ReplayBuffer(capacity=buffer_capacity,device=self.device)
        
        self.logger = logger(log_output_path=self.log_output_path)
        
        
    def train(self):
        tic = time.perf_counter()  # start clock
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for idx_episode in range(self.episode_num):
                self.logger.event_logger.info(f"=============== Episode {idx_episode+1:d} of {self.episode_num:d} =================")
                self.train_episode()
                self.test()
                toc = time.perf_counter()
                pbar.set_postfix({"Training time": f"{toc - tic:0.4f} seconds"})
                pbar.update(1)
                          
    def train_episode(self):
        options = {"Specific_Disturbance":None, "Expert_Policy_Required":True, "External_RNG":None}
        obs, info = None, None
        # 确保有解
        while True:
            obs, info = self.env.reset(options=options)
            if info["Expert_Policy"] != None:
                break
        
        # 制作训练集
        data_x = info["Expert_Policy"]["Branch_Obs"] # s0-s4
        data_y = info["Expert_Policy"]["TieLine_Action"] # a1-a5
        data = list(zip(data_x.T,data_y))
        for item in data:
            self.buffer.append(item) # 将数据加入buffer
        
        for _ in range(self.train_iterations):
            # obs_batch, action_batch = self.buffer.sample(batch_size=self.batch_size) # 从buffer中拿数据
            obs_batch, action_batch = self.buffer.fetch_all() # 从buffer中拿数据
            self.agent.learn(obs_batch,action_batch) # 训练
       
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
                a = self.agent.predict(s0)  # this action is one-hot encoded
                a = int(a.item()) # 将tensor转换为int
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

if __name__ == "__main__":
    current_path = os.getcwd()
    log_output_path = current_path + "/results/BC_TieLine/n_1/"
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
    manager = TrainManager(env=env,
                           log_output_path = log_output_path,
                           episode_num=200,
                           train_iterations=100,
                           lr=1e-3,
                           test_iterations=10, 
                           device=torch.device("cpu"),
                           seed=0)
    manager.train()
    
    


