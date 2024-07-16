import os
import time
import collections
import random
from tqdm import tqdm
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np

import selfhealing_env

import torch
import torch.nn.functional as F
from utils import logger, check_cuda, get_time, find_1Dtensor_value_indices
from configs import args

class BC_Agent():
    """Behavioral Cloning Agent"""
    def __init__(self,
        discrete_input_dim:int,
        discrete_output_dim:int,
        continuous_input_dim:int,
        continuous_output_dim:int,
        continuous_action_bias:torch.tensor,
        continuous_action_scale:torch.tensor,
        discrete_policy_network:torch.nn.Module,
        continuous_policy_networks:list,
        discrete_optimizer:torch.optim.Optimizer,
        continuous_optimizers:list,
        device:torch.device = torch.device("cpu")
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
        
    def predict_discrete(self, obs:np.ndarray)->torch.tensor:
        obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
        dist = self.discrete_policy_network(obs)
        picked_action = torch.argmax(dist)

        return picked_action
    
    def predict_continuous(self, 
                           obs:np.ndarray,
                           upper_idx:int
                           )->torch.tensor:
  
        obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
        out = self.continuous_policy_networks[upper_idx](obs)
        action = out * self.continuous_action_scale + self.continuous_action_bias
        
        return action
    
    def train_discrete(self, 
        obs:torch.tensor, 
        action:torch.tensor
    ) -> None:
        
        log_prob = torch.log(self.discrete_policy_network(obs).gather(1, action))
        loss = torch.mean(-log_prob)
        self.discrete_optimizer.zero_grad()
        loss.backward()
        self.discrete_optimizer.step()
        
    def train_continuous(self,
        obs:torch.tensor,
        action:torch.tensor,
        upper_idx:int
    ) -> None:
        action_hat = self.continuous_policy_networks[upper_idx](obs)
        loss = F.mse_loss(action_hat, action)
        self.continuous_optimizers[upper_idx].zero_grad()
        loss.backward()
        self.continuous_optimizers[upper_idx].step()

class Discrete_Policy_Network(torch.nn.Module):
    def __init__(
        self, input_dim:int, output_dim:int
    ) -> None:
        super(Discrete_Policy_Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, output_dim)
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)

class Continuous_Policy_Network(torch.nn.Module):
    def __init__(
        self, input_dim:int, output_dim:int
    ) -> None:
        super(Continuous_Policy_Network, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, output_dim)
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.tanh(self.fc5(x))

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
               ) -> Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        batch_size = min(batch_size, len(self.buffer)) # in case the buffer is not enough
        if shuffle:
            self.buffer = random.sample(self.buffer,len(self.buffer))     
        mini_batch = random.sample(self.buffer,batch_size)
        Xd_batch, Yd_batch, Xc_batch, Yc_batch = zip(*mini_batch) # discrete_obs, discrete_action, continuous_obs, continuous_action
        Xd_batch = torch.tensor(np.array(Xd_batch),dtype=torch.float32,device=self.device)
        Yd_batch = torch.tensor(np.array(Yd_batch),dtype=torch.int64,device=self.device)
        Xc_batch = torch.tensor(np.array(Xc_batch),dtype=torch.float32,device=self.device)
        Yc_batch = torch.tensor(np.array(Yc_batch),dtype=torch.float32,device=self.device)
          
        return Xd_batch, Yd_batch, Xc_batch, Yc_batch
    
    def fetch_all(self) -> Tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        Xd_batch, Yd_batch, Xc_batch, Yc_batch = zip(*self.buffer)
        Xd_batch = torch.tensor(np.array(Xd_batch),dtype=torch.float32,device=self.device)
        Yd_batch = torch.tensor(np.array(Yd_batch),dtype=torch.int64,device=self.device)
        Xc_batch = torch.tensor(np.array(Xc_batch),dtype=torch.float32,device=self.device)
        Yc_batch = torch.tensor(np.array(Yc_batch),dtype=torch.float32,device=self.device)
          
        return Xd_batch, Yd_batch, Xc_batch, Yc_batch
    
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
        training_epochs:int = 500,
        expert_episode_used:int = 100,
        expert_sample_used:int = 50,
        buffer_capacity:int = 1000000,
        batch_size:int = 32,
        train_iters:int = 5,
        test_iterations:int = 5,
        device:torch.device = torch.device("cpu"),
        seed:Optional[int] = None
    ) -> None:
        
        self.env = env
        self.discrete_obs_dim = gym.spaces.utils.flatdim(env.observation_space["X_branch"])
        self.discrete_action_dim = gym.spaces.utils.flatdim(env.action_space["Tieline"])
        self.continuous_obs_dim = gym.spaces.utils.flatdim(env.observation_space["PF"]) + gym.spaces.utils.flatdim(env.observation_space["QF"])
        self.continuous_action_dim = gym.spaces.utils.flatdim(env.action_space["Varcon"])
        self.device = device
        
        action_upper_bound = self.env.action_space["Varcon"].high
        action_lower_bound = self.env.action_space["Varcon"].low
        action_bias = (action_upper_bound + action_lower_bound) / 2.0
        action_bias = torch.tensor(action_bias,dtype=torch.float32).to(self.device)
        action_scale = (action_upper_bound - action_lower_bound) / 2.0
        action_scale = torch.tensor(action_scale,dtype=torch.float32).to(self.device)
        
        
        self.seed = seed
        self.test_rng = random.Random(self.seed) # Seed for testing
        _,_ = env.reset(seed=self.seed) # Seed for training
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        
        discrete_policy_network = Discrete_Policy_Network(self.discrete_obs_dim, self.discrete_action_dim).to(self.device)
        discrete_optimizer = torch.optim.Adam(discrete_policy_network.parameters(), lr=lr)
        continuous_policy_networks = [Continuous_Policy_Network(self.continuous_obs_dim, 
                                    self.continuous_action_dim).to(self.device) for _ in range(self.discrete_action_dim)]
        continuous_optimizers = [torch.optim.Adam(net.parameters(), lr=lr) for net in continuous_policy_networks]
        self.training_epochs = training_epochs
        self.train_iters = train_iters
        self.expert_episode_used = expert_episode_used
        self.expert_sample_used = expert_sample_used
        self.batch_size = batch_size
        self.log_output_path = log_output_path
        self.test_iterations = test_iterations
        
        self.agent = BC_Agent(
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
            device=self.device
        )
        
        self.buffer = ReplayBuffer(capacity=buffer_capacity,device=self.device)
        
        self.logger = logger(log_output_path=self.log_output_path)
        pass

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
        with tqdm(total=self.expert_episode_used, desc='Collecting Expert Data') as pbar:
            while sampled_idx < self.expert_episode_used:
                # Ensure that the env has a solution
                while True:
                    _, info = self.env.reset(options=options)
                    if info["Expert_Policy"] != None:
                        sampled_idx += 1
                        break
                # Collect expert experience
                Xd = info["Expert_Policy"]["Branch_Obs"]
                Yd = info["Expert_Policy"]["TieLine_Action"]
                Xc = np.vstack((info["Expert_Policy"]["PF"],info["Expert_Policy"]["QF"]))
                Yc = info["Expert_Policy"]["Q_svc"]
                data = list(zip(Xd.T,Yd, Xc.T,Yc.T))
                for item in data:
                    self.buffer.append(item) # append data to buffer
                pbar.update(1)
                
        Xd, Yd, Xc, Yc = self.buffer.sample(batch_size=self.expert_sample_used,shuffle=True)
                
        # 2. Train the agent     
        with tqdm(total=self.training_epochs, desc='BC Training') as pbar:
            for e in range(self.training_epochs):
                for _ in range(self.train_iters):
                    sample_indices = np.random.randint(low=0,
                                            high=Xd.shape[0],
                                            size=self.batch_size)
                    Xd_batch = Xd[sample_indices]
                    Yd_batch = Yd[sample_indices]
                    Xc_batch = Xc[sample_indices]
                    Yc_batch = Yc[sample_indices]
                    # Discrete policy learning
                    self.agent.train_discrete(Xd_batch, Yd_batch)
                    # Continuous policy learning
                    indices = find_1Dtensor_value_indices(Yd_batch, self.discrete_action_dim)
                    for idx in range(self.discrete_action_dim):
                        self.agent.train_continuous(Xc_batch[indices[idx],:], Yc_batch[indices[idx],:], idx)
                    
                    pass
                
                self.logger.event_logger.info(
                    f"=============== Epoch {e+1:d} of {self.training_epochs:d} ================="
                )
                # run test
                avg_test_success_rate = self.test()
                pbar.set_postfix({"test_avg_rate":avg_test_success_rate})
                pbar.update(1)
                pass
        
    
    def test(self) -> float:
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
            for step in self.env.unwrapped.exploration_seq_idx:
                s_top_0 = np.reshape(s0["X_branch"], (-1, self.env.unwrapped.system_data.N_Branch))
                a_tieline = self.agent.predict_discrete(s_top_0)
                a_tieline = int(a_tieline.item())
                s_pqf_0 = np.concatenate((s0["PF"], s0["QF"]), axis=0)
                a_svc = self.agent.predict_continuous(s_pqf_0, a_tieline)
                a_svc = a_svc.to("cpu").detach().numpy()
                self.logger.event_logger.info("Given state S[{}] = {}".format(step, s_top_0[0]))
                self.logger.event_logger.info("Agent action at S[{}] is Ad[{}] = {}, Ac[{}] = {}".format(step, step+1, a_tieline, step+1, a_svc))
                a = {"Tieline":a_tieline, "Varcon":a_svc}
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
        
        return np.mean(saved_success_rate)
    

if __name__ == "__main__":
    
    env = gym.make(
        id=args.env_id,
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
        V_max=args.V_max
    )
    
    current_path = os.getcwd()
    task_name = 'BC_Hybrid'
    log_output_path = current_path + "/" + args.result_folder_name + "/" + task_name + \
                    ("_n_" + str(args.min_disturbance) + "to" + str(args.max_disturbance)
                        + "_" + get_time() + "/" )
    
    tensorboard_path = log_output_path + "tensorboard/"

    if args.forced_cpu:
        device = torch.device("cpu")
    else:
        device = check_cuda()
        
    manager = TrainManager(
        env=env,
        log_output_path = log_output_path,
        training_epochs=args.train_epochs,
        expert_episode_used=args.IL_used_episodes,
        expert_sample_used=args.IL_used_samples,
        batch_size=args.IL_batch_size,
        train_iters = args.IL_update_iters,
        lr=args.IL_lr,
        test_iterations=args.test_iterations, 
        device=device,
        seed=args.seed
    )
    
    manager.logger.event_logger.info(
            f"=============== Task Info =================")
    manager.logger.event_logger.info(
        f"ENV Settings == {args.env_id}, Device == {device}, Seed == {args.seed}")
    manager.logger.event_logger.info(
        f"Task == {task_name}, Expert Sample Used == {args.IL_used_samples}, Test Iterations == {args.test_iterations}")
    manager.logger.event_logger.info(
        f"Opt_Framework == {args.opt_framework}, Solver == {args.solver}, system_file == {args.data_file}")
    manager.logger.event_logger.info(
        f"min_disturbance == {args.min_disturbance}, max_disturbance == {args.max_disturbance}")

    manager.train()