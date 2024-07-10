import os
import collections
import random
from tqdm import tqdm
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np

import selfhealing_env

# Torch should be imported after juliacall
import torch
import torch.nn.functional as F

## --In case of import error when you have to use python-jl to run the code, please use the following import statement--
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --------------------------------------------------------------------------------------------------------------
from utils import logger, check_cuda, get_time
from configs import args

class DAgger_Agent():
    """
    Dataset Aggregation (DAgger) Agent
    
    The simplest way to improve the performance of behavioral cloning is 
    to 'ASK' the expert when encounter to new states.
    
    """
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
        # categorical crossentropy： MLE loss
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
    
    def fetch_all(self,shuffle:bool = True) -> Tuple[torch.tensor,torch.tensor]:
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
        self.test_rng = random.Random(self.seed+1) # Seed for testing
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
        
        self.agent = DAgger_Agent(
            input_dim = self.obs_dim, output_dim = self.action_dim,
            policy_network = policy_network, optimizer = optimizer,
            device = self.device
        )
        
        self.buffer = ReplayBuffer(capacity=buffer_capacity,device=self.device)
        
        self.logger = logger(log_output_path=self.log_output_path)
        pass
        
    def train(self) -> None:
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for idx_episode in range(self.episode_num):
                self.logger.event_logger.info(
                    f"=============== Episode {idx_episode+1:d} of {self.episode_num:d} ================="
                )
                self.train_episode()
                # run test
                avg_test_success_rate = self.test()
                pbar.set_postfix({"test_avg_rate":avg_test_success_rate})
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
            sample_indices = np.random.randint(low=0,
                                            high=len(self.buffer),
                                            size=self.batch_size) # sample data
            obs_all, action_all = self.buffer.fetch_all(shuffle=True)
            self.agent.learn(obs_all[sample_indices],action_all[sample_indices])
       
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
        vvo=False,
        Sb=args.Sb,
        V0=args.V0,
        V_min=args.V_min,
        V_max=args.V_max
    )
    
    current_path = os.getcwd()
    task_name = 'DAgger_TieLine'
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
        episode_num=args.train_epochs,
        train_iterations=args.IL_update_iters,
        batch_size=args.IL_batch_size,
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
        f"Task == {task_name}, Training Epochs == {args.train_epochs}, Test Iterations == {args.test_iterations}")
    manager.logger.event_logger.info(
        f"Opt_Framework == {args.opt_framework}, Solver == {args.solver}, system_file == {args.data_file}")
    manager.logger.event_logger.info(
        f"min_disturbance == {args.min_disturbance}, max_disturbance == {args.max_disturbance}")

    manager.train()