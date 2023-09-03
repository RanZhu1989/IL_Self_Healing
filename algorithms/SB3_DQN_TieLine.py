import os
import time
import random
from tqdm import tqdm
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

import selfhealing_env
from utils import logger

class TrainManager():
    def __init__(
        self, 
        env:gym.Env,
        log_output_path:Optional[str]=None,
        episode_num:int = 200,
        test_iterations:int = 10,
        learning_rate: float = 0.001,
        buffer_size: int = 100000,
        learning_starts:int = 50,
        batch_size:int = 30,
        tau:float = 0.1,
        gamma:float = 0.95,
        exploration_final_eps:float = 0.05,
        verbose:int = 0,
        device:torch.device = torch.device("cpu"),
        seed:Optional[int] = None,
        tensorboard_log:Optional[str] = None
    ) -> None:
        
        self.env = env
        self.device = device
        self.seed = seed
        self.test_rng = random.Random(self.seed)
        self.episode_num = episode_num
        self.time_steps = self.env.system_data.NT
        self.log_output_path = log_output_path
        self.test_iterations = test_iterations
        
        self.agent = DQN(
            policy = "MlpPolicy",
            env = self.env,
            train_freq = (1, "episode"),
            learning_rate = learning_rate,
            buffer_size = buffer_size,
            learning_starts = learning_starts,
            batch_size = batch_size,
            tau = tau,
            gamma = gamma,
            exploration_final_eps = exploration_final_eps,
            verbose = verbose,
            device = self.device,
            seed = self.seed,
            tensorboard_log = tensorboard_log,
        )
        
        self.logger = logger(log_output_path=self.log_output_path) 
        pass
            
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
                a, _ = self.agent.predict(s0)
                self.logger.event_logger.info("Given state S[{}] = {}".format(step, s0[0]))
                self.logger.event_logger.info("Agent action at S[{}] is A[{}] = {}".format(step, step+1, a))
                s, reward, _, _, agent_info = self.env.step(a.item())
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
        
    def train(self) -> None:
        tic = time.perf_counter()  # start clock
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for e in range(self.episode_num):
                callback = RewardCallback()
                self.agent.learn(total_timesteps=self.time_steps,reset_num_timesteps=False,callback=callback)
                episode_reward = sum(callback.rewards)
                self.test()
                toc = time.perf_counter()
                pbar.set_postfix({"Training time": f"{toc - tic:0.2f} seconds", "Return": (episode_reward)})
                pbar.update(1)
                

# Callback function to get reward
class RewardCallback(BaseCallback):
    def __init__(self, verbose:int=0) -> None:
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][-1]
        self.rewards.append(reward)
        return True
    
    
if __name__ == "__main__":
    current_path = os.getcwd()
    log_output_path = current_path + "/results/SB3_DQN_TieLine/n_1/"
    tensorboard_path = log_output_path + "tensorboard/"
    
    env = gym.make(
        "SelfHealing-v0",
        opt_framework="JuMP",
        solver="cplex", 
        data_file="Case_33BW_Data.xlsx",
        solver_display = False, 
        vvo= False, 
        min_disturbance = 1, 
        max_disturbance = 1,
        Sb = 100,
        V0 = 1.05,
        V_min = 0.95,
        V_max = 1.05
    )
     
    manager = TrainManager(
        env=env,
        episode_num=200,
        learning_rate=1e-3,
        learning_starts=50,
        test_iterations=5,
        batch_size=32,
        tau=1,
        gamma=0.90,
        verbose=0,
        log_output_path=log_output_path,
        device=torch.device("cpu"),
        tensorboard_log=None,
        seed=0
    )
    
    manager.train()
    