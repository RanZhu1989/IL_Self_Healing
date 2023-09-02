import time
import os
import gymnasium as gym
import selfhealing_env
from typing import Optional 
import numpy as np
from stable_baselines3 import DQN
import torch
import random

from utils import logger

class TrainManager():
    
    def __init__(
        self, 
        env:gym.Env,
        log_output_path:Optional[str]=None,
        episode_num:int = 200,
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
        self.episode_num = episode_num
        self.time_steps = self.env.system_data.NT
        random.seed(self.seed)
        self.log_output_path = log_output_path
        
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
            
    def test_agent(self):
        
        print("Begin solving benchmark \n")
        stime = time.time()
            
        test_options = {"Specific_Disturbance":None, "Expert_Policy_Required":True, "External_Seed":True}
        s0, expert_info = None, None
        
        while True:
            s0, expert_info = self.env.reset(options=test_options)
            if expert_info["Expert_Policy"] != None:
                break

        test_disturbance_set = expert_info["Disturbance_Set"]
        # ================ calculate Benchmark value to normalize the restoration as ratio ==================
        self.logger.event_logger.info("-------------------Run_test begin--------------------")
        self.logger.event_logger.info("The testing disturbance is {}".format(sorted(test_disturbance_set)))
        
        # if expert_info["Expert_Policy"] == None:
        #     self.logger.event_logger.info("Warning: Numerical issues in solving the Rest Model made Xrec0 infeasible, rendering the Expert Model unsolvable.")
        #     self.logger.event_logger.info("This occurrence is rare, but if it happens frequently, please adjust the penalty coefficients in the objective function.")
        #     self.logger.event_logger.info("Test in this episode is cancelled.")
        #     return None
            
        # 计算goal
        load_rate_s0 = expert_info["Recovered_Load_Rate_S0"] # 初始负荷回复率
        expert_load_rate = expert_info["Expert_Policy"]["Load_Rate"]
        expert_load_rate_max = expert_load_rate[-1] # 因为负荷不可能降低
        goal = expert_load_rate_max - load_rate_s0 # 专家策略回复提升量(归一化)
        
        self.logger.event_logger.info("Expert policy is {}".format(expert_info["Expert_Policy"]["TieLine_Action"].T.tolist())) # 记录专家策略
        
        print("Complete solving benchmark using {}".format(time.time() - stime))
        print("Begin testing \n")
        stime = time.time()
        
        # ============== run agent using trained policy approximator ==================
        # 用智能体在这个env中输出动作，记录负荷恢复量
        agent_info = None
        agent_load_rate = []
        for step in self.env.exploration_seq_idx:
            s0 = np.reshape(s0, (-1, self.env.system_data.N_Branch)) # 将37维列向量转换为1*37的矩阵，-1是自动计算行数
            a, _ = self.agent.predict(s0)  # this action is one-hot encoded
            self.logger.event_logger.info("Given state S[{}] = {}".format(step, s0[0]))
            self.logger.event_logger.info("Agent action at S[{}] is A[{}] = {}".format(step, step+1, a))
                
            s, reward, _, _, agent_info = self.env.step(a.item()) # 此时动作其实是一个只有单个元素的np.ndarray
            load_rate = agent_info["Recovered_Load_Rate"]
            agent_load_rate.append(load_rate) # 保存当前load rate
            self.logger.event_logger.info("Event at S[{}] is '{}'".format(step+1, agent_info["Event_Log"]))
            self.logger.event_logger.info("Reward R[{}] is {}. Load rate at S[{}] is {}.".format(step+1, reward,step+1, load_rate))
            # update state
            s0 = s
        
        print("Complete testing using {} second \n".format(time.time() - stime))
        print("Begin recording \n")
        stime = time.time()
            
        # ================ evaluate success =====================
        self.logger.event_logger.info("Load rate at S0 is {}".format(load_rate_s0)) # 记录初始负荷恢复量
        performance = agent_load_rate[-1] - load_rate_s0 # 智能体策略回复提升量(归一化)
        if abs(goal) > 1e-4:
            self.logger.event_logger.info("performance: {}; goal: {}".format(performance, goal)) 
        else:
            self.logger.event_logger.info("Nothing we can do to improve the load profile. (Reason: performance={}, goal={})".format(performance,goal)) # 差异极小，系统没救了，没有自愈的必要
        
        self.logger.collect_to_csv(disturb=test_disturbance_set, agent_recovery_rate=np.array(agent_load_rate), expert_recovery_rate=expert_load_rate)
        pass
        
    
    def train(self):
        
        tic = time.perf_counter()  # start clock
        for idx_episode in range(self.episode_num):
            if idx_episode % 1 == 0: # 控制episode打印频率
                toc = time.perf_counter()
                print("===================================================")
                print(f"Training time: {toc - tic:0.4f} seconds; Mission {idx_episode:d} of {self.episode_num:d}")
                print("===================================================")
            self.logger.event_logger.info(f"=============== Mission {idx_episode:d} of {self.episode_num:d} =================")
            # executes the expert policy and perform Deep Q learning
            self.agent.learn(total_timesteps=self.time_steps,reset_num_timesteps=False) # 这里5就是ENV中的NT
            # test: execute learned policy on the environment
            self.test_agent()
        
        pass
    


if __name__ == "__main__":
    
    current_path = os.getcwd()
    log_output_path = current_path + "/results/DQN_TieLine/n_1/"
    tensorboard_path = log_output_path + "tensorboard/"
    
    env = gym.make(id="SelfHealing-v0",
                   opt_framework="JuMP",
                   solver="cplex", 
                   data_file="Case_33BW_Data.xlsx",
                   solver_display = False, 
                   vvo= False, 
                   min_disturbance = 1, 
                   max_disturbance = 1)
     
    manager = TrainManager(env=env,
                           episode_num=500,
                           learning_rate=1e-6,
                           learning_starts=100,
                           batch_size=64,
                           tau=1,
                           gamma=0.95,
                           verbose=0,
                           log_output_path=log_output_path,
                           device=torch.device("cpu"),
                           tensorboard_log=None,
                           seed=0
                        )
    
    manager.train()
    



