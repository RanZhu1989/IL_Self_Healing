import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from sys_data import System_Data

from julia import Main as jl

import numpy as np

import random

class pds_env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, solver="CPLEX", min_disturbance=1, max_disturbance=1):
        #---------设置求解器----------------------
        if solver == "CPLEX":
            self.solver = "CPLEX"
            
        elif solver == "Gurobi":
            self.solver = "Gurobi"
            
        else:
            raise Exception("Solver not supported!")
        
        #---------设置扰动----------------------
        self.min_disturbance = min_disturbance
        self.max_disturbance = max_disturbance
        
        # ---------读取系统数据----------------------
        self.system_data = System_Data(file_name='Case_33BW_Data.xlsx')
        varcon_lower_limit = self.system_data.Qsvc_lower_limit
        varcon_upper_limit = self.system_data.Qsvc_upper_limit
        
        # ---------初始化优化问题----------------------
        jl.eval("using " + self.solver)
        jl.include("system.jl")
        jl.init_opf_core(args_expert=self.system_data.args_expert,
                         args_step=self.system_data.args_step,
                         solver=self.solver)
        
        # ---动作状态空间相关参数设置---
        self.exploration_total = self.system_data.N_TL
        self.exploration_seq_idx = [i for i in range(self.exploration_total + 1)]
        self.action_space = spaces.Dict({
            "Tieline": spaces.Discrete(self.system_data.N_TL + 1), # 0~N_TL数字,0代表什么都不做
            "Varcon": spaces.Box(low=np.array(varcon_lower_limit), high=np.array(varcon_upper_limit))
            })
        
        # self.observation_space = spaces.MultiBinary(self.system_data.N_Branch) # 如果只用tieline
        self.observation_space = spaces.Dict({
            "X_branch": spaces.MultiBinary(self.system_data.N_Branch),
            "X_load": spaces.MultiBinary(self.system_data.N_Bus),
            "PF": spaces.Box(low=self.system_data.S_lower_limit, high=self.system_data.S_upper_limit),
            "QF": spaces.Box(low=self.system_data.S_lower_limit, high=self.system_data.S_upper_limit)
            })
        
        pass
    
    def reset(self, disturbance:list=None):
        
        # initialize a list to store the load status during an episode
        self.load_value_episode = []
        
        # index to determine the instants
        self.exploration_index = 0
        
        # ================== generate disturbance ====================
        # 随机产生一个扰动破坏普通线路，在接下来的时刻中
        # generate random disturbance if no specific one is given
        
        if disturbance == None:
            temp_disturbance_set = self.system_data.disturbance_set.copy()
            # generate disturbance upper bound for this episoid   N-k的k
            num_disturbance = random.randint(self.min_disturbance, self.max_disturbance)
            # record generated disturbance
            self.disturbance = []
            # 不放回随机抽取k个线路中断
            for _ in range(num_disturbance):
                # generate one line outage at a time
                random_disturbance = random.choice(temp_disturbance_set)
                # record
                self.disturbance.append(random_disturbance)
                # remove from the set
                temp_disturbance_set.remove(random_disturbance)
                
        else:
            self.disturbance = disturbance
            
        # =============== initialize the line and var control for optimization ===============
        a = np.ones((self.system_data.N_NL, self.system_data.NT)) # 设置普通线路的灾害状态
        for dmg in self.disturbance:
            a[dmg-1, :] = 0
        
        X_tieline0 = np.zeros(self.system_data.N_TL) # tieline默认一开始都是打开的
        Q_svc0 = np.zeros(self.system_data.N_DG-1) # svc默认输出均为0
        
        jl.set_dmg(a) # 对模型设置普通线路的灾害状态
        jl.set_ResetModel(X_tieline_input=X_tieline0, Q_svc_input=Q_svc0)
        # 求解Reset模型需要返回obs
        # 还需要用self记录上一步的负荷拾取情况、总负荷恢复量、tieline状态
        _b, _x_tieline, _x_load, _PF, _QF, load_value_current = jl.solve_ResetModel()
        self._x_load = np.round(_x_load).astype(int)
        self.obs = {"X_branch": np.round(_b).astype(int),
               "X_load": self._x_load,
               "PF": _PF.astype(np.float32),
               "QF": _QF.astype(np.float32)
               }
        
        self._x_tieline = np.round(_x_tieline).astype(int)
        self.load_value_episode.append(load_value_current) # 添加到该episode的得分记录表中
        self._x_nl = np.round(_b[0:self.system_data.N_NL-1]).astype(int) # 保存普通线路的状态，用于判断step的拓扑是否可行
        
        return self.obs
    
    def step(self, action):
        # step需要返回 observation, reward, terminated, truncated, info
        #                obs       reward     done       False      {}
        assert self.action_space.contains(action), "Invalid action. \n \
        A example can be action = {'Tieline': 4, 'Varcon': np.array([0.01, -0.01, 0.02, 0, 0, 0], dtype=np.float32)}"
        x_tieline_input = self._x_tieline.copy() # 用上一步的copy避免麻烦
        if action["Tieline"]>=1:
            x_tieline_input[action["Tieline"]-1] = 1 # 由于每次只动一个tieline，且不会关闭。如果执行了上次的命令则会保持原状
            
        q_svc_input = action["Varcon"]
        
        # ===================== determine termination condition and rewards =====================
        # first check if this is the last step in this episode
        if self.exploration_index == self.exploration_total:
            done = True  # we are done with this episode
            reward = 0
        else:
            done = False
        
        info = self._get_info() # 总要返回
        # =====================  solve for load status =====================
        jl.set_StepModel(X_rec0_input=self._x_load, X_tieline_input=x_tieline_input, Q_svc_input=q_svc_input)
        # 求解step模型需要返回obs,是否有解
        results = jl.solve_StepModel()
        # 若无解，说明tieline合不上, tieline状态不变，reward为-1000
        if isinstance(results,bool):
            reward = -1000
            self.load_value_episode.append(self.load_value_episode[-1]) # 负荷不回复，保持上一步状态
        else:
            reward = 0
            # 还需要记录上一步的负荷拾取情况、总负荷恢复量、tieline状态
            _, _b, _x_tieline, _x_load, _PF, _QF, load_value_new, e_Qvsc = results
            _x_nl = np.round(_b[0:self.system_data.N_NL-1]).astype(int)
            flag_x_nl = np.any(_x_nl<self._x_nl) # 普通线路状态不同，说明拓扑不可行，违反辐射状，
            flag_e_Qvsc = e_Qvsc >= 1e-4 # svc指令误差大于1e-4，说明svc指令不可行
            if flag_x_nl: # 拓扑不可行
                reward -= 1000 
                self.load_value_episode.append(self.load_value_episode[-1]) # 负荷不回复，保持上一步状态
            elif flag_e_Qvsc: # svc指令不可行
                reward -= 500
                self.load_value_episode.append(self.load_value_episode[-1])
            else: # 有效action
                if load_value_new > self.load_value_episode[-1]:
                    reward = 1000
                elif load_value_new == self.load_value_episode[-1]:
                    reward = -10
                else: 
                    reward = -100
                                
                self._x_nl = _x_nl # 保存普通线路的状态，用于判断step的拓扑是否可行
                self._x_tieline = np.round(_x_tieline).astype(int)
                self._x_load = np.round(_x_load).astype(int)
                self.load_value_episode.append(load_value_new) # 添加到该episode的得分记录表中
                self.obs = {"X_branch": np.round(_b).astype(int),
                            "X_load": np.round(self._x_load).astype(int),
                            "PF": _PF.astype(np.float32),
                            "QF": _QF.astype(np.float32)
                            }
        # update index
        self.exploration_index += 1
        
        return self.obs, reward, done, False, info
    
    def _get_info(self):
        return {}
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
env = pds_env(solver="Gurobi",min_disturbance=1, max_disturbance=1)
obs = env.reset(disturbance=[6,11,29,32])


action = {"Tieline":3,
          "Varcon": np.array([-0.002, 0, 0.002, 0.002, -0.002,0], dtype=np.float32)
        }

obs, reward, _, _, _ = env.step(action)
print(reward)
print(obs["X_branch"])


action = {"Tieline":1,
          "Varcon": np.array([-0.002, 0, 0.002, 0.002, -0.002,0], dtype=np.float32)
        }

obs, reward, _, _, _ = env.step(action)
print(reward)
print(obs["X_branch"])

action = {"Tieline":2,
          "Varcon": np.array([-0.002, 0, 0.002, 0.002, -0.002,0], dtype=np.float32)
        }

obs, reward, _, _, _ = env.step(action)
print(reward)
print(obs["X_branch"])

action = {"Tieline":4,
          "Varcon": np.array([-0.002, 0, 0.002, 0.002, -0.002,0], dtype=np.float32)
        }

obs, reward, _, _, _ = env.step(action)
print(reward)
print(obs["X_branch"])
