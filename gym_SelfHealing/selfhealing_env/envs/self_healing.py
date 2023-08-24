import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from .System_Data import System_Data
from julia import Main as jl
import os
import numpy as np
import random
from typing import Optional, Tuple, Union 
import warnings
try:
    from .OPF_Core import OPF_Core
except ImportError:
    warning_msg = "Gurobipy env support is not available: Importing Gurobipy failed."
    warnings.warn(warning_msg)
else:
    pass
    

class SelfHealing_Env(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, 
                 data_file:str, 
                 opt_framework:str = "JuMP",
                 solver:Optional[str] = "cplex", 
                 solver_display:bool = False, 
                 vvo:bool = True, 
                 min_disturbance:int = 1, 
                 max_disturbance:int = 1
                 ) -> None:
        
        self.opt_framework = opt_framework
        
        if self.opt_framework == "JuMP":
            """参数: 
            solver: 求解器,可选CPLEX或Gurobi
            vvo:是否考虑svc
            min_disturbance: 随机N-k中的k的最小值
            max_disturbance: 随机N-k中的k的最大值
            """
            #---------设置求解器----------------------
            if solver == "cplex":
                self.solver = "CPLEX"
                
            elif solver == "gurobi":
                self.solver = "Gurobi"
                
            else:
                raise Exception("Solver not supported!")
            
        elif self.opt_framework == "Gurobipy":
            pass
        
        else:
            raise Exception("Optimization framework not supported!")
        
        #---------设置扰动----------------------
        self.min_disturbance = min_disturbance
        self.max_disturbance = max_disturbance
        
        # ---------读取系统数据----------------------
        file_name = os.path.join(os.path.dirname(__file__), "case_data",data_file)
        self.system_data = System_Data(file_name=file_name)
        varcon_lower_limit = self.system_data.Qsvc_lower_limit
        varcon_upper_limit = self.system_data.Qsvc_upper_limit
        
        self.vvo = vvo
        
        """-------------------建立环境模型----------------------
        # 输入参数：
        # args_expert -> expert model
        # args_step -> step model and reset model
        # solver display
        # 目标是建立三个opf模型 分别是Expert, Reset, Step
        """
        #TODO Add Gurobipy env support : Create & initialize env
        if self.opt_framework == "JuMP":
            jl.eval("using " + self.solver)
            jl.include(os.path.join(os.path.dirname(__file__),"OPF_Core.jl"))
            jl.init_opf_core(args_expert=self.system_data.args_expert,
                            args_step=self.system_data.args_step,
                            solver=self.solver,
                            display=solver_display)
            
        elif self.opt_framework == "Gurobipy":
            self.core = OPF_Core(args_expert=self.system_data.args_expert,
                            args_step=self.system_data.args_step,
                            display=solver_display)
        
        else:
            raise Exception("Optimization framework not supported!")
        """----------------------------------------"""
        
        # ---动作状态空间相关参数设置---
        """ 
        Take 33BW system with 5 tie-lies as an example. The agent has five step chances to take actions.
        ----------------------------------------------------------------------------------------------
            S0   -A1->   S1    -A2->   S2    -A3->    S3    -A4->    S4    -A5->   S5 (terminal)
         reset  step1        step2          step3          step4          step5   
         ---------------------------------------------------------------------------------------------
                idx:0->1     idx=1->2       idx=2->3       idx=3->4       idx=4->5
              (_step->step_)                                           thus, if idx>=4, done=True after transition
         ----------------------------------------------------------------------------------------------
        """
        self.exploration_total = self.system_data.N_TL
        self.exploration_seq_idx = [i for i in range(self.exploration_total)]
        
        if self.vvo:
            self.action_space = spaces.Dict({
                "Tieline": spaces.Discrete(self.system_data.N_TL + 1), # 0~N_TL数字,0代表什么都不做
                "Varcon": spaces.Box(low=np.array(varcon_lower_limit), high=np.array(varcon_upper_limit))
                })
            
            self.observation_space = spaces.Dict({
                "X_branch": spaces.MultiBinary(self.system_data.N_Branch),
                "X_load": spaces.MultiBinary(self.system_data.N_Bus),
                "PF": spaces.Box(low=self.system_data.S_lower_limit, high=self.system_data.S_upper_limit),
                "QF": spaces.Box(low=self.system_data.S_lower_limit, high=self.system_data.S_upper_limit)
                })
        else:
            self.action_space = spaces.Discrete(self.system_data.N_TL + 1)
            self.observation_space = spaces.MultiBinary(self.system_data.N_Branch) 
        
        pass
    
    def reset(self, 
              options:dict = {"Specific_Disturbance": None, "Expert_Policy_Required": False},
              seed:Optional[int] = None) -> Tuple[dict,dict]:
        """
        options = {
            "Specific_Disturbance": list or None
            "Expert_Policy_Required": bool
            }
        
        If you want to use a SPECIFIC disturbance, use the following option:
        "Specific_Disturbance": list of # of line, e.g: [6,11,29,32]
        
        Otherwise, you should use the following option to generate a RANDOM disturbance:
        "Specific_Disturbance": None
        
        !! WARNING: Given the list = [] does not mean random disturbance, it means the environment will be reset to the original state !!
        """
        disturbance = options["Specific_Disturbance"]
        expert_policy_required = options["Expert_Policy_Required"]
        # initialize a list to store the load status during an episode
        self.load_rate_episode = []
        
        # index to determine the instants
        self.exploration_index = 0
        
        # ================== generate disturbance ====================
        # 随机产生一个扰动破坏普通线路，在接下来的时刻中
        # generate random disturbance if no specific one is given
        
        if seed is not None:
            random.seed(seed)
        
        if disturbance == None:
            random_mode = True
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
            random_mode = False
            num_disturbance = len(disturbance)
            self.disturbance = disturbance
            
        # =============== initialize the line and var control for optimization ===============
        a = np.ones((self.system_data.N_NL, self.system_data.NT)) # 设置普通线路的灾害状态
        for dmg in self.disturbance:
            a[dmg-1, :] = 0
            
        self.a = a
        
        X_tieline0 = np.zeros(self.system_data.N_TL) # tieline默认一开始都是打开的
        Q_svc0 = np.zeros(self.system_data.N_DG-1) # svc默认输出均为0
        
        """-------------------初始化N-k场景下的环境模型----------------------
        # 输入参数：
        # a -> expert/step/reset model
        # X_tieline0 -> expert/reset model
        # solver display
        # 目标是为每个模型设置N-k受损状态; 为Expert和Reset模型设置初始TieLine; 为Reset模型设置初始Q_svc
        """
        #TODO Add Gurobipy env support : Reset env models according to the disturbance
        if self.opt_framework == "JuMP":
            jl.set_dmg(self.a) # 对模型设置普通线路的灾害状态
            jl.set_ResetModel(X_tieline_input=X_tieline0, Q_svc_input=Q_svc0)
            
        elif self.opt_framework == "Gurobipy":
            self.core.set_dmg(self.a)
            self.core.set_ResetModel(X_tieline_input=X_tieline0, Q_svc_input=Q_svc0)
            
        else:
            raise Exception("Optimization framework not supported!")
        
        """----------------------------------------"""

        """---------------------求解Reset模型,返回observation----------------------"""
        #TODO Add Gurobipy env support : Solve reset model
        if self.opt_framework == "JuMP":
            _b, _x_tieline, _x_load, _PF, _QF, load_value_current = jl.solve_ResetModel()
            
        elif self.opt_framework == "Gurobipy":
            _b, _x_tieline, _x_load, _PF, _QF, load_value_current = self.core.solve_ResetModel()
            
        else:
            raise Exception("Optimization framework not supported!")
        
        """------------------------------------------------------------------------------"""
        
        self._x_load = np.round(_x_load).astype(np.int8) # 还需要用self记录上一步的负荷拾取情况、总负荷恢复量、tieline状态
        if self.vvo:
            self.obs = {"X_branch": np.concatenate((self.a[:,0].flatten(),_x_tieline)).astype(np.int8), # 注意实际上普通线路是不会断开的
                "X_load": self._x_load,
                "PF": _PF.astype(np.float32),
                "QF": _QF.astype(np.float32)
                }
        else:
            self.obs = np.concatenate((self.a[:,0].flatten(),_x_tieline)).astype(np.int8) # 注意实际上普通线路是不会断开的
        
        self._x_tieline = np.round(_x_tieline).astype(np.int8)
        load_rate_current = load_value_current / self.system_data.Pd_all
        self.load_rate_episode.append(load_rate_current) # 添加到该episode的得分记录表中
        self._x_nl = np.round(_b[0:self.system_data.N_NL-1]).astype(np.int8) # 保存普通线路的状态，用于判断step的拓扑是否可行
        
        """---------------------设置Expert模型----------------------"""
        #TODO Add Gurobipy env support : Set & solve expert model
        if self.opt_framework == "JuMP":
            jl.set_ExpertModel(X_tieline0_input=X_tieline0,X_rec0_input=self._x_load,X_line0_input=self._x_nl,vvo=self.vvo) # 设置Expert模型的初始状态
        elif self.opt_framework == "Gurobipy":
            self.core.set_ExpertModel(X_tieline0_input=X_tieline0,X_rec0_input=self._x_load,X_line0_input=self._x_nl,vvo=self.vvo)
        else:
            raise Exception("Optimization framework not supported!")
        """---------------------------------------------------------"""
        # 在reset的同时就可以求解Expert模型
        expert_policy = None
        if expert_policy_required:
            """---------------------求解Expert模型,返回expert policy----------------------"""
            if self.opt_framework == "JuMP":
                solved_flag, expert_b, expert_x_tieline, expert_x_load, \
                    expert_Pg, expert_Qg, load_value_expert = jl.solve_ExpertModel()
            elif self.opt_framework == "Gurobipy":
                solved_flag, expert_b, expert_x_tieline, expert_x_load, \
                    expert_Pg, expert_Qg, load_value_expert = self.core.solve_ExpertModel()
            else:
                raise Exception("Optimization framework not supported!")
            """------------------------------------------------------------------------------"""
            if solved_flag:
                expert_b = np.round(expert_b).astype(np.int8)
                expert_x_tieline = np.round(expert_x_tieline).astype(np.int8)
                expert_x_load = np.round(expert_x_load).astype(np.int8)
                expert_P_sub = expert_Pg[0,:]
                expert_Q_sub = expert_Qg[0,:]
                expert_Q_svc = expert_Qg[1:,:]
                load_value_expert = load_value_expert.flatten()
                expert_load_rate = load_value_expert / np.sum(self.system_data.Pd, axis=0)
                
                expert_policy = {"Branch_Energized": expert_b,
                                "Load_Energized": expert_x_load,
                                "TieLine_Action": expert_x_tieline,
                                "P_sub": expert_P_sub,
                                "Q_sub": expert_Q_sub,
                                "Q_svc": expert_Q_svc,
                                "Load_Rate": expert_load_rate}
                                
        # info返回当前episode case的属性
        info = {
            "VVO_Enabled": self.vvo,
            "Specific_Disturbance": not random_mode,
            "k of N-k": num_disturbance,
            "Disturbance_Set": self.disturbance,
            "Episode_Length": self.exploration_total,
            "Recovered_Load_Rate_S0": load_rate_current,
            "Expert_Policy_Required": expert_policy_required,
            "Expert_Policy": expert_policy
        }
        
        return self.obs, info
    
    
    def step(self, action:Union[dict,int]) -> Tuple[dict, float, bool, dict]:
        # step需要返回 observation, reward, terminated, truncated, info
        #                obs       reward     done       False      
        assert self.action_space.contains(action), "Invalid action. \n \
        A example can be action = {'Tieline': 4, 'Varcon': np.array([0.01, -0.01, 0.02, 0, 0, 0], dtype=np.float32)}"
        x_tieline_input = self._x_tieline.copy() # 用上一步的copy避免麻烦
        
        if self.vvo:
            action_tieline = action["Tieline"]
        else:
            action_tieline = action
            
        if action_tieline>=1:
            x_tieline_input[action_tieline-1] = 1 # 由于每次只动一个tieline，且不会关闭。如果执行了上次的命令则会保持原状
        
        if self.vvo:
            q_svc_input = action["Varcon"]
        else:
            q_svc_input = None
        
        # ===================== determine termination condition and rewards =====================
        event_log = None
        action_accepted = False
        # first check if this is the last step in this episode
        if self.exploration_index <= self.exploration_total-1:
            if self.exploration_index == self.exploration_total-1:
                done = True
            else:
                done = False
            # =====================  solve for load status =====================
            """-------------------将动作输入到环境模型并求解----------------------"""
            #TODO Add Gurobipy env support : Setup & Solve step model
            if self.opt_framework == "JuMP":
                jl.set_StepModel(X_rec0_input=self._x_load, X_tieline_input=x_tieline_input, 
                                Q_svc_input=q_svc_input, vvo=self.vvo)
                results = jl.solve_StepModel()
            elif self.opt_framework == "Gurobipy":
                self.core.set_StepModel(X_rec0_input=self._x_load, X_tieline_input=x_tieline_input, 
                                Q_svc_input=q_svc_input, vvo=self.vvo)
                results = self.core.solve_StepModel()
            else:
                raise Exception("Optimization framework not supported!")
            """----------------------------------------------------------------"""
            # 若无解，说明tieline合不上, tieline状态不变，reward为-1000

            solved, _b, _x_tieline, _x_load, _PF, _QF, load_value_new, e_Qsvc = results
            if not solved:
                event_log = "Infeasible Tieline Action"
                reward = -1000
                self.load_rate_episode.append(self.load_rate_episode[-1]) # 负荷不回复，保持上一步状态
            else:
                # 还需要记录上一步的负荷拾取情况、总负荷恢复量、tieline状态
                _x_nl = np.round(_b[0:self.system_data.N_NL-1]).astype(np.int8)
                flag_x_nl = np.any(_x_nl<self._x_nl) # 普通线路状态不同，说明拓扑不可行，违反辐射状，
                flag_e_Qsvc = e_Qsvc >= 1e-6 # svc指令误差大于1e-4，说明svc指令不可行
                if flag_x_nl: # 拓扑不可行
                    event_log = "Infeasible Topology"
                    reward = -1000 
                    self.load_rate_episode.append(self.load_rate_episode[-1]) # 负荷不回复，保持上一步状态
                elif flag_e_Qsvc: # svc指令不可行
                    event_log = "Infeasible SVC Scheduling"
                    reward = -1000
                    self.load_rate_episode.append(self.load_rate_episode[-1])
                else: # 有效action
                    # 因为负荷被拾取后不会再失去，所以至少是上一步的负荷恢复量
                    action_accepted = True
                    load_rate_new = load_value_new / self.system_data.Pd_all
                    if load_rate_new-self.load_rate_episode[-1]>=1e-4:
                        event_log = "Increased Load Recovery"
                        reward = 150
                    else:
                        event_log = "Maintained Load Recovery"
                        reward = -10
                                    
                    self._x_nl = _x_nl # 保存普通线路的状态，用于判断step的拓扑是否可行
                    self._x_tieline = np.round(_x_tieline).astype(np.int8)
                    self._x_load = np.round(_x_load).astype(np.int8)
                    self.load_rate_episode.append(load_rate_new) # 添加到该episode的得分记录表中
                    
                    if self.vvo:
                        self.obs = {"X_branch": np.concatenate((self.a[:,0].flatten(),self._x_tieline)).astype(np.int8),
                                    "X_load": self._x_load,
                                    "PF": _PF.astype(np.float32),
                                    "QF": _QF.astype(np.float32)
                                    }
                    else:
                        self.obs = np.concatenate((self.a[:,0].flatten(),self._x_tieline)).astype(np.int8)
        else:
            done = True
            event_log = "Episode Closed"
            reward = 0
                    
        # update index
        self.exploration_index += 1
        
        info = {"Attempted_Tieline_Action": action_tieline,
                "Action_Accepted": action_accepted,
                "Interaction_Step": self.exploration_index,
                "Event_Log": event_log,
                "Recovered_Load_Rate": self.load_rate_episode[-1]
        }
        
        return self.obs, reward, done, False, info
    
    def render(self):
        #TODO Add visualization using NetworkX
        pass
    
    
    def close(self):
        #NOTE I am not sure if it is necessary to kill the Julia process.
        pass
