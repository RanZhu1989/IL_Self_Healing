"""Gym entry point for SelfHealing_Env environment."""
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
    """A Gymnasium environment for Self-Healing Power System Restoration"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self, 
        data_file:str, 
        opt_framework:str = "JuMP",
        solver:Optional[str] = "cplex", 
        solver_display:bool = False, 
        vvo:bool = True, 
        min_disturbance:int = 1, 
        max_disturbance:int = 1
    ) -> None:
        """
        Initialize the environment.
        
        Args:
            data_file: Case data EXCEL file name in the data folder "./case_data". e.g. "Case_33BW_Data.xlsx"
                        
            opt_framework: Optimization framework used for the environment. 
                            Currently only JuMP and Gurobipy are supported.
                            You should use JuMP if Gurobipy is not installed.
            
            solver: Solver used for the environment. Currently only CPLEX and Gurobi are supported. 
                    When using Gurobipy framework, this argument is ignored.
            
            solver_display: Whether to display the solver log.
            
            vvo: Whether to enable VVO. If True, the action space will be a dictionary with two keys: "Tieline" and "Varcon".
                Otherwise, the action space will be a single integer.
            
            min_disturbance: Minimum number of line outages in the random N-k disturbance set.
            
            max_disturbance: Maximum number of line outages in the random N-k disturbance set.
        """
        
        # Set optimization framework and solver
        self.opt_framework = opt_framework
        if self.opt_framework == "JuMP":
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
        
        # Set disturbance range
        self.min_disturbance = min_disturbance
        self.max_disturbance = max_disturbance
        
        # Read system data
        file_name = os.path.join(os.path.dirname(__file__), "case_data",data_file)
        self.system_data = System_Data(file_name=file_name)
        varcon_lower_limit = self.system_data.Qsvc_lower_limit
        varcon_upper_limit = self.system_data.Qsvc_upper_limit
        
        # Set VVO mode
        self.vvo = vvo
        
        # Initialize OPF core
        """-------------------Initialize OPF Core----------------------
            The core models will be immediately initialized once the environment is created.
            The reset\step method will only modify some parameters of these pre-loaded core models.
        
           ==============        ====================        ===================
           Opt_Framework                Method                      Items
           ==============        ====================        ===================
           JuMP                 | init_opf_core()    |      |model_args, solver,|
                                | (through JuliaPy)  |      |  display          |
         
           Gurobipy              OPF_Core.__init__()         model_args, display
           ==============        ====================        ===================
           
          Parameters of initialize functions:
            system_data.args_expert -> expert model
            system_data.args_step -> step model and reset model
        """
        if self.opt_framework == "JuMP":
            jl.eval("using " + self.solver) # Load solver
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
            #NOTE You can add other optimization frameworks here.
            raise Exception("Optimization framework not supported!")
        
        """------------------------End of initialization------------------------"""
        
        # Set observation and action space
        self.exploration_total = self.system_data.N_TL
        self.exploration_seq_idx = [i for i in range(self.exploration_total)]
        
        """ Observation & Action Space Settings
        ===========    ======================     ===================
        VVO Enabled       Observation Space           Action Space
        ===========    ======================     ===================
            True        Dict{Box, MultiBinary}     Dict{Discrete, Box}
        
            False           MultiBinary                Discrete
        ===========    ======================     ===================
        """
        if self.vvo:
            self.action_space = spaces.Dict({
                "Tieline": spaces.Discrete(self.system_data.N_TL + 1), # 1 to 5: open tieline 1 to 5; 0: do nothing
                "Varcon": spaces.Box(low=np.array(varcon_lower_limit), high=np.array(varcon_upper_limit))
                })
            
            self.observation_space = spaces.Dict({
                "X_branch": spaces.MultiBinary(self.system_data.N_Branch), # Branch status
                "X_load": spaces.MultiBinary(self.system_data.N_Bus), # Load pick up status
                "PF": spaces.Box(low=self.system_data.S_lower_limit, high=self.system_data.S_upper_limit), # Branch power flow
                "QF": spaces.Box(low=self.system_data.S_lower_limit, high=self.system_data.S_upper_limit)
                })
        else:
            self.action_space = spaces.Discrete(self.system_data.N_TL + 1)
            self.observation_space = spaces.MultiBinary(self.system_data.N_Branch) 
        
        pass
    
    
    def reset(
        self, 
        options:dict = {"Specific_Disturbance": None, 
                        "Expert_Policy_Required": False,
                        "External_Seed": False},
        seed:Optional[int] = None
    ) -> Tuple[dict,dict]:
        """
        Reset the environment.
        
        Args:
            options: options := {
                                "Specific_Disturbance": list or None
                                "Expert_Policy_Required": bool
                                }
                    
                    If you want to use a SPECIFIC disturbance, use the following option:
                    "Specific_Disturbance": list of # of line, e.g: [6,11,29,32]
                    
                    Otherwise, you should use the following option to generate a RANDOM disturbance:
                    "Specific_Disturbance": None
                    
                    !! WARNING: Given the list = [] does not mean random disturbance, 
                                it means the environment will be reset to the original state !!
                    
                    The reset function can also return the expert policy if you set "Expert_Policy_Required" to True.
            
            
            seed: Random seed for the environment. 
                  !! WARNING: This parameter is only used for the first reset. 
                                You should NOT use this parameter for subsequent resets !!
                                
        Returns:
            obs: The initial observation.   
            
            info: A dictionary containing some information about the environment. 
                    If "Expert_Policy_Required" is True, the expert policy will be returned in the key "Expert_Policy". 
        """
        super().reset(seed=seed)
        
        disturbance = options["Specific_Disturbance"]
        expert_policy_required = options["Expert_Policy_Required"]
        external_seed = options["External_Seed"]
                
        self.load_rate_episode = [] # Initialize a list to store the load recovered RATE during an episode
        self.exploration_index = 0 # index to determine the instants
        
        # Generate random N-k disturbance if no specific one is given
        if disturbance == None:
            random_mode = True
            temp_disturbance_set = self.system_data.disturbance_set.copy()
            
            if external_seed:
                num_disturbance = random.randint(self.min_disturbance, self.max_disturbance)
            else:
                if self.min_disturbance==self.min_disturbance:
                    num_disturbance = self.min_disturbance
                else:
                    num_disturbance = self.np_random.integers(low=self.min_disturbance, high=self.max_disturbance) # Generate k
                
            self.disturbance = []
            # Non-repetitive random sampling
            for _ in range(num_disturbance):
                if external_seed:
                    random_disturbance = random.choice(temp_disturbance_set)
                else:
                    random_disturbance = self.np_random.choice(temp_disturbance_set)
                self.disturbance.append(random_disturbance)
                temp_disturbance_set.remove(random_disturbance)
                
        else:
            random_mode = False
            num_disturbance = len(disturbance)
            self.disturbance = disturbance
            
        # Disturbance_set -> non-tie line status
        a = np.ones((self.system_data.N_NL, self.system_data.NT))
        for dmg in self.disturbance:
            a[dmg-1, :] = 0 # 0 means the line is out of service
        self.a = a
        # Set initial condition
        X_tieline0 = np.zeros(self.system_data.N_TL) # Tieline opened by default
        Q_svc0 = np.zeros(self.system_data.N_DG-1) # SVC outputs set to 0 by default
        
        """------------Initialize N-k disturbance set & Set parameters for reset model------------
        Set N-k disturbance: a -> expert/step/reset model
        Set initial condition for reset model: X_tieline0 -> reset model
                                                Q_svc0 -> reset model
        """
        if self.opt_framework == "JuMP":
            jl.set_dmg(self.a) # Set N-k disturbance
            jl.set_ResetModel(X_tieline_input=X_tieline0, Q_svc_input=Q_svc0)
            
        elif self.opt_framework == "Gurobipy":
            self.core.set_dmg(self.a)
            self.core.set_ResetModel(X_tieline_input=X_tieline0, Q_svc_input=Q_svc0)
            
        else:
            #NOTE Add other optimization frameworks here.
            raise Exception("Optimization framework not supported!")
        
        """-------------End of initialization & setting reset model-------------"""

        """--------------Solve reset model, return initial observation--------------
        Return: _b: branch status
                _x_tieline: tieline status
                _x_load: load pick up status
                _PF: branch power flow
                _QF: branch reactive power flow
                load_value_current: total load recovered
        """
        if self.opt_framework == "JuMP":
            _b, _x_tieline, _x_load, _PF, _QF, load_value_current = jl.solve_ResetModel()
            
        elif self.opt_framework == "Gurobipy":
            _b, _x_tieline, _x_load, _PF, _QF, load_value_current = self.core.solve_ResetModel()
            
        else:
            #NOTE Add other optimization frameworks here.
            raise Exception("Optimization framework not supported!")
        
        """---------------End of solving reset model---------------"""
        
        # Record the initial observation
        self._x_load = np.round(_x_load).astype(np.int8) # Use round to avoid numerical error
        branch_obs0 = np.concatenate((self.a[:,0].flatten(),_x_tieline)).astype(np.int8)
        if self.vvo:
            self.obs = {"X_branch": branch_obs0, 
                "X_load": self._x_load,
                "PF": _PF.astype(np.float32),
                "QF": _QF.astype(np.float32)
                }
        else:
            self.obs = branch_obs0
        
        self._x_tieline = np.round(_x_tieline).astype(np.int8)
        load_rate_current = load_value_current / self.system_data.Pd_all
        self.load_rate_episode.append(load_rate_current) # Append load recovered rate to the list
        self._x_nl = np.round(_b[0:self.system_data.N_NL-1]).astype(np.int8) # Save the status of non-tie lines for expert/step model
        
        """---------------------Set expert model-----------------------------------------
        Set initial tieline status: X_tieline0 -> expert model
        Set initial load pick up status according to reset model: X_rec0 -> expert model
        Set initial non-tie line status according to reset model: X_line0 -> expert model
        Set VVO mode: vvo -> expert model
        """
        if self.opt_framework == "JuMP":
            jl.set_ExpertModel(X_tieline0_input=X_tieline0,X_rec0_input=self._x_load,X_line0_input=self._x_nl,vvo=self.vvo) # 设置Expert模型的初始状态
        elif self.opt_framework == "Gurobipy":
            self.core.set_ExpertModel(X_tieline0_input=X_tieline0,X_rec0_input=self._x_load,X_line0_input=self._x_nl,vvo=self.vvo)
        else:
            #NOTE Add other optimization frameworks here.
            raise Exception("Optimization framework not supported!")
        """---------------------End of setting expert model--------------------------------"""
        
        # Solve expert model if required
        expert_policy = None
        if expert_policy_required:
            """---------------------Solver expert model--------------------------------
            Return: 
                    solved_flag: whether the expert model is solved successfully
                    ----------(for each time step)-------------
                   | expert_b: branch status
                   | expert_x_tieline: tieline ACTIONS
                   | expert_x_load: load pick up ACTIONS
                   | expert_Pg: generator active power output
                   | expert_Qg: generator reactive power output
                   | load_value_expert: total load recovered
            """
            if self.opt_framework == "JuMP":
                solved_flag, expert_b, expert_x_tieline, expert_x_load, \
                    expert_Pg, expert_Qg, load_value_expert = jl.solve_ExpertModel()
            elif self.opt_framework == "Gurobipy":
                solved_flag, expert_b, expert_x_tieline, expert_x_load, \
                    expert_Pg, expert_Qg, load_value_expert = self.core.solve_ExpertModel()
            else:
                #NOTE Add other optimization frameworks here.
                raise Exception("Optimization framework not supported!")
            """----------------------End of solving expert model----------------------"""
            # Save expert policy if solved successfully
            if solved_flag:
                expert_b = np.round(expert_b).astype(np.int8)
                expert_x_tieline = np.round(expert_x_tieline).astype(np.int8)
                expert_x_branch = np.concatenate((self.a,expert_x_tieline)).astype(np.int8)
                expert_branch_obs = np.concatenate((branch_obs0.reshape(-1, 1),expert_x_branch[:,:-1]), axis=1).astype(np.int8)
                expert_x_load = np.round(expert_x_load).astype(np.int8)
                expert_P_sub = expert_Pg[0,:]
                expert_Q_sub = expert_Qg[0,:]
                expert_Q_svc = expert_Qg[1:,:]
                load_value_expert = load_value_expert.flatten()
                expert_load_rate = load_value_expert / np.sum(self.system_data.Pd, axis=0)
                
                # Calculate tieline action
                expert_tieline_action = np.zeros(expert_x_tieline.shape[1], dtype=np.int8)
                temp_expert_x_tieline = np.concatenate((X_tieline0.reshape(-1, 1), expert_x_tieline.copy()), axis=1) # Add initial tieline status
                for col in range(1, temp_expert_x_tieline.shape[1]):
                    diff = temp_expert_x_tieline[:, col] - temp_expert_x_tieline[:, col-1]
                    row_indices = np.where(diff != 0)[0]
                    if len(row_indices) > 0:
                        expert_tieline_action[col-1] = row_indices[0] + 1
                expert_tieline_action = np.expand_dims(expert_tieline_action, axis=1)
                expert_policy = {
                    "Branch_Energized": expert_b, 
                    "Load_Energized": expert_x_load,
                    "X_branch": expert_x_branch,
                    "Branch_Obs": expert_branch_obs,
                    "X_tieline": expert_x_tieline,
                    "TieLine_Action": expert_tieline_action,
                    "P_sub": expert_P_sub,
                    "Q_sub": expert_Q_sub,
                    "Q_svc": expert_Q_svc,
                    "Load_Rate": expert_load_rate
                    }
                
                # Otherwise, expert_policy keeps None
                                
        # Construct info
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
    
    
    def step(
        self, 
        action:Union[dict,int]
    ) -> Tuple[dict, float, bool, dict]:
        """
        Take an action and return the next observation and reward.
        
        Args:
            action: 
            ==========   ==========   ========================================================
            VVO Enable      Type                            Example
            ==========   ==========   ========================================================
            True          dict         action = {"Tieline": 4, 
                                                "Varcon": np.array([0.01, -0.01, 0.02, 0, 0, 0], 
                                                dtype=np.float32)}
                                                
            False         int          action = 4   
            ==========   ==========   ========================================================
            
        Returns:
                (defined by gymnasium)
                observation    reward    terminated    truncated    info
                -----------------------------------------------------------
                    obs        reward      done         False        info
        """

        # Automatically check if the action is valid  
        assert self.action_space.contains(action), "Invalid action. See docstring of step() for more information."
        
        # Set action
        x_tieline_input = self._x_tieline.copy() # use copy to avoid changing the original value
        if self.vvo:
            action_tieline = action["Tieline"]
        else:
            action_tieline = action
            
        if action_tieline>=1:
            x_tieline_input[action_tieline-1] = 1 # one step one tielie closed, and no tieline opened
        
        if self.vvo:
            q_svc_input = action["Varcon"]
        else:
            q_svc_input = None
        
        event_log = None # For distinguishing different events
        action_accepted = False # Whether the action is accepted by the environment
        
        """ 
        Check the exploration index and calculate the reward
        
        For clear illustration, we take 33BW system with 5 tie-lies as an example. The agent has five step chances to take actions:
        ----------------------------------------------------------------------------------------------------------------------------
            S0   -A1->   S1    -A2->   S2    -A3->    S3    -A4->    S4    -A5->   S5 (terminal)    -A6->       S5 (terminal)
         reset  step1        step2          step3          step4          step5                     step6        NO TRANSITION
                 R1            R2             R3            R4              R5                       0   
         --------------------------------------------------------------------------------------------------------------------------
                idx:0->1     idx=1->2       idx=2->3       idx=3->4       idx=4->5                
              (_step->step_)                                           thus, if idx>=4,          done keeps True
                                                                done=True after transition
         --------------------------------------------------------------------------------------------------------------------------
         S# : state; A# : action; R#: reward 
         _step: begin step(); step_: after step(); idx: exploration_index; step#: execution step() at step#
        """
        if self.exploration_index <= self.exploration_total-1:
            # Check if this is the last step in this episode      
            if self.exploration_index == self.exploration_total-1: 
                done = True
            else:
                done = False
    
            """-------------------Set & Solve step model-------------------
            Set load pick up status at last step: X_rec0 -> step model
            Input tieline action: x_tieline_input -> step model 
            Input SVC command: q_svc_input -> step model
            Set VVO mode: vvo -> step model
            
            Results:
                solved: whether the step model is solved successfully
                _b: branch status
                _x_tieline: tieline status
                _x_load: load pick up status
                _PF: branch power flow
                _QF: branch reactive power flow
                load_value_new: total load recovered
                e_Qsvc: track error of SVC command
            """
            if self.opt_framework == "JuMP":
                jl.set_StepModel(X_rec0_input=self._x_load, X_tieline_input=x_tieline_input, 
                                Q_svc_input=q_svc_input, vvo=self.vvo)
                results = jl.solve_StepModel()
            elif self.opt_framework == "Gurobipy":
                self.core.set_StepModel(X_rec0_input=self._x_load, X_tieline_input=x_tieline_input, 
                                Q_svc_input=q_svc_input, vvo=self.vvo)
                results = self.core.solve_StepModel()
            else:
                #NOTE Add other optimization frameworks here.
                raise Exception("Optimization framework not supported!")
            """-------------------End of setting & solving step model-------------------"""
   
            # Record the results
            solved, _b, _x_tieline, _x_load, _PF, _QF, load_value_new, e_Qsvc = results
            
            # If infeasible, it means the tieline cannot be closed,
            # the tieline status remains unchanged, and the load will not be recovered
            #NOTE: 'Soft-constraint' technologies are employed to distinguish different events
            if not solved:
                event_log = "Infeasible Tieline Action"
                reward = -1000
                self.load_rate_episode.append(self.load_rate_episode[-1]) # 负荷不回复，保持上一步状态
            else:
                # Check if the action is valid
                _x_nl = np.round(_b[0:self.system_data.N_NL-1]).astype(np.int8)
                flag_x_nl = np.any(_x_nl<self._x_nl) # If any closed non-tie line becomes open, the action is invalid (radiation violation)
                flag_e_Qsvc = e_Qsvc >= 1e-6 # If SVC command is infeasible, the track error is greater than the threshold
                if flag_x_nl: # For radiation violation
                    event_log = "Infeasible Topology"
                    reward = -1000 
                    self.load_rate_episode.append(self.load_rate_episode[-1]) # keep the previous load recovery rate
                elif flag_e_Qsvc: # For infeasible SVC command
                    event_log = "Infeasible SVC Scheduling"
                    reward = -1000
                    self.load_rate_episode.append(self.load_rate_episode[-1])
                else: # If the action is valid
                    action_accepted = True
                    load_rate_new = load_value_new / self.system_data.Pd_all
                    if load_rate_new-self.load_rate_episode[-1]>=1e-4: # Check if the recovered load increases
                        event_log = "Increased Load Recovery"
                        reward = 150
                    else:
                        event_log = "Maintained Load Recovery"
                        reward = -10
                    
                    # Update observation                
                    self._x_nl = _x_nl
                    self._x_tieline = np.round(_x_tieline).astype(np.int8)
                    self._x_load = np.round(_x_load).astype(np.int8)
                    self.load_rate_episode.append(load_rate_new) # Append load recovered rate to the list
                    
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
        
        # Construct info
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
        pass
