import selfhealing_env
import gymnasium as gym

import numpy as np

env = gym.make("SelfHealing-v0",
               opt_framework="JuMP",
               solver="cplex",
               data_file="Case_33BW_Data.xlsx",
               solver_display=False,
               min_disturbance=1, 
               max_disturbance=1, 
               vvo=True,
               V0=1.05,
               V_max=1.05,
               V_min=0.95)
reset_option = {
    "Specific_Disturbance": [27],
    "Expert_Policy_Required": True,
    "External_RNG": None
}
obs,info = env.reset(options=reset_option)

print(info)


