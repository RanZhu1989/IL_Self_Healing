import selfhealing_env
import gymnasium as gym

import numpy as np

env = gym.make("SelfHealing-v0",
               solver="cplex",
               data_file="Case_33BW_Data.xlsx",
               solver_display=True,
               min_disturbance=2, 
               max_disturbance=5, 
               vvo=False)
reset_option = {
    "Specific_Disturbance": [6],
    "Expert_Policy_Required": True
}
obs,info = env.reset(options=reset_option)
# obs,info = env.reset(disturbance=[3,9,14,21,23])
# obs,info = env.reset(disturbance=None)
print(info)

action = 1

obs, reward, done, _, info = env.step(action)
print(reward)
print(obs)
print(info)
print(done)

