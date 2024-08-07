import selfhealing_env
import gymnasium as gym

import numpy as np

env = gym.make("SelfHealing-v0",
               opt_framework="JuMP",
               solver="cplex",
               data_file="Case_33BW_Data.xlsx",
               solver_display=False,
               min_disturbance=2, 
               max_disturbance=5, 
               vvo=False,
               V0=1.05,
               V_max=1.05,
               V_min=0.95)
reset_option = {
    "Specific_Disturbance": [23,30],
    "Expert_Policy_Required": True,
    "External_RNG": None
}
obs,info = env.reset(options=reset_option)

print(info)

action = {"Tieline":3,
          "Varcon": np.array([0, 0, 0, 0, 0,0], dtype=np.float32)   # [-0.002, 0, 0.002, 0.002, -0.002,0]
        }
action = 5

obs, reward, done, _, info = env.step(action)
print(reward)
print(obs)
print(info)
print(done)


action = {"Tieline":1,
          "Varcon": np.array([0, 0, 0, 0, 0,0], dtype=np.float32)
        }
action = 4

obs, reward, done, _, info = env.step(action)
print(reward)
print(obs)
print(info)
print(done)

# action = {"Tieline":2,
#           "Varcon": np.array([0, 0, 0, 0, 0,0], dtype=np.float32)
#         }
# action = 2

# obs, reward, done, _, info = env.step(action)
# print(reward)
# print(obs)
# print(info)
# print(done)

# action = {"Tieline":4,
#           "Varcon": np.array([0, 0, 0, 0, 0,0], dtype=np.float32)
#         }
# action = 4

# obs, reward, done, _, info = env.step(action)
# print(reward)
# print(obs)
# print(info)
# print(done)

# action = {"Tieline":4,
#           "Varcon": np.array([0, 0, 0, 0, 0,0], dtype=np.float32)
#         }
# action = 4

# obs, reward, done, _, info = env.step(action)
# print(reward)
# print(obs)
# print(info)
# print(done)

# action = {"Tieline":4,
#           "Varcon": np.array([0, 0, 0, 0, 0,0], dtype=np.float32)
#         }
# action = 4

# obs, reward, done, _, info = env.step(action)
# print(reward)
# print(obs)
# print(info)
# print(done)