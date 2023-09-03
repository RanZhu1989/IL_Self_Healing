import selfhealing_env
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

env = gym.make("SelfHealing-v0",
               opt_framework="JuMP",
               solver="cplex",
               data_file="Case_33BW_Data.xlsx",
               solver_display=False,
               min_disturbance=2, 
               max_disturbance=5, 
               vvo=False,
               Sb=100,
               V0=1.05,
               V_min = 0.95,
               V_max = 1.05)
               

print(env.action_space)
print(env.observation_space)
check_env(env)