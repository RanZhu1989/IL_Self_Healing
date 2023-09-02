import selfhealing_env
import gymnasium as gym

env = gym.make("SelfHealing-v0",
               opt_framework="JuMP",
               solver="cplex",
               data_file="Case_33BW_Data.xlsx",
               solver_display=True,
               min_disturbance=2, 
               max_disturbance=5, 
               vvo=False)

print(env.action_space)
print(env.observation_space)