import argparse

parser = argparse.ArgumentParser()

# TODO: Current python-julia can not use argparse, so we need to set the parameters in the code
# Environment
parser.add_argument('--env_id', default='SelfHealing-v0')
parser.add_argument('--data_file', default='Case_33BW_Data.xlsx')
parser.add_argument('--opt_framework', default='Gurobipy')
parser.add_argument('--solver', default='gurobi')
parser.add_argument('--solver_display', default=False)
parser.add_argument('--vvo', default=False)
parser.add_argument('--min_disturbance', default=1)
parser.add_argument('--max_disturbance', default=1)
parser.add_argument('--Sb', default=100)
parser.add_argument('--V0', default=1.05)
parser.add_argument('--V_min', default=0.95)
parser.add_argument('--V_max', default=1.05)

# Public training parameters
parser.add_argument('--result_folder_name', default='results')
parser.add_argument('--forced_cpu', default=False)
parser.add_argument('--seed', default=0)
parser.add_argument('--test_iterations', default=5)
parser.add_argument('--train_epochs', default=500)


# Imitation learning parameters
parser.add_argument('--IL_lr', default=1.0e-3)
parser.add_argument('--IL_batch_size', default=32)
parser.add_argument('--IL_used_samples', default=250)
parser.add_argument('--DAgger_update_iters', default=5)
parser.add_argument('--GAIL_d_iters', default=5)
parser.add_argument('--GAIL_d_lr', default=1.0e-3)

# PPO parameters
parser.add_argument('--PPO_actor_lr', default=1.0e-3)
parser.add_argument('--PPO_critic_lr', default=1.0e-2)
parser.add_argument('--PPO_gamma', default=0.98)
parser.add_argument('--PPO_advantage_lambda', default=0.95)
parser.add_argument('--PPO_clip_epsilon', default=0.2)
parser.add_argument('--PPO_train_iters', default=10)

# DQN parameters
parser.add_argument('--DQN_lr', default=1.0e-3)
parser.add_argument('--DQN_gamma', default=0.90)
parser.add_argument('--DQN_epsilon', default=0.1)
parser.add_argument('--DQN_buffer_capacity', default=20000)
parser.add_argument('--DQN_replay_start_size', default=64)
parser.add_argument('--DQN_replay_frequent', default=4)
parser.add_argument('--DQN_target_sync_frequent', default=20)
parser.add_argument('--DQN_batch_size', default=32)



# parse arguments
args = parser.parse_args()

