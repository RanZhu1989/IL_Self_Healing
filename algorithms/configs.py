import argparse

parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--env_id', type=str, default='SelfHealing-v0')
parser.add_argument('--data_file', type=str,default='Case_33BW_Data.xlsx')
parser.add_argument('--opt_framework', type=str,default='Gurobipy')
parser.add_argument('--solver', type=str, default='gurobi')
parser.add_argument('--solver_display', type=bool,default=False)
parser.add_argument('--vvo', type=bool,default=False)
parser.add_argument('--min_disturbance', type=int,default=1)
parser.add_argument('--max_disturbance', type=int,default=1)
parser.add_argument('--Sb', type=int,default=100)
parser.add_argument('--V0', type=float,default=1.05)
parser.add_argument('--V_min', type=float,default=0.95)
parser.add_argument('--V_max', type=float,default=1.05)

# Public training parameters
parser.add_argument('--result_folder_name', type=str, default='results')
parser.add_argument('--forced_cpu', type=bool,default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--test_iterations', type=int, default=5)
parser.add_argument('--train_epochs', type=int, default=1000)
parser.add_argument('--on_policy_env_iters', type=int,default=20)


# Imitation learning parameters
parser.add_argument('--IL_lr', type=float,default=1.0e-3)
parser.add_argument('--IL_batch_size', type=int,default=16)
parser.add_argument('--IL_used_episodes', type=int,default=200)
parser.add_argument('--IL_used_samples', type=int,default=100)
parser.add_argument('--IL_update_iters', type=int,default=5)
parser.add_argument('--GAIL_d_iters',type=int, default=5)
parser.add_argument('--GAIL_d_lr', type=float,default=5.0e-3)

# PPO parameters
parser.add_argument('--PPO_actor_lr', type=float,default=1.0e-3)
parser.add_argument('--PPO_critic_lr',type=float, default=5.0e-3)
parser.add_argument('--PPO_gamma', type=float,default=0.98)
parser.add_argument('--PPO_advantage_lambda', type=float,default=0.95)
parser.add_argument('--PPO_clip_epsilon', type=float,default=0.2)
parser.add_argument('--PPO_train_iters', type=int,default=5)


# DQN parameters
parser.add_argument('--DQN_lr', type=float,default=1.0e-3)
parser.add_argument('--DQN_gamma', type=float,default=0.90)
parser.add_argument('--DQN_epsilon', type=float,default=0.1)
parser.add_argument('--DQN_buffer_capacity', type=int,default=20000)
parser.add_argument('--DQN_replay_start_size',type=int, default=64)
parser.add_argument('--DQN_replay_frequent',type=int, default=4)
parser.add_argument('--DQN_target_sync_frequent',type=int, default=20)
parser.add_argument('--DQN_batch_size', type=int,default=32)



# parse arguments
args = parser.parse_args()

