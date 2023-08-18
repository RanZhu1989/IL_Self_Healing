from gymnasium.envs.registration import register

register(
    id="SelfHealing-v0",
    entry_point="selfhealing_env.envs:SelfHealing_Env",
)