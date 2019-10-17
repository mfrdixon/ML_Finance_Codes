from gym.envs.registration import register

register(
    id='fcw-v0',
    entry_point='gym_fcw.envs:FcwEnv',
)
