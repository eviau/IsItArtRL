from gym.envs.registration import register

register(
    id='isitartrl-v0',
    entry_point='gym_isitartrl.envs:isitartrlEnv',
)
register(
    id='isitartrl-extrahard-v0',
    entry_point='gym_isitartrl.envs:isitartrlExtraHardEnv',
)
