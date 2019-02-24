from gym.envs.registration import register

register(
    id='solarescape-v0',
    entry_point='gym_solarescape.envs:SolarescapeEnv',
)