import os
from environment import *
# from gym.envs.registration import register
from gymnasium.envs.registration import register

register(
    id='liftsim-v0',
    entry_point='environment.env:LiftSim',
    kwargs={
        "config_file": os.path.join(os.path.dirname(__file__)+'/config.ini'),
    }
)
