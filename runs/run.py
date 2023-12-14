import gym
import gymnasium
import metagym.liftsim
import numpy as np
import torch
from metagym.liftsim.environment.env import LiftSim
from metagym.liftsim.environment.mansion.utils import state_transform,action_to_list,flatten_state
from gymnasium.wrappers.compatibility import EnvCompatibility
# env = LiftSim()
env = gymnasium.make('liftsim-v0')
# iteration = env.iterations
state,info= env.reset()

# action = [10, 1, 10, 1, 10, 1, 5, 1]
# q = env.action_space.sample()
# print(action_to_list(q))
for i in range(100):
    action = env.action_space.sample()
    action = action_to_list(action)
    next_state, reward, terminated,truncated,info = env.step(action)
    print(env.observation_space.contains(next_state),reward)
    # for key in next_state['elevator_states']:
    #     if env.observation_space['elevator_states'][key].contains(next_state['elevator_states'][key]) == False:
    #         print(key,env.observation_space['elevator_states'][key].sample().dtype,next_state['elevator_states'][key].dtype)
