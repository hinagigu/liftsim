import gym
import metagym.liftsim
import numpy as np
from metagym.liftsim.environment.env import LiftSim
from metagym.liftsim.environment.mansion.utils import state_transform
from gymnasium.wrappers.compatibility import EnvCompatibility
env = LiftSim()
# env = gym.make('liftsim-v0')
env.seed(1998)
# iteration = env.iterations
state = env.reset()
action = [0, 1, 0, 1, 0, 1, 0, 1]
# q,k,v = state_transform(state)
# print(type(q))
# print(len(q))
# print(q)
dict,q,k,v = state_transform(state)
# for key in dict['elevator_states']:
#     print(key,env.observation_space['elevator_states'][key].contains(dict['elevator_states'][key]))
# s = dict['elevator_states']['ReservedTargetFloors']
# print(s)
# p = env.observation_space['elevator_states']['ReservedTargetFloors'].sample()
# ll = np.zeros_like(p)
# print(s.dtype)
# print(ll.dtype)
# print(p.dtype)


# for i in q:

#     print(env.elevator_space.contains(i))
# print(q[0])
# for key in env.elevator_space:
#     print(key,env.elevator_space[key].contains(q[0][key]))

# print(env.elevator_space.sample())

# for i in range(10000):
#     next_state, reward, _, _ = env.step(action)
#     if len(next_state.RequiringUpwardFloors) or len(next_state.RequiringDownwardFloors) >0:
#         print(next_state.RequiringUpwardFloors,next_state.RequiringDownwardFloors)
#     env.render()
    # print(env.get_mansion().get_generator().generate_person())
#     env.render()
