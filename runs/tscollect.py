import gymnasium
import tianshou as ts
from tianshou.data.collector import Collector as Collector
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data import Batch as Batch
from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net as Net,MLP as Mlp
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor,Critic

import tianshou.policy as Policy
from typing import Optional,Any,Union,Dict
import gymnasium as gym
from metagym import liftsim
from metagym.liftsim.environment.mansion.utils import state_transform,action_to_list,flatten_state
import numpy as np
import torch
import torch.nn as nn
single = gym.make('liftsim-v0')

env_nums = 2
env = DummyVectorEnv([lambda : gym.make('liftsim-v0') for i in range(2)])


class my_random(Policy.BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        act = single.action_space.sample()

        for i in range(env_nums-1):
            act = np.concatenate([single.action_space.sample(),act],axis=0)
        act = np.reshape(act, (env_nums,-1))

        return Batch(act=act)
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}
    def map_action_inverse(self,act):
        return act
    def map_action(self,action):
        # action_len = len(action)
        return action

def process_fn(**kwargs)->Batch:
    if(kwargs.__contains__('obs')):
        kwargs['obs']=[flatten_state(_) for _ in kwargs['obs']]
    if(kwargs.__contains__('obs_next')):
        kwargs['obs_next'] = [flatten_state(_) for _ in kwargs['obs_next']]
    return Batch(kwargs)
test_policy = my_random()

# 电梯的状态元组是一个多状态的，无法对应于DQN
state_shape = single.observation_dim
action_shape = 8

Actornet = Net(state_shape=state_shape,action_shape = action_shape,hidden_sizes=[256,256,128,64,32])
Criticnet = Net(state_shape=state_shape,action_shape = action_shape,hidden_sizes=[256,256,128,64,32])
ActorCriticnet = ActorCritic(Actornet,Criticnet)

optim = torch.optim.SGD(ActorCriticnet.parameters(),lr = 0.003)
test_policy2 = Policy.PPOPolicy(actor=Actornet,critic=Criticnet,dist_fn=torch.distributions.Categorical
                                ,optim=optim)
# env = DummyVectorEnv([lambda : gym.make('liftsim-v0') for i in range(5)])
buffer = VectorReplayBuffer(total_size=10000,buffer_num=2)
collector = Collector(policy=test_policy2,env=env,buffer=buffer,preprocess_fn=process_fn)

collector.collect(n_step=2)

# 2步的话就2个？
