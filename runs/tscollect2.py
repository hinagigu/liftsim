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
from metagym.liftsim.Distributions.Distribution_cate import EleCategorical
import numpy as np
import torch
import torch.nn as nn

from  critic_test import criticnet
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
env_nums = 2
env = DummyVectorEnv([lambda : gym.make('liftsim-v0') for i in range(env_nums)])


def stat_parameters(model):
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())
        else:
            biases.append(param.data.cpu().numpy())
    fl_wei = []
    fl_bia = []
    for i in weights:
        fl_wei.append(np.ravel(i))
    for i in biases:
        fl_bia.append(np.ravel(i))
    weights = np.concatenate(fl_wei)
    biases = np.concatenate(fl_bia)

    return np.mean(weights), np.std(weights), np.mean(biases), np.std(biases)

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

def preprocess_fn(**kwargs)->Batch:
    if(kwargs.__contains__('obs')):
        kwargs['obs']=[flatten_state(_) for _ in kwargs['obs']]
    if(kwargs.__contains__('obs_next')):
        kwargs['obs_next'] = [flatten_state(_) for _ in kwargs['obs_next']]
    return Batch(kwargs)
test_policy = my_random()

# 电梯的状态元组是一个多状态的，无法对应于DQN
state_shape = 100
action_shape = 52

Actornet = Net(state_shape=state_shape,action_shape = 64,hidden_sizes=[256,128,64],device='cuda:0').to('cuda:0')
actor = Actor(Actornet,action_shape=52,softmax_output=False,preprocess_net_output_dim=64,device='cuda:0').to('cuda:0')
critic = criticnet().to('cuda:0')
ActorCriticnet = ActorCritic(actor,critic).to('cuda:0')


optim = torch.optim.SGD(ActorCriticnet.parameters(),lr = 0.001)
test_policy2 = Policy.PPOPolicy(actor=actor,critic=critic,dist_fn=EleCategorical
                                ,optim=optim).to('cuda:0')
buffer = VectorReplayBuffer(total_size=10000,buffer_num=2)
train_collector = Collector(policy=test_policy2,env=env,buffer=buffer,preprocess_fn=preprocess_fn)
test_collector = Collector(policy=test_policy2,env=env,buffer=VectorReplayBuffer(total_size=1000,buffer_num=2),preprocess_fn=preprocess_fn)

writer = SummaryWriter('tsboard')
# writer.add_text("args")

def train_fn(num_epoch: int, step_idx: int):
    mean_w, std_w, mean_b, std_b = stat_parameters(actor)
    writer.add_scalars('actor_params', {
        'mean_w': mean_w,
        'std_w': std_w,
        'mean_b': mean_b,
        'std_b': std_b
    },num_epoch)


logger = TensorboardLogger(writer)

# trainner = ts.trainer.OnpolicyTrainer(
#     policy=test_policy2,train_collector=train_collector,test_collector=test_collector,max_epoch=10,repeat_per_collect=2,
#     batch_size=128,step_per_collect=512,episode_per_test=1,step_per_epoch=1000,logger=logger,train_fn=train_fn
# )
trainner = ts.trainer.OnpolicyTrainer(
    policy=test_policy2,train_collector=train_collector,test_collector=test_collector,max_epoch=10,repeat_per_collect=1,
    batch_size=1024,episode_per_test=1,episode_per_collect=1,logger=logger,train_fn=train_fn,step_per_epoch=100
)

result = trainner.run()
# 2步的话就2个？
