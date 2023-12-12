from collections import defaultdict
import metagym.liftsim
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from gymnasium.spaces import Dict,Discrete,MultiDiscrete,MultiBinary,Sequence,Tuple,Box
from gymnasium.spaces.utils import flatten_space,flatten


device = "cpu" if not torch.has_cuda else "cuda:0"
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frame_skip = 1 # 跳帧 对于一定数量的帧使用同样的动作
frame_per_batch = 1000
total_frames = 10000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
# 学习的时候用更小的batch 训练

'''我们以一个例子来说明这个过程：
假设我们有一个大的数据集，
这个数据集的大小是frames_per_batch。
在训练开始时，我们会把这个大的数据集划分为多个小的数据子集，
每个子集的大小就是sub_batch_size。
然后，在每次训练过程中，我们会使用一个这样的小数据子集来训练我们的模型。
当所有的子集都用于训练一次后，我们就完成了一个epoch。这个过程会重复进行，直到满足一定的训练周期或者其它早停条件。
这种做法的好处是，它可以有效地利用内存，
避免一次性加载过大的数据集导致内存不足的问题。同时，通过调整sub_batch_size的大小，我们可以控制每次训练所使用的数据量，从而更好地控制训练过程。
'''
num_epochs = 10 # 总共10个epoch

clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = GymEnv("liftsim-v0", device=device,frame_skip=frame_skip)
