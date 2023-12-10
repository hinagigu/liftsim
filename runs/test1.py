import gymnasium.spaces as spaces
from gymnasium.vector.utils.spaces import batch_space

import numpy as np

# 原始列表，包含多个字典
# original_list = [
#     {"key": 1},
#     {"key": 2},
#     {"key": 3}
# ]
# fd = spaces.Dict({'key':spaces.Discrete(10)})
# batch_spaces = batch_space(fd,3)
# # 新的字典，将每个键的值扩展一个维度
# new_dict = {key: np.stack([d[key] for d in original_list]) for key in original_list[0].keys()}
#

# print(batch_spaces.contains(new_dict))
test  = spaces.Box(0,1)
print(test.contains([0.5]))