import torch
import torch.nn as nn

k = 10
num_elevators = 4
def logit_actions(logit):
    for j in range(logit.shape[0]):
        action_parts = []
        for i in range(4):
            target_logit = logit[j:j + 1, i * (k + 3):i * (k + 3) + k]
            print(target_logit)
            target_dist = nn.functional.softmax(target_logit, dim=-1)
            target_action = torch.distributions.Categorical(target_dist).sample()

            dir_logit = logit[j:j + 1, i * (k + 3) + k:i * (k + 3) + k + 3]
            dir_dist = nn.functional.softmax(dir_logit, dim=-1)
            dir_action = torch.distributions.Categorical(dir_dist).sample()

            action_parts.append(target_action + 1)
            action_parts.append(dir_action - 1)
        batch_actions.append(
            torch.cat(action_parts, dim=-1)
        )
        actions = torch.stack(batch_actions)
        return actions
class EleCategorical(torch.distributions.Categorical):
    def __init__(self,logit):
        # print(logit)
        self.logit = logit # 这是神经网络的输出
        self._batch_shape = torch.Size([logit.shape[0]])
        self._event_shape = torch.Size([num_elevators*2]),
        self.distributions = []
        for j in range(self.logit.shape[0]):
            for i in range(num_elevators):
                target_logit = self.logit[j:j + 1, i * (k + 3):i * (k + 3) + k]
                target_dist = nn.functional.softmax(target_logit, dim=-1)
                self.distributions.append(torch.distributions.Categorical(target_dist))
                dir_logit = self.logit[j:j + 1, i * (k + 3) + k:i * (k + 3) + k + 3]
                dir_dist = nn.functional.softmax(dir_logit, dim=-1)
                self.distributions.append(torch.distributions.Categorical(dir_dist))

    def sample(self):
        actions = [distribution.sample() + (1 if i%2==0 else -1 ) for i,distribution in enumerate(self.distributions)]
        actions = torch.cat(actions,dim=0).reshape(self.logit.shape[0],-1)
        return actions

    def log_prob(self, value):
        # 验证值形状
        expected_size = num_elevators * 2
        if value.shape[1] != expected_size:
            raise ValueError(f"Invalid input shape {value.shape}...")

        probs = []
        tmp = 0
        for actions in value:
            act_prob = 0
            for act in actions:
                v = act.clone()
                v += -1 if tmp % 2==0 else 1
                act_prob += self.distributions[tmp].log_prob(v)
                tmp += 1
            probs.append(act_prob)

        return torch.cat(probs, dim=0).view(self.batch_shape[0],-1)

    def entropy(self):
        ents = []
        for dist in self.distributions:
            ents.append(dist.entropy())
        return torch.cat(ents, dim=0).reshape(self.batch_shape[0], -1).sum(-1)

# ex1 = torch.ones(100,112)
# ex2 = torch.ones(100,8)
# dis = EleCategorical(ex1)
# end =dis.log_prob(ex2)
# print(end.shape)