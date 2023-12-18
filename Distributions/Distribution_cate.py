import torch
import torch.nn as nn

k = 10
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
class ele_dict(torch.distributions.Categorical):
    def __init__(self,logit):
        self.logit = logit
    def sample(self, sample_shape=torch.Size()):
        batch_actions = []
        for j in range(self.logit.shape[0]):
            action_parts = []
            for i in range(4):
                target_logit = self.logit[j:j + 1, i * (k + 3):i * (k + 3) + k]
                target_dist = nn.functional.softmax(target_logit, dim=-1)
                target_action = torch.distributions.Categorical(target_dist).sample()

                dir_logit = self.logit[j:j + 1, i * (k + 3) + k:i * (k + 3) + k + 3]
                dir_dist = nn.functional.softmax(dir_logit, dim=-1)
                dir_action = torch.distributions.Categorical(dir_dist).sample()
                action_parts.append(target_action + 1)
                action_parts.append(dir_action - 1)
            batch_actions.append(
                torch.cat(action_parts, dim=-1)
            )
        actions = torch.stack(batch_actions)
        return actions