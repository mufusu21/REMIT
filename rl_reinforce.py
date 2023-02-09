import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space_n):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space_n)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)


class REINFORCE():
    """https://github.com/chingyaoc/pytorch-REINFORCE/blob/4a38741bcc32ed9a4ed92cd880e098b91ba82137/reinforce_discrete.py#L47"""
    def __init__(self, hidden_size, num_inputs, action_space_n, lr):
        self.model = Policy(hidden_size, num_inputs, action_space_n).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):     # [128, 58]
        probs = self.model(state)   # [128, 3]
        return probs

    def policy_learn(self, reward, log_probs, retain_graph=False):
        loss = -1.0 * reward * log_probs.sum()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
