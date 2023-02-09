import numpy as np
import torch
import tqdm
from collections import namedtuple, deque
import random
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'X', 'y'))

class ReplayBuffer():
    """https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Trainer():
    """Runs games for given agents. Optionally will visualise and save the results"""
    def __init__(self, env, model, agent, data_loader, capacity):
        self.env = env
        self.cdr_model = model
        self.agent = agent
        self.replay_buffer = ReplayBuffer(capacity)
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.data_loader = data_loader

    def initial_agent_policy(self, max_episode):
        """
        get the initial params for three Aspect Selector agent policy models.
        only one epoch of the data_meta.
        """
        self.agent.model.train()
        i_episode = 0
        for X, y in self.data_loader:
        # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):
            i_episode = i_episode + 1
            action = self.event_softmax(torch.ones(X.shape[0], self.env.action_space_n).cuda())    # [128, 3]
            log_probs = action.log()  # [128, 3]
            reward, _, _ = self.env.step(X, y, action)
            self.agent.policy_learn(reward, log_probs, True)
            # print("Episode: {}, reward: {}".format(i_episode, reward))
            if i_episode >= max_episode:
                break

    def cdr_with_fixed_as(self, optimizer):
        """
        in the CDR process, we fix the AS policy param and learn the CDR param
        """
        self.cdr_model.train()
        all_reward = 0
        for X, y in self.data_loader:
        # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):
            state = self.env.reset(X, y)    # state [128, 58]
            action = self.agent.select_action(state)     # [128, 3]
            reward, loss, _ = self.env.step(X, y, action)   # reward [1]
            self.replay_buffer.push(state.clone().detach(), action.clone().detach(), reward.clone().detach(), X.clone().detach(), y.clone().detach())
            all_reward += reward.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print("all Episode: {}, all_reward: {}".format(len(self.replay_buffer), all_reward))

    def as_from_replay_buffer(self, method):
        """
        in the AS process, we learn from the Replay Buffer
        """
        self.agent.model.train()
        if method == 'ori_reward':
            for transition in self.replay_buffer.memory:
                action = Variable(transition.action.data, requires_grad=True)
                reward = Variable(transition.reward.data, requires_grad=True)
                self.agent.policy_learn(reward, action.log())

        elif method == 'new_reward':
            for transition in self.replay_buffer.memory:
                X = transition.X.data
                y = transition.y.data
                action = Variable(transition.action.data, requires_grad=True)
                reward, _, _ = self.env.step(X, y, action)
                self.agent.policy_learn(reward, action.log())
        else:
            raise ValueError("other_reward not defined...")


class Infer():
    def __init__(self, env, model, agent, data_loader):
        self.env = env
        self.cdr_model = model
        self.agent = agent
        self.data_loader = data_loader

    def infer_weight(self):
        self.cdr_model.eval()
        self.agent.model.eval()
        weight = []
        ids = []
        with torch.no_grad():
            for X, y in self.data_loader:
                state = self.env.reset(X, y)    # state [128, 58]
                userids, itemids = X[:, 0].cpu().tolist(), X[:, 1].cpu().tolist()
                action = self.agent.select_action(state)     # [128, 3]
                weight.extend(action.cpu().tolist())
                ids.extend(list(zip(userids, itemids)))
        weight = [list(ids[i]) + w for i,w in enumerate(weight)]
        return weight

    def infer_emb(self):
        self.cdr_model.eval()
        self.agent.model.eval()
        uiu_df = []
        uiciu_df = []
        uibiu_df = []
        with torch.no_grad():
            for X, y in self.data_loader:
            # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):
                state = self.env.reset(X, y)    # state [128, 58]
                action = self.agent.select_action(state)     # [128, 3]
                uiu_emb, uiciu_emb, uibiu_emb = self.env.get_emb(X, y, action)   # [128]
                uiu_df.extend(uiu_emb.cpu().tolist())
                uiciu_df.extend(uiciu_emb.cpu().tolist())
                uibiu_df.extend(uibiu_emb.cpu().tolist())
        return uiu_df, uiciu_df, uibiu_df

    def eval_mae(self):
        print('Evaluating MAE:')
        self.cdr_model.eval()
        self.agent.model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in self.data_loader:
            # for X, y in tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0):
                state = self.env.reset(X, y)    # state [128, 58]
                action = self.agent.select_action(state)     # [128, 3]
                _, _, pred = self.env.step(X, y, action)   # [128]
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
