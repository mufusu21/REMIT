import torch


class CDREnvironment():
    """https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/135d3e2e06bbde2868047d738e3fc2d73fd8cc93/environments/Ant_Navigation_Environments.py"""
    def __init__(self, reward_name, action_space_n, model, emb_dim, stage):
        self.reward_name = reward_name
        self.state = None
        self.action_space_n = action_space_n
        self.model = model
        self.state_dim = 4 + 5*emb_dim
        self.stage = stage

    def reset(self, X, y):
        iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb = self.model(X, stage=self.stage)
        tgt_pred = torch.sum(uid_emb * iid_emb, dim=1).unsqueeze(1)     # [128, 1]
        #tgt_mae = self.criterion(tgt_pred, y.float())     # [128, 1]
        uiu_pred = torch.sum(uiu_emb * iid_emb, dim=1).unsqueeze(1)
        #uiu_mae = self.criterion(uiu_pred, y.float())
        uiciu_pred = torch.sum(uiciu_emb * iid_emb, dim=1).unsqueeze(1)
        #uiciu_mae = self.criterion(uiciu_pred, y.float())
        uibiu_pred = torch.sum(uibiu_emb * iid_emb, dim=1).unsqueeze(1)
        #uibiu_mae = self.criterion(uibiu_pred, y.float())
        # print(iid_emb.shape, uid_emb.shape, uiu_emb.shape, uiciu_emb.shape, uibiu_emb.shape, tgt_pred.shape, y.shape, tgt_mae.shape)
        #self.state = torch.cat([iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb,
        #                        tgt_pred, tgt_mae, uiu_pred, uiu_mae, uiciu_pred,
        #                        uiciu_mae, uibiu_pred, uibiu_mae], 1)   # [128, 58]
        self.state = torch.cat([iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb,
                                tgt_pred, uiu_pred, uiciu_pred,
                                uibiu_pred], 1)   # [128, 54]
        return self.state

    def get_emb(self, X, y, action):
        _, _, uiu_emb, uiciu_emb, uibiu_emb = self.model(X, stage=self.stage)
        return uiu_emb, uiciu_emb, uibiu_emb

    def step(self, X, y, action):
        reward, loss, output = self.reward(X, y, action)
        return reward, loss, output

    def reward(self, X, y, action):
        iid_emb, _, uiu_emb, uiciu_emb, uibiu_emb = self.model(X, stage=self.stage)
        src_map_seq_emb = torch.cat([uiu_emb.unsqueeze(1), uiciu_emb.unsqueeze(1), uibiu_emb.unsqueeze(1)], 1)  # [128, 3, 10])
        agg_uid_emb = torch.bmm(action.unsqueeze(1), src_map_seq_emb)  # [128, 1, 10]
        emb = torch.cat([agg_uid_emb, iid_emb.unsqueeze(1)], 1)     # [128, 2, 10]
        output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)  # [128]
        criterion = torch.nn.MSELoss()
        loss = criterion(output, y.squeeze().float())
        if self.reward_name == 'r1':
            return -1.0 * loss, loss, output
        elif self.reward_name == 'r2':
            # reward = infer()
            return 0, loss, output
        else:
            raise ValueError("Unknown reward name")

    def criterion(self, x, y):
        return torch.abs(x - y)
