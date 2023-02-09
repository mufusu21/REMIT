import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from recbole.utils import InputType
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss

class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.meta_dim = meta_dim
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.modulator1 = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                              torch.nn.Linear(meta_dim, emb_dim * meta_dim))
        self.modulator2 = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                              torch.nn.Linear(meta_dim, meta_dim * emb_dim * emb_dim))
        self.modulator3 = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * meta_dim))
        self.modulator4 = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim * emb_dim * emb_dim))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))
        self.decoder2 = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(meta_dim, emb_dim * emb_dim))
        self.decoder3 = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(meta_dim, emb_dim * emb_dim))
        self.decoderV2 = torch.nn.Sequential(torch.nn.Linear(emb_dim*2, meta_dim), torch.nn.ReLU(),
                                            #  torch.nn.Linear(meta_dim, meta_dim), torch.nn.ReLU(),
                                             torch.nn.Linear(meta_dim, emb_dim * emb_dim))
        self.decoderV3 = torch.nn.Sequential(torch.nn.Linear(emb_dim*3, meta_dim), torch.nn.ReLU(),
                                             torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def hisFea(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()     # [128, 20]
        event_K = self.event_K(emb_fea)     # [128, 20, 10] -> [128, 20, 1]
        t = event_K - torch.unsqueeze(mask, 2) * 1e8    # [128, 20, 1]
        att = self.event_softmax(t)     # [128, 20, 1]
        his_fea = torch.sum(att * emb_fea, 1)   # [128, 10]. [128, 20, 1] * [128, 20, 10] -> [128, 20, 10]
        return his_fea

    def hisFeaTar(self, q_emb, emb_fea, seq_index):
        srs_atten = torch.matmul(emb_fea, q_emb.transpose(1, 2).contiguous())  # [128, 20, 1]
        mask = (seq_index == 0).float()     # [128, 20]
        t = srs_atten - torch.unsqueeze(mask, 2) * 1e8    # [128, 20, 1]
        att = self.event_softmax(t)    # [128, 20, 1]
        his_fea = torch.sum(att * emb_fea, 1)   # [128, 10]. [128, 20, 1] * [128, 20, 10] -> [128, 20, 10]
        return his_fea

    def forward(self, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)
        output = self.decoder(his_fea)
        return output.squeeze(1)

    def forwardB2(self, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)
        output = self.decoder2(his_fea)
        return output.squeeze(1)

    def forwardB3(self, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)
        output = self.decoder3(his_fea)
        return output.squeeze(1)

    def forwardV1(self, emb):
        output = self.decoder(emb)  # use global-ge directly
        return output.squeeze(1)

    def forwardV2(self, ge_emb, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)
        decoder_input = torch.cat([his_fea, ge_emb], 1)     # [128, 20]
        output = self.decoderV2(decoder_input)
        return output.squeeze(1)

    def forwardV3(self, ge_emb, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)
        last_item = emb_fea[:, -1, :]   # [128, 10]
        decoder_input = torch.cat([his_fea, ge_emb, last_item], 1)     # [128, 30]
        output = self.decoderV3(decoder_input)
        return output.squeeze(1)

    def forwardV4(self, tgt_i_emb, emb_fea, seq_index, bridge=1):
        his_fea = self.hisFeaTar(tgt_i_emb, emb_fea, seq_index)
        if bridge == 1:
            output = self.decoder(his_fea)
        elif bridge == 2:
            output = self.decoder2(his_fea)
        elif bridge == 3:
            output = self.decoder3(his_fea)
        else:
            raise Exception("Invalid bridge!", bridge)
        return output.squeeze(1)

    def forwardV5(self, ge_emb, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index).unsqueeze(1)   # [128, 1, 10]
        map1 = self.modulator1(ge_emb).view(-1, self.emb_dim, self.meta_dim)  # [128, 10] -> [128, 10 * 50]-> [128, 10, 50]
        map2 = self.modulator2(ge_emb).view(-1, self.meta_dim, self.emb_dim * self.emb_dim)  # [128, 10] -> [128, 50 * 10 * 10]-> [128, 50, 10*10]
        hidden1_output = torch.bmm(his_fea, map1)   # [128, 1, 50]
        active_fuc = torch.nn.ReLU()
        hidden2_input = active_fuc(hidden1_output)
        output = torch.bmm(hidden2_input, map2)   # [128, 1, 10*10]
        return output.squeeze(1)

    def forwardV6(self, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)   # [128, 10]
        map1 = self.modulator1(his_fea).view(-1, self.emb_dim, self.meta_dim)  # [128, 10] -> [128, 10 * 50]-> [128, 10, 50]
        map2 = self.modulator2(his_fea).view(-1, self.meta_dim, self.emb_dim * self.emb_dim)  # [128, 10] -> [128, 50 * 10 * 10]-> [128, 50, 10*10]
        his_fea = his_fea.unsqueeze(1)   # [128, 1, 10]
        hidden1_output = torch.bmm(his_fea, map1)   # [128, 1, 50]
        active_fuc = torch.nn.ReLU()
        hidden2_input = active_fuc(hidden1_output)
        output = torch.bmm(hidden2_input, map2)   # [128, 1, 10*10]
        return output.squeeze(1)

    def forwardV7(self, ge_emb, tgt_i_emb, emb_fea, seq_index):
        his_fea = self.hisFeaTar(tgt_i_emb, emb_fea, seq_index)
        last_item = emb_fea[:, -1, :]   # [128, 10]
        decoder_input = torch.cat([his_fea, last_item], 1)     # [128, 30]
        output = self.decoderV2(decoder_input)
        return output.squeeze(1)

    def forwardV8(self, ge_emb, tgt_i_emb, emb_fea, seq_index):
        his_fea = self.hisFeaTar(tgt_i_emb, emb_fea, seq_index)
        o1 = self.decoder(his_fea).view(-1, self.emb_dim, self.emb_dim)  # [128, 10, 10]
        # o2 = self.decoder2(ge_emb).view(-1, self.emb_dim, self.emb_dim)  # [128, 10, 10]
        # last_item = emb_fea[:, -1, :]   # [128, 10]
        # o3 = self.decoder3(last_item).view(-1, self.emb_dim, self.emb_dim)  # [128, 10, 10]
        # output1 = torch.bmm(o1, o2)  # [128, 10, 10]
        # output = torch.bmm(output1, o3)  # [128, 10, 10]
        # output = torch.bmm(o1, o2)  # [128, 10, 10]
        output = o1
        return output.squeeze(1)

    def forwardV8_1(self, ge_emb, emb_fea, seq_index):
        his_fea = self.hisFea(emb_fea, seq_index)
        o1 = self.decoder(his_fea).view(-1, self.emb_dim, self.emb_dim)  # [128, 10, 10]
        o2 = self.decoder2(ge_emb).view(-1, self.emb_dim, self.emb_dim)  # [128, 10, 10]
        last_item = emb_fea[:, -1, :]   # [128, 10]
        o3 = self.decoder3(last_item).view(-1, self.emb_dim, self.emb_dim)  # [128, 10, 10]
        output1 = torch.bmm(o1, o2)  # [128, 10, 10]
        output = torch.bmm(output1, o3)  # [128, 10, 10]
        return output.squeeze(1)


class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)


class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x


class TargetAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, use_cuda):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.use_cuda = use_cuda

    def forward(self, query, key_value):
        srs_atten = torch.matmul(key_value, query.transpose(1, 2).contiguous())  # [batch, num_heads, 1]
        srs_atten = self.event_softmax(srs_atten.squeeze(2))    # [batch, num_heads]
        key_value_reshape = torch.reshape(key_value, (-1, self.emb_dim))    # [batch*num_heads, emb_dim]
        range_index = torch.arange(0, query.shape[0]).cuda() if self.use_cuda else torch.arange(0, query.shape[0])
        gather_index = torch.argmax(srs_atten, dim=1) + range_index * self.num_heads
        gather_index2 = gather_index.unsqueeze(1).expand(query.shape[0], self.emb_dim).long()
        output = torch.gather(key_value_reshape, 0, gather_index2)  # [batch, emb_dim]
        return output

    def forward_avg(self, key_value):
        srs_atten = torch.ones(key_value.shape[0], self.num_heads).cuda()  # [batch, num_heads]
        srs_atten = self.event_softmax(srs_atten)    # [128, 3]
        output = torch.bmm(srs_atten.unsqueeze(1), key_value)  # [128, 1, 10]
        return output.squeeze(1)    # [128, 10]

class LookupEmbeddingPretrain(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, src_u_emb_path, src_i_emb_path):
        super().__init__()
        u_df1 = pd.DataFrame({'uid': np.arange(uid_all)})
        u_df2 = pd.read_csv(src_u_emb_path, names=['uid', 'uiu_emb', 'uiciu_emb', 'uibiu_emb', 'u_emb'], header=None)
        u_df3 = u_df1.merge(u_df2, on='uid', how='left')
        u_df3 = u_df3.fillna(value=','.join([str(0.0) for _ in range(emb_dim)]))

        u_uiu_weight = torch.FloatTensor(u_df3.uiu_emb.str.split(',', expand=True).values.astype(np.float32))
        u_uiciu_weight = torch.FloatTensor(u_df3.uiciu_emb.str.split(',', expand=True).values.astype(np.float32))
        u_uibiu_weight = torch.FloatTensor(u_df3.uibiu_emb.str.split(',', expand=True).values.astype(np.float32))
        u_pre_weight = torch.FloatTensor(u_df3.u_emb.str.split(',', expand=True).values.astype(np.float32))
        self.uiu_embedding = torch.nn.Embedding.from_pretrained(u_uiu_weight)
        self.uiciu_embedding = torch.nn.Embedding.from_pretrained(u_uiciu_weight)
        self.uibiu_embedding = torch.nn.Embedding.from_pretrained(u_uibiu_weight)
        self.uid_embedding = torch.nn.Embedding.from_pretrained(u_pre_weight)

        i_df1 = pd.DataFrame({'iid': np.arange(iid_all + 1)})
        i_df2 = pd.read_csv(src_i_emb_path, names=['iid', 'iui_emb', 'ici_emb', 'ibi_emb', 'i_emb'], header=None)
        i_df3 = i_df1.merge(i_df2, on='iid', how='left')
        i_df3 = i_df3.fillna(value=','.join([str(0.0) for _ in range(emb_dim)]))

        i_iui_weight = torch.FloatTensor(i_df3.iui_emb.str.split(',', expand=True).values.astype(np.float32))
        i_ici_weight = torch.FloatTensor(i_df3.ici_emb.str.split(',', expand=True).values.astype(np.float32))
        i_ibi_weight = torch.FloatTensor(i_df3.ibi_emb.str.split(',', expand=True).values.astype(np.float32))
        i_pre_weight = torch.FloatTensor(i_df3.i_emb.str.split(',', expand=True).values.astype(np.float32))
        self.iui_embedding = torch.nn.Embedding.from_pretrained(i_iui_weight)
        self.ici_embedding = torch.nn.Embedding.from_pretrained(i_ici_weight)
        self.ibi_embedding = torch.nn.Embedding.from_pretrained(i_ibi_weight)
        self.iid_embedding = torch.nn.Embedding.from_pretrained(i_pre_weight)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


class SrcMLP(torch.nn.Module):
    def __init__(self, emb_dim, emb_num):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim*emb_num, emb_dim*2), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim*2, emb_dim))

    def forward(self, emb_input):
        output = self.decoder(emb_input)    # [128, 1, 30]) -> [128, 1, 10])
        return output

class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0, use_cuda, src_u_emb_path, src_i_emb_path, mlp_hiddens_size=[64,64], tgt_items=None, tgt_interacted_items=None):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.src_pretrain_model = LookupEmbeddingPretrain(uid_all, iid_all, emb_dim, src_u_emb_path, src_i_emb_path)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.meta_net2 = MetaNet(emb_dim, meta_dim_0)
        self.meta_net3 = MetaNet(emb_dim, meta_dim_0)
        self.mlp = SrcMLP(emb_dim, 3)   # three src_embs.
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.num_heads = 3
        self.iid_all = iid_all
        # self.gate_net = GateNet(emb_dim, self.num_heads, use_cuda)
        self.target_attention = TargetAttention(emb_dim, self.num_heads, use_cuda)
        # self.multihead_attn = torch.nn.MultiheadAttention(emb_dim, 1)
        self.mlp_hiddens_size = mlp_hiddens_size

        self.sscdr_mapping = MLPLayers(layers=[self.emb_dim] + self.mlp_hiddens_size + [self.emb_dim],
                                            activation='tanh', dropout=0, bn=False)
        # self.tgt_items = list(range(iid_all + 1))
        # self.tgt_interacted_items = tgt_interacted_items

    def sample(self, ids, tgt_items, interacted_history):
        ids = np.array(ids)
        pos_samples = [list() for i in range(self.uid_all)] 
        neg_samples = [list() for i in range(self.uid_all)]
        for u in tqdm.tqdm(range(self.uid_all)):
            if u in interacted_history and len(interacted_history[u]) > 0:
                pos_samples[u] = np.random.choice(np.array(interacted_history[u]), 20)
            else:
                pos_samples[u] = [0]
        return pos_samples

    @staticmethod
    def embedding_normalize(embeddings):
        emb_length = torch.sum(embeddings**2, dim=1, keepdim=True)
        ones = torch.ones_like(emb_length)
        norm = torch.where(emb_length > 1, emb_length, ones)
        return embeddings / norm

    def forward(self, x, samples=None, stage=None):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['ptup_scatter']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            # uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            emb = torch.cat([uid_emb, iid_emb], 1)
            # output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return emb[:, 0, :]
        elif stage in ['tgt_scatter']:
            emb = self.tgt_model.forward(x)
            return emb[:, 0, :]

        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['env_v6']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1])    # [128, 10])
            uid_emb = self.tgt_model.uid_embedding(x[:, 1])    # [128, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1).squeeze(1)   # [128, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2).squeeze(1)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3).squeeze(1)
            return iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb
        elif stage in ['train_meta_v6']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward_avg(src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output

        elif stage in ['train_sscdr']:
            src_emb = self.src_pretrain_model.uid_embedding(x[:, 0])
            src_usr_emb = self.sscdr_mapping(src_emb)
            tgt_usr_emb = self.tgt_model.uid_embedding(x[:, 0])
            # sample on gpu
            history,tgt_items = samples
            # pos_samples, neg_samples = samples
            batch = x[:, 0].cpu().numpy()
            pos_items = np.zeros_like(batch)
            neg_items = np.zeros_like(batch)
            for index,i in enumerate(batch):
                i = int(i)
                pos_items[index] = np.random.choice(history[i], 1)[0] if i in history else 0
                neg_items[index] = np.random.choice(tgt_items, 1)[0]
            pos_items, neg_items = torch.from_numpy(pos_items).cuda(), torch.from_numpy(neg_items).cuda()
            pos_item_emb, neg_item_emb = self.sscdr_mapping(self.tgt_model.iid_embedding(pos_items)), self.sscdr_mapping(self.tgt_model.iid_embedding(neg_items))
            norm_src_usr_emb, norm_pos_item_emb, norm_neg_item_emb = MFBasedModel.embedding_normalize(src_usr_emb), MFBasedModel.embedding_normalize(pos_item_emb), MFBasedModel.embedding_normalize(neg_item_emb)
            return src_usr_emb, tgt_usr_emb, norm_src_usr_emb, norm_pos_item_emb, norm_neg_item_emb
            
        elif stage in ['test_sscdr']:
            uid_emb = self.sscdr_mapping(self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x

        elif stage in ['train_meta_v5_8_1', 'test_meta_v5_8_1']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping = self.meta_net.forwardV8_1(uid_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_8', 'test_meta_v5_8']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        
        elif stage in ['train_meta_v5_8_contrast', 'test_meta_v5_8_contrast']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            # attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            # uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            uid_emb = self.target_attention.forward_avg(src_map_seq_emb).unsqueeze(1)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output

        elif stage in ['train_meta_v5_7_1', 'test_meta_v5_7_1']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV7(uid_emb_src, uiu_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping2 = self.meta_net.forwardV7(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping3 = self.meta_net.forwardV7(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_7', 'test_meta_v5_7']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            mapping1 = self.meta_net.forwardV4(uiu_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping2 = self.meta_net.forwardV4(uiciu_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping3 = self.meta_net.forwardV4(uibiu_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_6', 'test_meta_v5_6']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            mapping = self.meta_net.forwardV6(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_5', 'test_meta_v5_5']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping = self.meta_net.forwardV5(uid_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_4', 'test_meta_v5_4']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            mapping = self.meta_net.forwardV4(iid_emb, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_3', 'test_meta_v5_3']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping = self.meta_net.forwardV3(uid_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_2', 'test_meta_v5_2']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping = self.meta_net.forwardV2(uid_emb_src, ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v5_1', 'test_meta_v5_1']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            # ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            # mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping = self.meta_net.forwardV1(uid_emb_src).view(-1, self.emb_dim, self.emb_dim)

            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v3_5_2', 'test_meta_v3_5_2']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping1 = self.meta_net.forwardV4(uiu_emb_src, ufea, x[:, 2:], 1).view(-1, self.emb_dim, self.emb_dim)
            mapping2 = self.meta_net.forwardV4(uiciu_emb_src, ufea, x[:, 2:], 2).view(-1, self.emb_dim, self.emb_dim)
            mapping3 = self.meta_net.forwardV4(uibiu_emb_src, ufea, x[:, 2:], 3).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v3_5_1', 'test_meta_v3_5_1']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping2 = self.meta_net.forwardB2(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping3 = self.meta_net.forwardB3(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v3_5', 'test_meta_v3_5']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping2 = self.meta_net2.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping3 = self.meta_net3.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v3', 'test_meta_v3']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping2 = self.meta_net2.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            mapping3 = self.meta_net3.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)
            src_emb_input = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 2)   # [128, 1, 30])
            uid_emb = self.mlp.forward(src_emb_input)   # mlp   # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v2_5', 'test_meta_v2_5']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward(iid_emb, src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v2', 'test_meta_v2']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)
            src_emb_input = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 2)   # [128, 1, 30])
            uid_emb = self.mlp.forward(src_emb_input)   # mlp   # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v2_contrast', 'test_meta_v2_contrast']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))
            # uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)

            # uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            uiu_emb = torch.bmm(uiu_emb_src, mapping)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping)
           
            src_emb_input = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)   # [128, 3, 10])
            uid_emb = self.target_attention.forward_avg(src_emb_input).unsqueeze(1) # [128, 1, 10]
            # uid_emb = self.mlp.forward(src_emb_input)   # mlp   # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta_v1', 'test_meta_v1']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            # uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta', 'test_meta']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage == 'train_map_v1':
            src_emb = self.src_pretrain_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map_v1':
            uid_emb = self.mapping.forward(self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage == 'train_map':
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x


class GMFBasedModel(torch.nn.Module):   
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim, use_cuda, src_u_emb_path, src_i_emb_path):
        # def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.src_pretrain_model = LookupEmbeddingPretrain(uid_all, iid_all, emb_dim, src_u_emb_path, src_i_emb_path)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.num_heads = 3
        # self.gate_net = GateNet(emb_dim, self.num_heads, use_cuda)
        self.target_attention = TargetAttention(emb_dim, self.num_heads, use_cuda)

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['env_v6']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1])    # [128, 10])
            uid_emb = self.tgt_model.embedding.uid_embedding(x[:, 1])    # [128, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1).squeeze(1)   # [128, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2).squeeze(1)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3).squeeze(1)
            return iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb
        elif stage in ['train_meta_v6']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward_avg(src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        # elif stage in ['train_meta_v1', 'test_meta_v1']:
        #     iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
        #     uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
        #     ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
        #     mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
        #     uid_emb = torch.bmm(uid_emb_src, mapping)   # linear op, src_emb * map_matrix
        #     emb = torch.cat([uid_emb, iid_emb], 1)
        #     output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        #     return output
        elif stage in ['test_meta_v1', 'train_meta_v1']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return output.squeeze(1)

        elif stage == 'train_map_v1':
            src_emb = self.src_pretrain_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map_v1':
            uid_emb = self.mapping.forward(self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1)


class DNNBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim, use_cuda, src_u_emb_path, src_i_emb_path):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.src_pretrain_model = LookupEmbeddingPretrain(uid_all, iid_all, emb_dim, src_u_emb_path, src_i_emb_path)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.num_heads = 3
        # self.gate_net = GateNet(emb_dim, self.num_heads, use_cuda)
        self.target_attention = TargetAttention(emb_dim, self.num_heads, use_cuda)

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        
        elif stage in ['env_v6']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1])    # [128, 10])
            uid_emb = self.tgt_model.embedding.uid_embedding(x[:, 1])    # [128, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1).squeeze(1)   # [128, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2).squeeze(1)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3).squeeze(1)
            return iid_emb, uid_emb, uiu_emb, uiciu_emb, uibiu_emb
        elif stage in ['train_meta_v6']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))    # [128, 1, 10])
            uiu_emb_src = self.src_pretrain_model.uiu_embedding(x[:, 0].unsqueeze(1))   # [128, 1, 10])
            uiciu_emb_src = self.src_pretrain_model.uiciu_embedding(x[:, 0].unsqueeze(1))
            uibiu_emb_src = self.src_pretrain_model.uibiu_embedding(x[:, 0].unsqueeze(1))

            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])  # [128, 20, 10])
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0])   # [128, 10])
            mapping1 = self.meta_net.forwardV8(uid_emb_src, uiu_emb_src, ufea, x[:, 2:])
            mapping2 = self.meta_net.forwardV8(uid_emb_src, uiciu_emb_src, ufea, x[:, 2:])
            mapping3 = self.meta_net.forwardV8(uid_emb_src, uibiu_emb_src, ufea, x[:, 2:])

            uiu_emb = torch.bmm(uiu_emb_src, mapping1)   # [128, 1, 10])
            uiciu_emb = torch.bmm(uiciu_emb_src, mapping2)
            uibiu_emb = torch.bmm(uibiu_emb_src, mapping3)

            src_map_seq_emb = torch.cat([uiu_emb, uiciu_emb, uibiu_emb], 1)  # [128, 3, 10])
            attn_output = self.target_attention.forward_avg(src_map_seq_emb)   # [128, 10]
            uid_emb = attn_output.unsqueeze(1)    # [128, 1, 10])
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output

        elif stage in ['test_meta_v1', 'train_meta_v1']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_pretrain_model.uid_embedding(x[:, 0].unsqueeze(1))
            # uid_emb_src = self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            ufea = self.src_pretrain_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return output

        elif stage == 'train_map_v1':
            # src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.src_pretrain_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map_v1':
            uid_emb = self.mapping.forward(self.src_pretrain_model.uid_embedding(x.unsqueeze(1)).squeeze())
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return x