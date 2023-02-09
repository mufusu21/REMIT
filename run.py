import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
from models import MFBasedModel, GMFBasedModel, DNNBasedModel
from rl_reinforce import REINFORCE
from rl_environment import CDREnvironment
from rl_trainer_infer import Trainer, Infer
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Run():
    def __init__(self,
                 config
                 ):
        self.use_cuda = config['use_cuda']
        self.version = config['version']
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src
        self.algo = config['algo']

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.wd = config['wd']

        self.reward_name = config['reward_name']
        self.rl_lr = config['rl_lr']
        self.env_stage = config['env_stage']
        self.action_space = config['action_space']
        self.as_learn_method = config['as_learn_method']


        self.input_root = self.root + 'ready/_' + str(self.ratio[0]) + '_' + str(self.ratio[1]) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'

        self.src_u_emb_path = self.input_root + '/tgt_CDs_and_Vinyl_src_Movies_and_TV_2_8__src_u_emb.csv'
        self.src_i_emb_path = self.input_root + '/tgt_CDs_and_Vinyl_src_Movies_and_TV_2_8__src_i_emb.csv'

        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,
                        'aug_mae': 10, 'aug_rmse': 10,
                        'emcdr_mae': 10, 'emcdr_rmse': 10,
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10,
                        'rlcdr_mae': 10, 'rlcdr_rmse': 10
                        }

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def record_num(self, path):
        data = pd.read_csv(path, header=None)
        return data.shape[0]

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20,
                                                                 padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter

    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_aug = self.read_aug_data()
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        return data_src, data_tgt, data_meta, data_map, data_aug, data_test

    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim, 
                                 self.use_cuda, self.src_u_emb_path, self.src_i_emb_path)
        elif self.base_model == 'DNN':
            model = DNNBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim, 
                                 self.use_cuda, self.src_u_emb_path, self.src_i_emb_path)
        elif self.base_model == 'GMF':
            model = GMFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim, 
                                 self.use_cuda, self.src_u_emb_path, self.src_i_emb_path)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        # optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=0.01, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in data_loader:
            # for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage=stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        # for X, y in data_loader:
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:
                src_emb, tgt_emb = model(X, stage=stage)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred = model(X, stage=stage)
                loss = criterion(pred, y.squeeze().float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # print('for test... break')
            # break

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse


    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        # for i in range(self.epoch):
        for i in range(10):
            self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        print('=========DataAug========')
        for i in range(self.epoch):
            self.train(data_aug, model, criterion, optimizer, i, stage='train_aug')
            mae, rmse = self.eval_mae(model, data_test, stage='test_aug')
            self.update_results(mae, rmse, 'aug')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def PTUPCDRWithGE(self, model, data_map, data_meta, data_test,
                  criterion, optimizer_map, optimizer_meta):
        print('==========PTUPCDR==========')
        print('stage: ', 'train_meta_{}'.format(self.version), 'test_meta_{}'.format(self.version))
        for i in range(self.epoch):
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta_{}'.format(self.version))
            mae, rmse = self.eval_mae(model, data_test, stage='test_meta_{}'.format(self.version))
            self.update_results(mae, rmse, 'ptupcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))
    
    def EMCDRWithGE(self, model, data_map, data_meta, data_test,
                  criterion, optimizer_map, optimizer_meta):
        print('==========EMCDR==========')
        for i in range(self.epoch):
            self.train(data_map, model, criterion, optimizer_map, i, stage='train_map_v1', mapping=True)
            mae, rmse = self.eval_mae(model, data_test, stage='test_map_v1')
            self.update_results(mae, rmse, 'emcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def CDRWithRL(self, model, data_meta, data_test, criterion, optimizer_meta):
        print('===get the initial param for CDR model by reweighting all aspect-mapped-embs with same weight===')
        print('version: ', self.version)
        for i in range(10):
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta_{}'.format(self.version))

        print('===get the initial param for Aspect Selector model.===')
        env = CDREnvironment(self.reward_name, self.action_space, model, self.emb_dim, self.env_stage)
        agent = REINFORCE(env.state_dim // 2, env.state_dim, env.action_space_n, self.rl_lr)
        data_meta_records = self.record_num(self.meta_path)
        capacity = data_meta_records // self.batchsize_meta
        trainer = Trainer(env, model, agent, data_meta, capacity)
        infer = Infer(env, model, agent, data_test)
        trainer.initial_agent_policy(capacity)    # max episode, todo: equal to capacity.

        print('===iteratively conduct CDR and Aspect Selector in turn===')
        for i in range(self.epoch):
            print('Training Epoch {}:'.format(i + 1))
            print('===in the CDR process, we fix the AS policy param and learn the CDR param===')
            trainer.cdr_with_fixed_as(optimizer_meta)
            print('===in the AS process, we learn from the Replay Buffer===')
            trainer.as_from_replay_buffer(self.as_learn_method)
            mae, rmse = infer.eval_mae()
            self.update_results(mae, rmse, 'rlcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))
    
    def main(self):
        model = self.get_model()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()
        if self.algo == 'cmf':
            self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
            return self.results['aug_mae'], self.results['aug_rmse']
        elif self.algo == 'tgt':
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            return self.results['tgt_mae'], self.results['tgt_rmse']
        elif self.algo == 'emcdr':
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.EMCDRWithGE(model, data_map, data_meta, data_test,
                    criterion, optimizer_map, optimizer_meta)
            return self.results['emcdr_mae'], self.results['emcdr_rmse']
        elif self.algo == 'ptupcdr':
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.PTUPCDRWithGE(model, data_map, data_meta, data_test,
                    criterion, optimizer_map, optimizer_meta)
            return self.results['ptupcdr_mae'], self.results['ptupcdr_rmse']
        elif self.algo == 'remit':
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.CDRWithRL(model, data_meta, data_test, criterion, optimizer_meta)
            return self.results['rlcdr_mae'], self.results['rlcdr_rmse']
        elif self.algo == 'sscdr':
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            optimizer_sscdr = torch.optim.Adam(params=model.sscdr_mapping.parameters(), lr=self.lr, weight_decay=self.wd)
            bst_mae, bst_rmse = self.sscdr_withGE(model, data_tgt, data_test, criterion, optimizer_sscdr)
            return bst_mae, bst_rmse
        else:
            print("Input algorithm wrong, only tgt, cmf, sscdr, ptupcdr, remit are available...")
            exit(1)
        