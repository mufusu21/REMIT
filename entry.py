import os
import torch
import numpy as np
import random
import argparse
import json
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default= None)
    parser.add_argument('--ratio', nargs="+", default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cuda', type=int, default=0)
    # parser.add_argument('--version', default='v1')
    # parser.add_argument('--reward_name', default='r1')
    parser.add_argument('--rl_lr', type=float, default=0.01)
    # parser.add_argument('--env_stage', default='env_v6')
    # parser.add_argument('--action_space', type=int, default=3)
    parser.add_argument('--algo', type=str, default='others')
    # parser.add_argument('--as_learn_method', default='ori_reward')
    args = parser.parse_args()

    with open(config_path, 'r') as f:
        config = json.load(f)
        
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
        config['use_cuda'] = args.use_cuda
        config['gpu'] = args.gpu
        # config['version'] = args.version
        # config['reward_name'] = args.reward_name
        config['rl_lr'] = args.rl_lr
        # config['env_stage'] = args.env_stage
        # config['action_space'] = args.action_space
        # config['as_learn_method'] = args.as_learn_method
        config['algo'] = args.algo
        config['seed'] = args.seed
        config['version'] = 'v6' if config['algo'] == 'remit' else 'v1'
    return args, config


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    print(f"number of used gpu: {torch.cuda.device_count()}")
    print(f"cuda is available: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
            DataPreprocessingMid(config['root'], dealing).main()

    if args.process_data_ready:
        for ratio in [[0.2, 0.8]]:
            for task in ['1']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()

    if not args.process_data_mid and not args.process_data_ready:
        if config['seed'] is None:
            print("No seed is provided. Run algorithms on five random seeds(2020, 10, 1000, 900, 500).... ")
            res = []
            for seed in [2020, 10, 1000, 900, 500]:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                res.append(Run(config).main())
            mae = [d[0] for d in res]
            rmse = [d[1] for d in res]
            print(f'Avg results over five runs: \n Avg/mae: {sum(mae)/len(mae)}, Avg/rmse:{sum(rmse)/len(rmse)}')
        else:
            print(f"task:{config['task']}; version:{config['version']}; model:{config['base_model']}; ratio:{config['ratio']}; epoch:{config['epoch']}; lr:{config['lr']}; gpu:{config['gpu']}; seed:{config['seed']}; algo:{config['algo']}")
            Run(config).main()
