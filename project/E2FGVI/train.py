import os
import json
import argparse
from shutil import copyfile

import torch
import torch.multiprocessing as mp

from core.trainer import Trainer
from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='E2FGVI')
parser.add_argument('-c',
                    '--config',
                    default='configs/train_e2fgvi.json',
                    type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch')
        print('using GPU {}-{} for training'.format(int(config['global_rank']),
                                                    int(config['local_rank'])))

    config['save_dir'] = os.path.join(
        config['save_dir'],
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['save_metric_dir'] = os.path.join(
        './scores',
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['save_metric_dir'], exist_ok=True)
        config_path = os.path.join(config['save_dir'],
                                   args.config.split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print('[**] create folder {}'.format(config['save_dir']))

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    mp.set_sharing_strategy('file_system')

    # loading configs
    config = json.load(open(args.config))

    # setting distributed configurations
    config['world_size'] = get_world_size()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False
    print(config['world_size'])
    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1":
        # manually launch distributed processes
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config, ))
    else:
        # multiple processes have been launched by openmpi
        config['local_rank'] = get_local_rank()
        config['global_rank'] = get_global_rank()
        main_worker(-1, config)
