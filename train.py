import argparse
import importlib
from utils import *

MODEL_DIR = None
DATA_DIR = '/data/incremental_learning/datasets'
PROJECT = 'base'


def get_command_line_parser():
    parsers = argparse.ArgumentParser()

    # about dataset and network
    parsers.add_argument('-project', type=str, default=PROJECT)
    parsers.add_argument('-dataset', type=str, default='cub200',
                         choices=['mini_imagenet', 'cub200', 'cifar100', 'cifar100_1', 'cifar100_2'])
    parsers.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parsers.add_argument('-epochs_base', type=int, default=1)
    parsers.add_argument('-epochs_new', type=int, default=100, help='epochs in update_fc_ft')
    parsers.add_argument('-lr_base', type=float, default=0.1)
    parsers.add_argument('-lr_new', type=float, default=0.1, help='Learning rate to finetune new_fc')
    parsers.add_argument('-schedule', type=str, default='Step', choices=['Step', 'Milestone', 'ExponentialLR'])
    parsers.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parsers.add_argument('-step', type=int, default=40)
    parsers.add_argument('-decay', type=float, default=0.0005)
    parsers.add_argument('-momentum', type=float, default=0.9)
    parsers.add_argument('-gamma', type=float, default=0.1)
    parsers.add_argument('-temperature', type=int, default=16)
    parsers.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parsers.add_argument('-batch_size_base', type=int, default=128)
    parsers.add_argument('-batch_size_new', type=int, default=0,
                         help='set 0 will use all the available training image for new')
    parsers.add_argument('-test_batch_size', type=int, default=100)
    # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parsers.add_argument('-base_mode', type=str, default='ft_cos', choices=['ft_dot', 'ft_cos'])
    # ft_dot means using linear classifier,
    # ft_cos means using cosine classifier,
    # avg_cos means using average data embedding and cosine classifier
    parsers.add_argument('-new_mode', type=str, default='avg_cos', choices=['ft_dot', 'ft_cos', 'avg_cos'])

    # for episode learning PIL
    parsers.add_argument('-train_episode', type=int, default=50)
    parsers.add_argument('-episode_shot', type=int, default=1)
    parsers.add_argument('-episode_way', type=int, default=15)
    parsers.add_argument('-episode_query', type=int, default=15)

    # for cec
    parsers.add_argument('-lrg', type=float, default=0.1)  # lr for graph attention network
    parsers.add_argument('-low_shot', type=int, default=1)
    parsers.add_argument('-low_way', type=int, default=15)

    parsers.add_argument('-start_session', type=int, default=0)
    parsers.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parsers.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parsers.add_argument('-device', default='cuda:0')
    parsers.add_argument('-gpu', default='0,1,2,3')
    parsers.add_argument('-num_workers', type=int, default=8)
    parsers.add_argument('-seed', type=int, default=1)
    parsers.add_argument('-debug', action='store_true')

    return parsers


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % args.project).FSCILTrainer(args)
    trainer.train()
