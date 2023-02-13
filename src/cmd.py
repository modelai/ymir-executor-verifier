import argparse

import yaml
from easydict import EasyDict as edict

from .pipeline import PipeLine


def get_args():
    parser = argparse.ArgumentParser(prog='ymir executor verifier')

    parser.add_argument('--docker_image',
                        help='docker image name, will overwrite config file if offered',
                        required=False)
    parser.add_argument('--tasks',
                        help='the task to test, will overwrite config file if offered',
                        required=False,
                        choices=['training', 'mining', 'infer', 'tmi', 'ttmi', 'mi', 't', 'm', 'i'])
    parser.add_argument('--config', help='the config file', required=True)
    parser.add_argument('--pretrain_weights_dir', default=None, help='use for mining and infer only')
    parser.add_argument('--gpu_id', nargs='?')

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        cfg = edict(yaml.safe_load(fp))

    if args.tasks:
        if args.tasks in ['t', 'training']:
            tasks = ['training']
        elif args.tasks in ['m', 'mining']:
            tasks = ['mining']
        elif args.tasks in ['i', 'infer']:
            tasks = ['infer']
        elif args.tasks in ['tmi']:
            tasks = ['training', 'mining', 'infer']
        elif args.tasks in ['mi']:
            tasks = ['mining', 'infer']
        elif args.tasks in ['ttmi']:
            tasks = ['training', 'training', 'mining', 'infer']
        else:
            raise Exception(f'unknown task {args.task}')

        cfg.tasks = tasks

    if args.docker_image:
        cfg.docker_image_name = args.docker_image

    if args.pretrain_weights_dir:
        cfg.pretrain_weights_dir = args.pretrain_weights_dir

    v = PipeLine(cfg)
    v.run()


if __name__ == '__main__':
    main()
