import argparse
import os.path as osp
import time
import warnings

import yaml
from easydict import EasyDict as edict

from .utils import print_error
from .verifier_detection import VerifierDetection


def get_args():
    parser = argparse.ArgumentParser(prog='ymir executor verifier')

    parser.add_argument('--docker_image',
                        help='docker image name, will overwrite config file if offered',
                        required=False)
    parser.add_argument('--tasks',
                        help='the task to test, will overwrite config file if offered',
                        required=False,
                        choices=['training', 'mining', 'infer', 'tmi', 'mi', 't', 'm', 'i'])
    parser.add_argument('--config', help='the config file', required=True)
    parser.add_argument('--project', help='the project name', default=str(int(time.time())))
    parser.add_argument('--reuse', default=False, action='store_true', help='reuse output directory')
    parser.add_argument('--pretrain_weights_dir', default=None, help='use for mining and infer only')

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        cfg = edict(yaml.safe_load(fp))

    if not args.reuse and osp.exists(cfg.out_dir):
        cfg.out_dir = osp.join(cfg.out_dir, args.project)
        warnings.warn(f'change output directory to {cfg.out_dir}')

    v = VerifierDetection(cfg)

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
        else:
            raise Exception(f'unknown task {args.task}')
    else:
        tasks = cfg['tasks']

    if args.docker_image:
        docker_image_name = args.docker_image
    else:
        docker_image_name = cfg.docker_image

    root_out_dir = cfg.out_dir
    for task in tasks:
        cfg.out_dir = osp.join(root_out_dir, task)
        # use training model weight for infer and mining
        if task in ['mining', 'infer']:
            if 'training' in tasks:
                cfg.pretrain_weights_dir = osp.join(root_out_dir, 'training', 'models')
            else:
                assert args.pretrain_weights_dir and osp.exists(args.pretrain_weights_dir)
                cfg.pretrain_weights_dir = args.pretrain_weights_dir

        v = VerifierDetection(cfg)
        verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
        print_error(verify_result)


if __name__ == '__main__':
    main()
