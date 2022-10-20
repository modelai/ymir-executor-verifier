import os.path as osp
import argparse
import yaml
from easydict import EasyDict as edict

from .verifier_detection import VerifierDetection
from .utils import print_error

def get_args():
    parser = argparse.ArgumentParser(prog='ymir executor verifier')

    parser.add_argument('--docker_image', help='docker image name, will overwrite config file', required=False)
    parser.add_argument('--tasks',
                        help='the task to test, will overwrite config file',
                        required=False,
                        choices=['training', 'mining', 'infer', 'tmi', 't', 'm', 'i'])
    parser.add_argument('--config', help='the config file', required=True)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        cfg = edict(yaml.safe_load(fp))

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
            cfg.pretrain_weights_dir = osp.join(root_out_dir, 'training', 'models')

        v = VerifierDetection(cfg)
        verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
        print_error(verify_result)


if __name__ == '__main__':
    main()
