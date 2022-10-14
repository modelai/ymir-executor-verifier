import os
import argparse
import yaml
from easydict import EasyDict as edict
from .verifier_detection import VerifierDetection


def get_args():
    parser = argparse.ArgumentParser(prog='ymir executor verifier')

    parser.add_argument('--image', help='docker image name, will overwrite config file', required=False)
    parser.add_argument('--task',
                        help='the task to test, will overwrite config file',
                        required=False,
                        choices=['training', 'mining', 'infer'])
    parser.add_argument('--config', help='the config file', required=True)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        test_config = yaml.safe_load(fp)

    v = VerifierDetection(edict(test_config))


if __name__ == '__main__':
    main()
