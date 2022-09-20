import os
from pprint import pprint

import yaml
from easydict import EasyDict as edict

from src.verifier_detection import VerifierDetection


def main():
    cfg = edict()
    cfg.in_dir = 'tests/data/voc_dog/in'
    cfg.out_dir = 'tests/data/voc_dog/out'

    v = VerifierDetection(cfg)
    docker_image_name = 'youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi'

    for task in ['training', 'mining', 'infer']:
        verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
        pprint(verify_result)

        if verify_result[task]['error']:
            break

        break


if __name__ == '__main__':
    main()
