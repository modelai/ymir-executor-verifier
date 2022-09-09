import os
from src.verifier_detection import VerifierDetection
from easydict import EasyDict as edict
from pprint import pprint
import yaml


def main():
    cfg = edict()
    cfg.in_dir = 'tests/data/voc_dog/in'
    cfg.out_dir = 'tests/data/voc_dog/out'

    v = VerifierDetection(cfg)
    docker_image_name = 'youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi'

    for command in ['cat /img-man/training-template.yaml', 'cat /img-man/mining-template.yaml']:
        result = v.run(docker_image_name=docker_image_name, command=command)
        pprint(result, width=120)
        template_config = yaml.safe_load(result['run']['result'])
        pprint(template_config)


if __name__ == '__main__':
    main()
