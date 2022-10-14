import os
import unittest
from pprint import pprint

import yaml
from easydict import EasyDict as edict

from src.verifier_detection import VerifierDetection


class TestTraining(unittest.TestCase):
    def test_main(self):
        cfg = edict()
        cfg.in_dir = 'tests/data/voc_dog/in'
        cfg.out_dir = 'tests/data/voc_dog/out'
        cfg.pretrain_weights_dir = 'tests/pretrain_weights_dir'
        cfg.env_config_file = 'tests/configs/env.yaml'
        cfg.param_config_file = 'tests/configs/test-config.yaml'
        cfg.class_names = ['dog']

        v = VerifierDetection(cfg)
        with open('ymir_docker_images.txt', 'r') as fp:
            lines = fp.readlines()

        for line in lines:
            docker_image_name = line.strip()

            for task in ['training']:
                verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
                pprint(verify_result)


if __name__ == '__main__':
    unittest.main()
