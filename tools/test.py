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
        cfg.class_names = ['dog']

        v = VerifierDetection(cfg)
        docker_image_name = 'youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi'

        for task in ['mining', 'infer']:
            verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
            pprint(verify_result)


if __name__ == '__main__':
    unittest.main()
