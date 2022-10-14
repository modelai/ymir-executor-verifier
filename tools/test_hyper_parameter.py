import os
import unittest
from pprint import pprint

import yaml
from easydict import EasyDict as edict

from src.verifier_detection import VerifierDetection
import shutil

class TestTraining(unittest.TestCase):

    def test_main(self):
        cfg = edict()
        cfg.in_dir = 'tests/data/voc_dog/in'
        root_out_dir = 'tests/data/voc_dog/out'
        cfg.pretrain_weights_dir = 'tests/pretrain_weights_dir'
        cfg.env_config_file = 'tests/configs/env.yaml'
        cfg.param_config_file = 'tests/configs/test-config.yaml'
        cfg.class_names = ['dog']

        docker_image_name = 'youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi'
        hyper_parameter_file = 'tests/configs/yolov5_hyper_parameters.yaml'

        with open(hyper_parameter_file, 'r') as fp:
            hyper_parameters = yaml.safe_load(fp)

        for key, values in hyper_parameters.items():
            for idx, value in enumerate(values):
                for task in ['training']:
                    cfg.out_dir = os.path.join(root_out_dir, f'{key}_{idx}')
                    os.makedirs(cfg.out_dir, exist_ok=True)
                    v = VerifierDetection(cfg)
                    v.test_config[task][key] = value
                    verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
                    pprint(verify_result)




if __name__ == '__main__':
    unittest.main()
