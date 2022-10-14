import os.path as osp
import unittest
from pprint import pprint

import yaml
from easydict import EasyDict as edict

from src.verifier_detection import VerifierDetection


class TestTMI(unittest.TestCase):
    """
    test Training, Mining and Infer
    """
    def test_main(self):
        cfg = edict()
        cfg.in_dir = 'tests/data/voc_dog/in'
        root_out_dir = 'tests/data/voc_dog/out'
        cfg.pretrain_weights_dir = 'tests/pretrain_weights_dir'
        cfg.env_config_file = 'tests/configs/env.yaml'
        cfg.param_config_file = 'tests/configs/test-config.yaml'
        cfg.class_names = ['dog']

        for task in ['training', 'mining', 'infer']:
            if task == 'training':
                continue

            cfg.out_dir = osp.join(root_out_dir, task)

            # use training model weight for infer and mining
            if task in ['mining', 'infer']:
                cfg.pretrain_weights_dir = osp.join(root_out_dir, 'training', 'models')

            v = VerifierDetection(cfg)
            docker_image_name = v.test_config['image']

            verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
            pprint(verify_result)


if __name__ == '__main__':
    unittest.main()
