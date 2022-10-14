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
        config_file = 'tests/configs/all-in-one.yaml'
        with open(config_file, 'r') as fp:
            cfg = edict(yaml.safe_load(fp))

        docker_image_name = cfg.docker_image

        root_out_dir = cfg.out_dir
        for task in ['training', 'mining', 'infer']:
            cfg.out_dir = osp.join(root_out_dir, task)

            # use training model weight for infer and mining
            if task in ['mining', 'infer']:
                cfg.pretrain_weights_dir = osp.join(root_out_dir, 'training', 'models')

            v = VerifierDetection(cfg)
            verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
            pprint(verify_result)


if __name__ == '__main__':
    unittest.main()
