import os
import os.path as osp
import shutil
import time
import unittest
from pprint import pprint

import yaml
from easydict import EasyDict as edict
from src.verifier_segmentation import VerifierSegmentation


class TestTMI(unittest.TestCase):
    """
    test Training, Mining and Infer
    """

    def test_main(self):
        config_file = 'tests/configs/mmseg.yaml'
        with open(config_file, 'r') as fp:
            cfg = edict(yaml.safe_load(fp))

        docker_image_name = cfg.docker_image

        root_out_dir = cfg.out_dir
        for task in cfg.tasks:
            cfg.out_dir = osp.join(root_out_dir, task)

            # use training model weight for infer and mining
            if task in ['mining', 'infer']:
                pretrain_weights_dir = osp.join(root_out_dir, 'training', 'models')
                des_weights_dir = osp.join(root_out_dir, 'models')
                if osp.exists(des_weights_dir):
                    shutil.rmtree(des_weights_dir)

                shutil.copytree(pretrain_weights_dir, des_weights_dir)
                cfg.pretrain_weights_dir = des_weights_dir

            v = VerifierSegmentation(cfg)
            verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
            pprint(verify_result)


if __name__ == '__main__':
    unittest.main()
