import os
import os.path as osp
import unittest
from pprint import pprint

import yaml
from easydict import EasyDict as edict

from src.verifier_detection import VerifierDetection


class TestTraining(unittest.TestCase):
    """
    cd ymir-executor-fork/live-code-executor
    docker build -t youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-cu111-live -f yolov5.dockerfile .
    """
    def test_live(self):
        cfg = edict()
        cfg.in_dir = 'tests/data/voc_dog/in'
        cfg.out_dir = 'tests/data/voc_dog/out'
        cfg.env_config = osp.join(cfg.in_dir, 'env.yaml')
        cfg.user_config = 'tests/configs/live-code-config.yaml'
        cfg.pretrain_weights_dir = ''
        cfg.class_names = ['dog']

        v = VerifierDetection(cfg)
        docker_image_name = 'youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-cu111-live'

        for task in ['training', 'mining', 'infer']:
            verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
            pprint(verify_result)
            break

    def test_live_with_task_id(self):
        cfg = edict()
        task_id = 't00000020000027b1b061661221529'
        ymir_train_dir = '/data2/wangxujinfeng/ymir/ymir-workplace/sandbox/work_dir/TaskTypeTraining'
        cfg.in_dir = osp.join(ymir_train_dir, task_id, 'sub_task', task_id, 'in')
        cfg.out_dir = osp.join(ymir_train_dir, task_id, 'sub_task', task_id, 'out3')
        cfg.env_config = osp.join(cfg.in_dir, 'env.yaml')
        cfg.user_config = 'tests/configs/live-code-config.yaml'
        cfg.pretrain_weights_dir = ''

        in_config_yaml = osp.join(cfg.in_dir, 'config.yaml')
        with open(in_config_yaml, 'r') as fp:
            in_config = yaml.safe_load(fp)

        cfg.class_names = in_config['class_names']
        v = VerifierDetection(cfg)
        docker_image_name = 'youdaoyzbx/ymir-executor:ymir1.3.0-yolov5-cu111-live'

        for task in ['training', 'mining', 'infer']:
            verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
            pprint(verify_result)
            break


if __name__ == '__main__':
    unittest.main()
