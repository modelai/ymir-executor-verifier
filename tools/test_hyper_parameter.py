import copy
import os
import os.path as osp
import unittest
import warnings

import yaml
from easydict import EasyDict as edict
from src.utils import print_error
from src.verifier_detection import VerifierDetection


def run_task(cfg: edict, task: str):
    docker_image_name = cfg.docker_image
    os.makedirs(cfg.out_dir, exist_ok=True)

    v = VerifierDetection(cfg)
    result_file = v.env_config['output'][f'{task}_result_file'].replace('/out', cfg.out_dir)
    if osp.exists(result_file):
        warnings.warn('result file {result_file} exist, auto remove it')
        os.remove(result_file)
    verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
    print_error(verify_result)


class TestTraining(unittest.TestCase):
    """
    batch test hyper parameters of ymir executor
    """

    def test_main(self):
        # config_file = 'tests/configs/yolov5_hyper_parameters.yaml'
        config_file = 'tests/configs/mmyolo_hyper_parameters.yaml'
        with open(config_file, 'r') as fp:
            cfg = edict(yaml.safe_load(fp))

        docker_image_name = cfg.docker_image
        hyper_parameters = cfg.hyper_parameters
        root_out_dir = cfg.out_dir
        pretrain_weights_dir = None
        for task in cfg.tasks:
            cfg.out_dir = osp.join(root_out_dir, task)

            # use training model weight for infer and mining
            if task in ['mining', 'infer']:
                cfg.pretrain_weights_dir = pretrain_weights_dir or cfg.pretrain_weights_dir

            new_cfg = edict(copy.deepcopy(cfg))
            if task in hyper_parameters:
                for key, values in hyper_parameters[task].items():
                    for idx, value in enumerate(values):
                        new_cfg.out_dir = os.path.join(root_out_dir, task, f'{key}_{idx}')

                        v = VerifierDetection(new_cfg)
                        v.param_config[task][key] = value
                        verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
                        print_error(verify_result)
            else:
                v = VerifierDetection(new_cfg)
                verify_result = v.verify_task(docker_image_name=docker_image_name, task=task, detach=True)
                print_error(verify_result)

            if task in ['training']:
                pretrain_weights_dir = osp.join(new_cfg.out_dir, 'models')


if __name__ == '__main__':
    unittest.main()
