import json
import os.path as osp
from typing import List

import yaml
from easydict import EasyDict as edict

from .verifier_detection import VerifierDetection


class VerifierSegmentation(VerifierDetection):

    def __init__(self, cfg: edict):
        super().__init__(cfg)
        self.supported_algorithms = ['segmentation']

    def verify_training_result_file(self, training_result_file) -> None:
        with open(training_result_file, 'r') as fp:
            result = yaml.safe_load(fp)

        if self.object_type == 2:
            metric = 'mAP'
        elif self.object_type == 3:
            metric = 'mIoU'
        elif self.object_type == 4:
            metric = 'maskAP'
        else:
            raise Exception(f'unknown object type {self.object_type}')

        if metric in result:
            self.assertTrue(
                isinstance(result[metric], (float, int)) or result[metric].isnumeric(),
                msg=f'{metric} in training result file {training_result_file} is not number: {result[metric]}')

        # no model stage support
        if 'model' in result:
            self.assertTrue(isinstance(result['model'], List),
                            msg=f'model in training result file {training_result_file} is not list: {result["model"]}')
            for f in result['model']:
                self.assertTrue(osp.isfile(osp.join(self.host_out_dir, 'models', f)),
                                msg=(f'file {f} in training result file {training_result_file} is not valid'
                                     f' or relative to {self.host_out_dir}/models'))
                self.assertFalse(osp.isabs(f), msg=f'{f} in training result file is not relative path')

        # with model stage support
        if 'model_stages' in result:
            self.assertTrue(isinstance(result['model_stages'], dict),
                            msg='model_stages in {training_result_file} must be dict')

            for stage_name, stage in result['model_stages'].items():
                for key in ['stage_name', 'files', 'timestamp', metric]:
                    self.assertTrue(key in stage,
                                    msg=f'{key} not in model_stages[{stage_name}] in {training_result_file}')

                self.assertIsInstance(stage['files'],
                                      list,
                                      msg=f'files in model_stages[{stage_name}] in {training_result_file} not list')

                for f in stage['files']:
                    in_stage_dir = osp.isfile(osp.join(self.host_out_dir, 'models', stage_name, f))
                    in_root_dir = osp.isfile(osp.join(self.host_out_dir, 'models', f))
                    self.assertTrue(
                        in_stage_dir or in_root_dir,
                        msg=(f'file {f} in training result file {training_result_file} is not valid'
                             f' or relative to {self.host_out_dir}/models and {self.host_out_dir}/models/{stage_name}'))
                    self.assertFalse(osp.isabs(f), msg=f'{f} in training result file is not relative path')
                    self.assertFalse(osp.islink(f), msg=f'{f} in training result file is a link file')

    def verify_infer_output(self) -> None:
        if self.object_type == 2:
            return super().verify_infer_output()
        elif self.object_type == 3:
            pass

        ymir_env = self.env_config

        # check infer result file
        docker_task_result_file = ymir_env['output']['infer_result_file']
        valid, task_result_file = self.verify_docker_path(docker_task_result_file, is_file=True)
        self.assertTrue(valid, msg=f'cannot find {docker_task_result_file} in docker, {task_result_file} in host')

        with open(task_result_file, 'r') as f:
            results = json.loads(f.read())

        for ann in results['annotations']:
            self.assertTrue('segmentation' in ann)
            self.assertTrue('size' in ann['segmentation'])
            self.assertTrue('counts' in ann['segmentation'])

        # check process monitor file
        docker_monitor_file = ymir_env['output']['monitor_file']
        self.verify_monitor_file(docker_monitor_file=docker_monitor_file)
