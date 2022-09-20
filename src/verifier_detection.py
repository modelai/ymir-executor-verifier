import os.path as osp
import warnings
from glob import glob
from typing import List

import docker
import yaml
from easydict import EasyDict as edict

from .verifier import Verifier


class VerifierDetection(Verifier):
    def __init__(self, cfg: edict):
        super().__init__(cfg)
        self.supported_algorithms = ['detection']

    def verify(self,
               docker_image_name: str = 'youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi',
               tasks: List[str] = ['training', 'infer']) -> dict:

        verify_result = dict()
        for task in tasks:
            task_result = self.verify_task(docker_image_name, task)
            verify_result.update(task_result)

        return verify_result

    def verify_task(self, docker_image_name: str, task: str, detach: bool = False) -> dict:
        assert task in self.supported_tasks, f'task {task} not in supported tasks {self.supported_tasks}'

        command = f'cat /img-man/{task}-template.yaml'

        verify_result = self.run(docker_image_name=docker_image_name, command=command, tag=f'get-{task}-template')

        try:
            template_config = yaml.safe_load(verify_result[f'get-{task}-template']['result'])
            verify_result[f'{task}_template'] = dict(error='', config=template_config)
        except yaml.YAMLError as e:
            verify_result[f'{task}_template'] = dict(error=f'{e}')

        if verify_result[f'{task}_template']['error']:
            return verify_result

        ## generate config.yaml
        self.generate_yaml(template_config=template_config, task=task)

        ## running task
        task_result = self.run(docker_image_name=docker_image_name,
                               command='bash /usr/bin/start.sh',
                               tag=task,
                               detach=detach)
        verify_result.update(task_result)
        return verify_result

    def verify_training_output(self):
        """
        1. save weight file to /out/models
        2. save tensorboard file to /out/tensorboard_dir
        3. monitor process
        """
        task = 'training'
        with open(self.ymir_env_file, 'r') as fp:
            ymir_env = yaml.safe_load(fp)

        docker_training_result_file = ymir_env['output']['training_result_file']
        training_result_file = self.get_host_path(docker_training_result_file)

        verify_result = dict()
        if not osp.exist(training_result_file):
            verify_result[f'{task}-output'] = dict(
                error=f'cannot find {docker_training_result_file} in docker, {training_result_file} in host')
            return verify_result

        docker_model_out_dir = ymir_env['output']['models_dir']
        model_out_dir = self.get_host_path(docker_model_out_dir)
        weight_files = glob.glob(osp.join(model_out_dir, '*'))
        if len(weight_files) == 0:
            verify_result[f'{task}-output'] = dict(
                error=f'no file found for {docker_model_out_dir} in docker, {model_out_dir} in host')
            return verify_result

        recommand_weight_suffix = ('.pt', '.pth', '.weight', '.param', '.weights', '.params', '.pb')
        recommand_weight_files = [f for f in weight_files if f.endswith(recommand_weight_suffix)]

        if len(recommand_weight_files) == 0:
            verify_result[f'{task}-output'] = dict(
                warnings=
                f'no file found for recommand weight suffix {recommand_weight_suffix} for {docker_model_out_dir} in docker, {model_out_dir} in host'
            )

        docker_tensorboard_dir = ymir_env['output']['tensorboard_dir']
        valid, tensorboard_dir = self.verify_docker_path(docker_tensorboard_dir, is_file=False)

        if not valid:
            verify_result[f'{task}-output'] = dict(
                error='no tensorboard directory for {docker_tensorboard_dir} in docker, {tensorboard_dir} in host')

        tensorboard_logs = glob.glob(osp.join(tensorboard_dir, '*'))
        if len(tensorboard_logs) == 0:
            verify_result[f'{task}-output'] = dict(
                error='no tensorboard logs for {docker_tensorboard_dir} in docker, {tensorboard_dir} in host')

        docker_monitor_file = ymir_env['output']['monitor_file']
        valid, monitor_file = self.verify_docker_path(docker_monitor_file, is_file=True)

        if not valid:
            verify_result[f'{task}-output'] = dict(
                error='no monitor file for {docker_monitor_file} in docker, {monitor_file} in host')

        return verify_result
