import docker
from easydict import EasyDict as edict
from .verifier import Verifier
import warnings
import yaml
from typing import List


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

    def verify_task(self, docker_image_name: str, task: str) -> dict:
        assert task in self.supported_tasks, f'task {task} not in supported tasks {self.supported_tasks}'

        command = f'cat /img-man/{task}-template.yaml'

        verify_result = self.run(docker_image_name=docker_image_name, command=command)

        try:
            template_config = yaml.safe_load(verify_result['run']['result'])
            verify_result[f'{task}_template'] = dict(error='', config=template_config)
        except yaml.YAMLError as e:
            verify_result[f'{task}_template'] = dict(error=f'{e}')

        if verify_result[f'{task}_template']['error']:
            return verify_result

        ## generate config.yaml
        self.generate_yaml(template_config=template_config, task=task)

        ## running task
        return verify_result
