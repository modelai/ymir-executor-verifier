import glob
import json
import os.path as osp
import warnings
from typing import List

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

        verify_result = self.docker_run(docker_image_name=docker_image_name,
                                        command=command,
                                        tag=f'get-{task}-template')

        try:
            template_config = yaml.safe_load(verify_result[f'get-{task}-template']['result'])
            verify_result[f'{task}_template'] = dict(error='', config=template_config)
        except yaml.YAMLError as e:
            verify_result[f'{task}_template'] = dict(error=f'{e}')
        except Exception as e:
            print(verify_result)
            raise e

        if verify_result[f'{task}_template']['error']:
            return verify_result

        ## generate config.yaml
        self.generate_yaml(template_config=template_config, task=task)

        ## running task
        task_result = self.docker_run(docker_image_name=docker_image_name,
                                      command='bash /usr/bin/start.sh',
                                      tag=task,
                                      detach=detach)
        verify_result.update(task_result)

        # if task finished without error, check the output
        if verify_result[task]['error'] == '':
            if task == 'training':
                self.verify_training_output()
            elif task == 'infer':
                self.verify_infer_output()
            elif task == 'mining':
                self.verify_mining_output()
            else:
                raise Exception(f'unknown task {task}')

        return verify_result

    def verify_training_output(self) -> None:
        """
        0. save training_result_file
        1. save weight file to /out/models
        2. save tensorboard file to /out/tensorboard_dir
        3. monitor process
        """
        ymir_env = self.env_config

        # check training result file
        docker_task_result_file = ymir_env['output']['training_result_file']
        valid, task_result_file = self.verify_docker_path(docker_task_result_file, is_file=True)
        self.assertTrue(valid, msg=f'cannot find {docker_task_result_file} in docker, {task_result_file} in host')

        self.verify_training_result_file(task_result_file)

        # check model save directory and model weight file
        docker_model_out_dir: str = ymir_env['output']['models_dir']
        valid, model_out_dir = self.verify_docker_path(docker_model_out_dir, is_file=False)
        self.assertTrue(valid,
                        msg=f'model directory {docker_model_out_dir} is not valid in docker, {model_out_dir} in host')

        weight_files = glob.glob(osp.join(model_out_dir, '*'))
        self.assertGreater(len(weight_files),
                           0,
                           msg=f'no file found for {docker_model_out_dir} in docker, {model_out_dir} in host')

        if len(weight_files) > 0:
            recommand_weight_suffix = ('.pt', '.pth', '.weight', '.param', '.weights', '.params', '.pb')
            recommand_weight_files = [f for f in weight_files if f.endswith(recommand_weight_suffix)]

            if len(recommand_weight_files) == 0:
                msg = f'no file found with suffix {recommand_weight_suffix} for {docker_model_out_dir} in docker'
                warnings.warn(msg)

        # check tensorboard directory
        docker_tensorboard_dir = ymir_env['output']['tensorboard_dir']
        valid, tensorboard_dir = self.verify_docker_path(docker_tensorboard_dir, is_file=False)

        self.assertTrue(
            valid, msg=f'no tensorboard directory for {docker_tensorboard_dir} in docker, {tensorboard_dir} in host')

        tensorboard_logs = glob.glob(osp.join(tensorboard_dir, '*'))
        self.assertGreater(len(tensorboard_logs),
                           0,
                           msg=f'no tensorboard logs for {docker_tensorboard_dir} in docker, {tensorboard_dir} in host')

        # check process monitor file
        docker_monitor_file = ymir_env['output']['monitor_file']
        self.verify_monitor_file(docker_monitor_file=docker_monitor_file)

    def verify_training_result_file(self, training_result_file) -> None:
        """ check the content of training result file
        1. check map
        2. check saved file
        """
        with open(training_result_file, 'r') as fp:
            result = yaml.safe_load(fp)

        if 'map' in result:
            self.assertTrue(isinstance(result['map'], (float, int)) or result['map'].isnumeric(),
                            msg=f'map in training result file {training_result_file} is not number: {result["map"]}')

        if 'model' in result:
            self.assertTrue(isinstance(result['model'], List),
                            msg=f'model in training result file {training_result_file} is not list: {result["model"]}')
            for f in result['model']:
                self.assertTrue(
                    osp.isfile(osp.join(self.ymir_out_dir, 'models', f)),
                    msg=
                    f'file {f} in training result file {training_result_file} is not valid or relative to {self.ymir_out_dir}/models'
                )
                self.assertFalse(osp.isabs(f), msg=f'{f} in training result file is not relative path')

        if 'model_stages' in result:
            self.assertTrue(isinstance(result['model_stages'], dict),
                            msg='model_stages in {training_result_file} must be dict')

            for stage_name, stage in result['model_stages'].items():
                for key in ['stage_name', 'files', 'timestamp', 'mAP']:
                    self.assertTrue(key in stage,
                                    msg=f'{key} not in model_stages[{stage_name}] in {training_result_file}')

                self.assertIsInstance(stage['files'],
                                      list,
                                      msg=f'files in model_stages[{stage_name}] in {training_result_file} not list')

                for f in stage['files']:
                    self.assertTrue(
                        osp.isfile(osp.join(self.ymir_out_dir, 'models', f)),
                        msg=
                        f'file {f} in training result file {training_result_file} is not valid or relative to {self.ymir_out_dir}/models'
                    )
                    self.assertFalse(osp.isabs(f), msg=f'{f} in training result file is not relative path')

    def verify_monitor_file(self, docker_monitor_file: str) -> None:
        """ check the format of process monitor file
        docker_monitor_file: the monitor file path in docker
        """
        valid, monitor_file = self.verify_docker_path(docker_monitor_file, is_file=True)
        self.assertTrue(valid, msg=f'no monitor file for {docker_monitor_file} in docker, {monitor_file} in host')

    def verify_infer_output(self) -> None:
        """
        1. verify infer output result (json format)
        2. check monitor file
        """
        ymir_env = self.env_config

        # check infer result file
        docker_task_result_file = ymir_env['output']['infer_result_file']
        valid, task_result_file = self.verify_docker_path(docker_task_result_file, is_file=True)
        self.assertTrue(valid, msg=f'cannot find {docker_task_result_file} in docker, {task_result_file} in host')

        if not valid:
            return None

        with open(task_result_file, 'r') as f:
            results = json.loads(f.read())

        max_boxes = 50
        self.assertIn('detection', results, msg='unexpected infer result format')
        if 'detection' in results:
            names_annotations_dict = results['detection']

            docker_candidate_index_file = ymir_env['input']['candidate_index_file']
            candidate_index_file = self.get_host_path(docker_candidate_index_file)

            with open(candidate_index_file, 'r') as fp:
                basename_list = [osp.basename(f.strip()) for f in fp.readlines()]

            infer_basename_list = []
            for basename, annotations_dict in names_annotations_dict.items():
                infer_basename_list.append(basename)
                if 'annotations' in annotations_dict and isinstance(annotations_dict['annotations'], list):
                    annotations_dict['annotations'].sort(key=(lambda x: x['score']), reverse=True)
                    annotations_dict['annotations'] = annotations_dict['annotations'][:max_boxes]

            n1 = len(basename_list)
            n2 = len(infer_basename_list)
            self.assertEqual(n1, n2, msg=f'candidate basename list size {n1} != infer basename list size {n2}')

            self.assertEqual(set(basename_list),
                             set(infer_basename_list),
                             msg='candidate basename set != infer basename set')

        # check process monitor file
        docker_monitor_file = ymir_env['output']['monitor_file']
        self.verify_monitor_file(docker_monitor_file=docker_monitor_file)

    def verify_mining_output(self) -> None:
        """
        1. verify mining output result
        2. check monitor file
        """
        ymir_env = self.env_config

        # check infer result file
        docker_task_result_file = ymir_env['output']['mining_result_file']
        valid, task_result_file = self.verify_docker_path(docker_task_result_file, is_file=True)
        self.assertTrue(valid, msg=f'cannot find {docker_task_result_file} in docker, {task_result_file} in host')

        if not valid:
            return None

        with open(task_result_file, 'r') as fp:
            lines = fp.readlines()

        mining_image_list = []
        for line in lines:
            img_path, score_str = line.strip().split()
            score = float(score_str)
            mining_image_list.append(img_path)

        docker_candidate_index_file = ymir_env['input']['candidate_index_file']
        candidate_index_file = self.get_host_path(docker_candidate_index_file)

        with open(candidate_index_file, 'r') as fp:
            lines = fp.readlines()

        self.assertEqual(len(lines), len(mining_image_list), msg='mining result image number != candidate image number')
        # check process monitor file
        docker_monitor_file = ymir_env['output']['monitor_file']
        self.verify_monitor_file(docker_monitor_file=docker_monitor_file)
