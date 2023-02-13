import glob
import logging
import os
import os.path as osp
import shutil
import time
import unittest
import warnings
from pathlib import Path
from typing import List

import docker
import yaml
from easydict import EasyDict as edict

from .utils import append_binds, run_cmd, run_docker_cmd


def todict(cfg):
    if isinstance(cfg, dict):
        return {key: todict(value) for key, value in cfg.items()}
    else:
        return cfg


class Verifier(unittest.TestCase):

    def __init__(self, cfg: edict):
        super().__init__()
        warnings.simplefilter('ignore', ResourceWarning)

        self.supported_tasks = ['training', 'mining', 'infer']
        self.supported_algorithms = ['detection', 'segmentation', 'classification']
        # docker image config
        self.cfg = cfg
        self.task_id = cfg.get('task_id', str(round(time.time())))
        self.gpu_id = self.cfg.get('gpu_id', '0')
        self.class_names = cfg.class_names

        # os.path.abspath('') = '.'
        in_dir = cfg.get('in_dir', './in')
        assert osp.isdir(in_dir)
        self.host_in_dir = os.path.abspath(in_dir)

        out_dir = cfg.get('out_dir', './out')
        assert out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.host_out_dir = os.path.abspath(out_dir)

        self.pretrain_files = []
        self.pretrain_weights_dir = osp.abspath(cfg.get('pretrain_weights_dir', './pretrain_weights_dir'))

        if osp.isdir(self.pretrain_weights_dir):
            # generate docker image absolute path for pretrained weight files
            # note pretrain_weights_dir will mount to docker container
            copy_files = glob.glob(osp.join(self.pretrain_weights_dir, '*'), recursive=False)
            for f in copy_files:
                if osp.isfile(f):
                    self.pretrain_files.append(osp.join('/in/models', osp.relpath(f, start=self.pretrain_weights_dir)))

        # ymir env config, affect /in/env.yaml
        if cfg.get('env_config'):
            # support for all-in-one.yaml
            self.env_config = todict(cfg.env_config)
        else:
            self.env_config_file = cfg.get('env_config_file', './env.yaml')
            if osp.exists(self.env_config_file):
                with open(self.env_config_file, 'r') as fp:
                    self.env_config = yaml.safe_load(fp)
            else:
                self.env_config = self.get_default_env()

        self.docker_in_dir = self.cfg.env_config.input.root_dir  # '/in'
        self.docker_out_dir = self.cfg.env_config.output.root_dir  # '/out'

        # hyper-parameter config, affect /in/config.yaml
        if cfg.get('param_config'):
            # support for all-in-one.yaml
            self.param_config = todict(cfg.param_config)
        else:
            self.param_config = {}
            test_config_file = cfg.get('param_config_file', './test-config.yaml')
            if osp.exists(test_config_file):
                with open(test_config_file, 'r') as fp:
                    self.param_config = yaml.safe_load(fp)
            else:
                for task in self.supported_tasks:
                    self.param_config[task] = dict()

        # set pretrain_files to /in/config.yaml
        # note this will overwrite custom value
        if self.pretrain_files:
            for task in self.supported_tasks:
                if task == 'training':
                    if 'pretrained_model_params' in self.param_config[task]:
                        warnings.warn(f'overwrite test config {task} pretrained_model_params')
                    self.param_config[task]['pretrained_model_params'] = self.pretrain_files
                else:
                    if 'model_params_path' in self.param_config[task]:
                        warnings.warn(f'overwrite test config {task} model_params_path')
                    self.param_config[task]['model_params_path'] = self.pretrain_files

        # docker client
        self.docker_image = self.cfg.get('docker_image', 'youdaoyzbx/ymir-executor:ymir2.1.0-mmyolo-cu113-tmi')

    def get_host_path(self, docker_file_path: str):
        """
        convert the input/output file path from docker to host
        """
        if Path('/out') in Path(docker_file_path).parents:
            docker_root_dir = '/out'
            host_root_dir = self.host_out_dir
        elif Path('/in') in Path(docker_file_path).parents:
            docker_root_dir = '/in'
            host_root_dir = self.host_in_dir
        else:
            raise Exception(f'unknown docker file path {docker_file_path}')
        host_file_path = osp.join(host_root_dir, osp.relpath(docker_file_path, start=docker_root_dir))

        return host_file_path

    def verify_docker_path(self, docker_path, is_file=True):
        host_path = self.get_host_path(docker_path)

        if is_file:
            return osp.isfile(host_path), host_path
        else:
            return osp.isdir(host_path), host_path

    def verify_exist(self, docker_image_name: str) -> None:
        client = docker.from_env()
        client.images.get(docker_image_name)

    def get_default_env(self) -> dict:
        """
        input:
            annotations_dir: /in/annotations
            assets_dir: /in/assets
            candidate_index_file: /in/candidate-index.tsv
            config_file: /in/config.yaml
            models_dir: /in/models
            root_dir: /in
            training_index_file: /in/train-index.tsv
            val_index_file: /in/val-index.tsv
        output:
            infer_result_file: /out/infer-result.json
            mining_result_file: /out/result.tsv
            models_dir: /out/models
            monitor_file: /out/monitor.txt
            root_dir: /out
            tensorboard_dir: /out/tensorboard
            training_result_file: /out/models/result.yaml
        run_infer: false
        run_mining: false
        run_training: true
        task_id: t00000020000029d077c1662111056
        """
        env = edict()
        env.input = edict()
        env.output = edict()

        env.input.annotations_dir = '/in/annotations'
        env.input.assets_dir = '/in/assets'
        env.input.candidate_index_file = '/in/candidate-index.tsv'
        env.input.config_file = '/in/config.yaml'
        env.input.models_dir = 'in/models'
        env.input.root_dir = '/in'
        env.input.training_index_file = '/in/train-index.tsv'
        env.input.val_index_file = '/in/val-index.tsv'

        env.output.infer_result_file = '/out/infer-result.json'
        env.output.mining_result_file = '/out/result.tsv'
        env.output.models_dir = '/out/models'
        env.output.monitor_file = '/out/monitor.txt'
        env.output.root_dir = '/out'
        env.output.tensorboard_dir = '/out/tensorboard'
        env.output.training_result_file = '/out/models/result.yaml'

        env.run_infer = False
        env.run_mining = False
        env.run_training = True
        env.task_id = 't00000020000029d077c1662111056'

        return todict(env)

    def create_workspace(self, task: str, pretrain_weights_dir: str = '') -> List[str]:
        """create a worksapce for docker

        Parameters
        ----------
        task : str
            training, infer or mining
        pretrain_weights_dir : str, optional
            pretrain weights directory, by default ''

        create a new workspace with task_id

        workspace
        - in
            - assets  # softlink from self.cfg.data_dir + assets
            - models  # softlink from pretrain_weights_dir
            - annotations # softlink from self.cfg.data_dir + anntations
            - config.yaml
            - env.yaml
            - train-index.tsv # copy from self.cfg.data_dir
            - val-index.tsv # copy from self.cfg.data_dir
            - candidate-index.tsv # copy from self.cfg.data_dir
        - out
        """
        # create in, out and subdir
        self.data_dir = self.cfg.get('data_dir', None) or self.cfg.get('in_dir')
        self.work_dir = self.cfg.get('work_dir', None) or self.cfg.get('out_dir')

        self.cfg.in_dir = osp.join(self.work_dir, self.task_id, task, 'in')
        self.cfg.out_dir = osp.join(self.work_dir, self.task_id, task, 'out')
        os.makedirs(self.cfg.in_dir, exist_ok=True)
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        volumes = [f'-v{osp.abspath(self.cfg.in_dir)}:/in:ro', f'-v{osp.abspath(self.cfg.out_dir)}:/out:rw']

        basename_assets_dir = osp.relpath(self.cfg.env_config.input.assets_dir, start=self.docker_in_dir)
        basename_anntations_dir = osp.relpath(self.cfg.env_config.input.annotations_dir, start=self.docker_in_dir)
        basename_models_dir = osp.relpath(self.cfg.env_config.input.models_dir, start=self.docker_in_dir)
        for subdir in [basename_assets_dir, basename_anntations_dir]:
            xxx_dir = osp.join(self.data_dir, subdir)
            os.symlink(osp.abspath(xxx_dir), osp.join(osp.abspath(self.cfg.in_dir), subdir))
            append_binds(volumes, xxx_dir)

        self.cfg.pretrain_weights_dir = pretrain_weights_dir
        if task in ['mining', 'infer']:
            assert osp.isdir(pretrain_weights_dir)

        if pretrain_weights_dir:
            os.symlink(osp.abspath(pretrain_weights_dir), osp.join(osp.abspath(self.cfg.in_dir), basename_models_dir))
            append_binds(volumes, pretrain_weights_dir)

        # generate config.yaml and env.yaml
        in_config_file = osp.join(self.cfg.in_dir, 'config.yaml')
        env_config_file = osp.join(self.cfg.in_dir, 'env.yaml')
        self.generate_hyperparameter_yaml(task, in_config_file)
        self.generate_env_yaml(task, env_config_file)

        # copy train-index.tsv, val-index.tsv, candidate-index.tsv
        if task in ['training']:
            train_index_file = self.cfg.env_config.input.training_index_file
            val_index_file = self.cfg.env_config.input.val_index_file

            host_train_index_file = train_index_file.replace(self.docker_in_dir, self.data_dir)
            host_val_index_file = val_index_file.replace(self.docker_in_dir, self.data_dir)
            shutil.copy(host_train_index_file, self.cfg.in_dir)
            shutil.copy(host_val_index_file, self.cfg.in_dir)
        else:
            candidate_index_file = self.cfg.env_config.input.candidate_index_file
            host_candidate_index_file = candidate_index_file.replace(self.docker_in_dir, self.data_dir)
            shutil.copy(host_candidate_index_file, self.cfg.in_dir)
        return volumes

    def generate_hyperparameter_yaml(self, task: str, in_config_file: str) -> None:
        assert task in ['training', 'infer', 'mining'], f'task is {task}'

        output = run_docker_cmd(self.docker_image, f'cat /img-man/{task}-template.yaml'.split())

        template_config = yaml.safe_load(output)

        # the real gpu id
        real_gpu_id: str = self.gpu_id
        gpu_count: int = len(real_gpu_id.split(','))
        # the fake gpu id
        gpu_id = ','.join([str(i) for i in range(gpu_count)])

        task_id = self.task_id
        if task == 'training':
            task_config = dict(gpu_id=gpu_id,
                               gpu_count=gpu_count,
                               task_id=task_id,
                               class_names=self.class_names,
                               pretrained_model_params=self.pretrain_files)

            in_config = template_config.copy()
            in_config.update(task_config)
        elif task == 'infer':
            task_config = dict(gpu_id=gpu_id,
                               gpu_count=gpu_count,
                               task_id=task_id,
                               class_names=self.class_names,
                               model_params_path=self.pretrain_files)

            in_config = template_config.copy()
            in_config.update(task_config)
        elif task == 'mining':
            task_config = dict(gpu_id=gpu_id,
                               gpu_count=gpu_count,
                               task_id=task_id,
                               class_names=self.class_names,
                               model_params_path=self.pretrain_files)

            in_config = template_config.copy()
            in_config.update(task_config)
        else:
            raise Exception(f'unknown task name {task}')

        ### apply user define config
        if self.param_config[task]:
            in_config.update(self.param_config[task])
            logging.info(f'modify training template config with {self.param_config[task]}')

        with open(in_config_file, 'w') as fp:
            yaml.dump(in_config, fp)

    def generate_env_yaml(self, task: str, env_config_file: str) -> None:
        env_config = self.env_config.copy()
        task_id = self.task_id

        if task == 'training':
            task_config = dict(run_training=True,
                               run_mining=False,
                               run_infer=False,
                               task_id=task_id)
            env_config['input']['candidate_index_file'] = ''
        elif task == 'infer':
            task_config = dict(run_training=False,
                               run_mining=False,
                               run_infer=True,
                               task_id=task_id)
            env_config['input']['training_index_file'] = ''
            env_config['input']['val_index_file'] = ''
        elif task == 'mining':
            task_config = dict(run_training=False,
                               run_mining=True,
                               run_infer=False,
                               task_id=task_id)
            env_config['input']['training_index_file'] = ''
            env_config['input']['val_index_file'] = ''
        else:
            raise Exception(f'unknown task {task}')

        env_config.update(task_config)
        with open(env_config_file, 'w') as fw:
            yaml.dump(env_config, fw)

    def run_task(self, task: str, pretrain_weights_dir: str = ''):
        # create workspace in self.cfg.in_dir
        volumes = self.create_workspace(task, pretrain_weights_dir)

        cmd = 'docker run --rm'.split()
        cmd += volumes
        if self.gpu_id:
            cmd.extend(['--gpus', f"\"device={self.gpu_id}\""])

        cmd.extend(['--ipc', 'host', self.docker_image])
        cmd.extend(['bash', '/usr/bin/start.sh'])

        run_cmd(cmd)
