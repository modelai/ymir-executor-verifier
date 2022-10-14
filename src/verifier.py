import glob
import logging
import os
import os.path as osp
import shutil
import unittest
import warnings
from pathlib import Path
import docker
import yaml
from easydict import EasyDict as edict


class Verifier(unittest.TestCase):

    def __init__(self, cfg: edict):
        super().__init__()
        warnings.simplefilter('ignore', ResourceWarning)

        self.supported_tasks = ['training', 'mining', 'infer']
        self.supported_algorithms = ['detection', 'segmentation', 'classification']
        # docker image config
        self.cfg = cfg
        self.overwrite = True
        self.class_names = cfg.class_names
        # os.path.abspath('') = '.'
        in_dir = cfg.get('in_dir', './in')
        assert osp.isdir(in_dir)
        self.ymir_in_dir = os.path.abspath(in_dir)

        out_dir = cfg.get('out_dir', './out')
        assert out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ymir_out_dir = os.path.abspath(out_dir)

        self.pretrain_files = []
        pretrain_weights_dir = cfg.get('pretrain_weights_dir', './pretrain_weights_dir')
        if osp.isdir(pretrain_weights_dir):
            # copy pretrained weight files to /in/models in docker
            copy_files = glob.glob(osp.join(pretrain_weights_dir, '*'), recursive=False)
            os.makedirs(osp.join(self.ymir_in_dir, 'models'), exist_ok=True)
            for f in copy_files:
                if osp.isfile(f):
                    shutil.copy(f, osp.join(self.ymir_in_dir, 'models'))
                    self.pretrain_files.append(osp.basename(f))

        # ymir env config, affect /in/env.yaml
        self.ymir_env_file = cfg.get('env_config_file', './env.yaml')
        if osp.exists(self.ymir_env_file):
            with open(self.ymir_env_file, 'r') as fp:
                self.ymir_env = yaml.safe_load(fp)
        else:
            self.ymir_env = self.get_default_env()

        # hyper-parameter config, affect /in/config.yaml
        self.test_config = {}
        test_config_file = cfg.get('param_config_file', './test-config.yaml')
        if osp.exists(test_config_file):
            with open(test_config_file, 'r') as fp:
                self.test_config = yaml.safe_load(fp)
        else:
            for task in self.supported_tasks:
                self.test_config[task] = dict()

        # note this will overwrite custom value
        if self.pretrain_files:
            for task in self.supported_tasks:
                if task == 'training':
                    if 'pretrained_model_params' in self.test_config[task]:
                        warnings.warn(f'overwrite test config {task} pretrained_model_params')
                    self.test_config[task]['pretrained_model_params'] = self.pretrain_files
                else:
                    if 'model_params_path' in self.test_config[task]:
                        warnings.warn(f'overwrite test config {task} model_params_path')
                    self.test_config[task]['model_params_path'] = self.pretrain_files

        # docker client
        self.client = docker.from_env()

    def get_host_path(self, docker_file_path: str):
        """
        convert the input/output file path from docker to host
        """
        if Path('/out') in Path(docker_file_path).parents:
            docker_root_dir = '/out'
            host_root_dir = self.ymir_out_dir
        elif Path('/in') in Path(docker_file_path).parents:
            docker_root_dir = '/in'
            host_root_dir = self.ymir_in_dir
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

    def docker_run(self, docker_image_name: str, command: str, tag: str = 'run', detach=False) -> dict:
        """
        run docker image for target task
        """
        verify_result = self.verify_exist(docker_image_name)
        if verify_result['image_exist']['error']:
            return verify_result

        target_image = self.client.images.get(docker_image_name)

        workspace = target_image.attrs['Config']['WorkingDir']
        if workspace != '/app':
            verify_result['workspace'] = dict(warn=f'docker image workspace is not /app, but {workspace}')

        cmd = target_image.attrs['Config']['Cmd']
        if cmd[-1] != 'bash /usr/bin/start.sh':
            verify_result['cmd'] = dict(warn=f'docker image cmd is not "bash /usr/bin/start.sh" but {cmd}')

        try:
            if detach:
                container = self.client.containers.run(
                    image=target_image,
                    command=command,
                    runtime='nvidia',
                    auto_remove=True,
                    volumes=[f'{self.ymir_in_dir}:/in:ro', f'{self.ymir_out_dir}:/out:rw'],
                    environment=['YMIR_VERSION=1.1.0'],
                    shm_size='64G',
                    detach=detach)

                print('use follow command to view docker logs')
                print(f'docker logs -f {container.short_id}')
                # container.start()
                stream = container.logs(stream=True, follow=True)
                for line in stream:
                    print(line.decode('utf-8'))
                verify_result[tag] = dict(error='', result='', docker=docker_image_name, command=command)
                container.wait()
            else:
                print('this task may take long time, view `docker ps` and `docker logs -f xxx` for process')
                run_result = self.client.containers.run(
                    image=target_image,
                    command=command,
                    runtime='nvidia',
                    auto_remove=True,
                    volumes=[f'{self.ymir_in_dir}:/in:ro', f'{self.ymir_out_dir}:/out:rw'],
                    environment=['YMIR_VERSION=1.1.0'],
                    shm_size='64G',
                    detach=detach,
                    stderr=True,
                    stdout=True)
                verify_result[tag] = dict(error='',
                                          result=run_result.decode('utf-8'),
                                          docker=docker_image_name,
                                          command=command)
        except docker.errors.ContainerError as e:
            verify_result[tag] = dict(error=f'container error {e}')
        except docker.errors.APIError as e:
            verify_result[tag] = dict(error=f'API error {e}')
        except docker.errors.ImageNotFound as e:
            verify_result[tag] = dict(error=f'docker image not found {e}')
        finally:
            if verify_result[tag]['error']:
                return verify_result

        return verify_result

    def verify_exist(self, docker_image_name: str) -> dict:
        try:
            self.client.images.get(docker_image_name)
            return dict(image_exist=dict(error=''))
        except docker.errors.ImageNotFound as e:
            return dict(image_exist=dict(error=f'docker image {docker_image_name} not found {e}'))
        except docker.errors.APIError as e:
            return dict(image_exist=dict(error=f'unknown api error{e}'))

    def generate_yaml(self, template_config: dict, task: str) -> None:
        self.assertIn(task, self.supported_tasks)

        in_config_file = osp.join(self.ymir_in_dir, 'config.yaml')
        env_config_file = osp.join(self.ymir_in_dir, 'env.yaml')

        if osp.exists(in_config_file):
            warnings.warn(f'exists {in_config_file}, no needs to generate')

        gpu_id: str = str(self.cfg.get('gpu_id', '0'))
        gpu_count: int = len(gpu_id.split(','))
        task_id: str = 't00000020000020167c11661328921'
        if task == 'training':
            task_config = dict(gpu_id=gpu_id,
                               gpu_count=gpu_count,
                               task_id=task_id,
                               class_names=self.class_names,
                               pretrained_model_params=[])

            in_config = template_config.copy()
            in_config.update(task_config)
        elif task == 'infer':
            task_config = dict(gpu_id=gpu_id,
                               gpu_count=gpu_count,
                               task_id=task_id,
                               class_names=self.class_names,
                               model_params_path=[])

            in_config = template_config.copy()
            in_config.update(task_config)
        elif task == 'mining':
            task_config = dict(gpu_id=gpu_id,
                               gpu_count=gpu_count,
                               task_id=task_id,
                               class_names=self.class_names,
                               model_params_path=[])

            in_config = template_config.copy()
            in_config.update(task_config)
        else:
            raise Exception(f'unknown task name {task}')

        ### apply user define config
        if self.test_config[task]:
            in_config.update(self.test_config[task])
            logging.info(f'modify training template config with {self.test_config[task]}')

        with open(in_config_file, 'w') as fp:
            yaml.dump(in_config, fp)

        ### generate env.yaml
        if osp.exists(env_config_file):
            if self.overwrite:
                warnings.warn(f'exists {env_config_file}, overwrite them')
            else:
                warnings.warn(f'exists {env_config_file}, skip generate them')
                return None

        env_config = self.ymir_env.copy()

        if task == 'training':
            training_index_file = osp.join(self.ymir_in_dir, 'train-index.tsv')
            if not osp.exists(training_index_file):
                raise Exception(f'{training_index_file} not exist')
            val_index_file = osp.join(self.ymir_in_dir, 'val-index.tsv')
            if not osp.exists(val_index_file):
                raise Exception(f'{val_index_file} not exist')

            task_config = dict(run_training=True,
                               run_mining=False,
                               run_infer=False,
                               task_id=task_id,
                               input=dict(training_index_file='/in/train-index.tsv',
                                          val_index_file='/in/val-index.tsv',
                                          candidate_index_file=''))

        elif task == 'infer':
            mining_index_file = osp.join(self.ymir_in_dir, 'candidate-index.tsv')
            if not osp.exists(mining_index_file):
                raise Exception(f'{mining_index_file} not exist')

            task_config = dict(run_training=False,
                               run_mining=False,
                               run_infer=True,
                               task_id=task_id,
                               input=dict(training_index_file='',
                                          val_index_file='',
                                          candidate_index_file='/in/candidate-index.tsv'))
        elif task == 'mining':
            mining_index_file = osp.join(self.ymir_in_dir, 'candidate-index.tsv')
            if not osp.exists(mining_index_file):
                raise Exception(f'{mining_index_file} not exist')
            task_config = dict(run_training=False,
                               run_mining=True,
                               run_infer=False,
                               task_id=task_id,
                               input=dict(training_index_file='',
                                          val_index_file='',
                                          candidate_index_file='/in/candidate-index.tsv'))
        else:
            raise Exception(f'unknown task {task}')

        env_config.update(task_config)
        with open(env_config_file, 'w') as fw:
            yaml.dump(env_config, fw)

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

        pydict = dict(env)
        pydict['input'] = dict(env.input)
        pydict['output'] = dict(env.output)
        return pydict
