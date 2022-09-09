from easydict import EasyDict as edict
import docker
import warnings
import os
import os.path as osp
import yaml


class Verifier(object):
    def __init__(self, cfg: edict):
        self.supported_tasks = ['training', 'mining', 'infer']
        self.supported_algorithms = ['detection', 'segmentation', 'classification']
        # docker image config
        self.cfg = cfg
        self.ymir_in_dir = os.path.abspath(cfg.get('in_dir', '/in'))
        self.ymir_out_dir = os.path.abspath(cfg.get('out_dir', '/out'))
        self.ymir_env_file = os.path.abspath(cfg.get('env_file', 'tests/data/env.yaml'))
        # docker client
        self.client = docker.from_env()

    def run(self, docker_image_name: str, command: str) -> dict:
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
            run_result = self.client.containers.run(
                image=target_image,
                command=command,
                runtime='nvidia',
                auto_remove=True,
                volumes=[f'{self.ymir_in_dir}:/in:ro', f'{self.ymir_out_dir}:/out:rw'],
                environment=['YMIR_VERSION=1.1.0'],
                shm_size='64G')
            verify_result['run'] = dict(error='', result=run_result.decode('utf-8'))
        except docker.errors.ContainerError as e:
            verify_result['run'] = dict(error=f'container error {e}')
        except docker.errors.APIError as e:
            verify_result['run'] = dict(error=f'API error {e}')
        except docker.errors.ImageNotFound as e:
            verify_result['run'] = dict(error=f'docker image not found {e}')
        finally:
            if verify_result['run']['error']:
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

    def generate_yaml(self, template_config: dict, task: str):
        in_config_file = osp.join(self.ymir_in_dir, 'config.yaml')
        env_config_file = osp.join(self.ymir_in_dir, 'env.yaml')

        if osp.exists(in_config_file):
            warnings.warn(f'exists {in_config_file}, no needs to generate')

        gpu_id: str = str(self.cfg.get('gpu_id', '0'))
        gpu_count: int = len(gpu_id.split(','))
        task_id: str = 't00000020000020167c11661328921'
        if task == 'training':
            task_config = dict(gpu_id=gpu_id, gpu_count=gpu_count, task_id=task_id, class_names=['dog'])

            in_config = template_config.copy()
            in_config.update(task_config)
        elif task == 'infer':
            task_config = dict(gpu_id=gpu_id, gpu_count=gpu_count, task_id=task_id)

            in_config = template_config.copy()
            in_config.update(task_config)
        elif task == 'mining':
            task_config = dict(gpu_id=gpu_id, gpu_count=gpu_count, task_id=task_id)

            in_config = template_config.copy()
            in_config.update(task_config)

        with open(in_config_file, 'w') as fp:
            yaml.dump(in_config, fp)

        if osp.exists(env_config_file):
            warnings.warn(f'exists {env_config_file}, no needs to generate')

        with open(self.ymir_env_file, 'r') as fr:
            env_config = yaml.safe_load(fr)

        if task == 'training':
            task_config = dict(run_training=True,
                               run_mining=False,
                               run_infer=False,
                               task_id=task_id,
                               training_index_file='/in/train-index.tsv',
                               val_index_file='/in/val-index.tsv',
                               candidate_index_file='')
        elif task == 'infer':
            task_config = dict(run_training=True,
                               run_mining=False,
                               run_infer=True,
                               task_id=task_id,
                               training_index_file='',
                               val_index_file='',
                               candidate_index_file='/in/candidate-index.tsv')
        elif task == 'mining':
            task_config = dict(run_training=False,
                               run_mining=True,
                               run_infer=False,
                               task_id=task_id,
                               training_index_file='',
                               val_index_file='',
                               candidate_index_file='/in/candidate-index.tsv')

        env_config.update(task_config)
        with open(env_config_file, 'w') as fw:
            yaml.dump(env_config, fw)
