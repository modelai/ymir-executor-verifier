import copy
import os.path as osp
import shutil
import time
from pathlib import Path

import yaml
from easydict import EasyDict as edict

from .utils import run_docker_cmd
from .verifier_detection import VerifierDetection
from .verifier_segmentation import VerifierSegmentation


class PipeLine(object):
    """use a pipeline to run training/infer/mining task
    """

    def __init__(self, cfg: edict):
        """use data_dir and work_dir to generate in_dir and out_dir for multiple tasks

        <data_dir> = cfg.data_dir
        <work_dir> = cfg.work_dir
        <task_id> = str(round(time.time()))
        <task> = training/infer/mining
        in_dir = <work_dir>/task_id/<task>/in which contains [<data_dir>/assets, <data_dir>/annotations, ...]
        out_dir = <work_dir>/task_id/<task>/in
        """
        self.cfg = copy.deepcopy(cfg)
        self.task_id = self.cfg.get('task_id', None) or str(round(time.time()))

        self.data_dir = self.cfg.data_dir
        self.work_dir = self.cfg.work_dir
        self.class_names = self.cfg.class_names
        self.gpu_id = self.cfg.get('gpu_id', '0')
        self.param_config = self.cfg.param_config  # hyper-parameter config
        self.env_config = self.cfg.env_config  # ymir enviroment config
        self.docker_in_dir = self.cfg.env_config.input.root_dir  # '/in'
        self.docker_out_dir = self.cfg.env_config.output.root_dir  # '/out'
        self.docker_image = self.cfg.docker_image
        # use in check data dir for get_host_path
        self.host_in_dir = ''
        self.host_out_dir = ''
        self.check_data_dir()

    def check_data_dir(self):
        """check image directory and annotations directory

        """
        self.host_in_dir = self.data_dir

        assets_dir = self.get_host_path(self.cfg.env_config.input.assets_dir)
        annotations_dir = self.get_host_path(self.cfg.env_config.input.annotations_dir)

        assert osp.isdir(assets_dir)
        assert osp.isdir(annotations_dir)

        for task in self.cfg.tasks:
            if task == 'training':
                training_index_file = self.get_host_path(self.cfg.env_config.input.training_index_file)
                val_index_file = self.get_host_path(self.cfg.env_config.input.val_index_file)
                assert osp.isfile(training_index_file)
                assert osp.isfile(val_index_file)
            elif task in ['mining', 'infer']:
                candidate_index_file = self.get_host_path(self.cfg.env_config.input.candidate_index_file)

                assert osp.isfile(candidate_index_file)

        self.host_in_dir = ''

    def get_host_path(self, docker_file_path: str):
        """
        convert the input/output file path from docker to host
        """
        if Path(self.docker_out_dir) in Path(docker_file_path).parents:
            docker_root_dir = self.docker_out_dir
            host_root_dir = self.host_out_dir
        elif Path(self.docker_in_dir) in Path(docker_file_path).parents:
            docker_root_dir = self.docker_in_dir
            host_root_dir = self.host_in_dir
        else:
            raise Exception(f'unknown docker file path {docker_file_path}')
        host_file_path = osp.join(host_root_dir, osp.relpath(docker_file_path, start=docker_root_dir))

        return host_file_path

    def get_object_type(self) -> int:
        """
        get the object_type in /img-man/manifest.yaml
        object detection: object_type = 2
        semantic segmenation: object_type = 3
        instance_segmantation: object_type = 4
        """
        command = 'cat /img-man/manifest.yaml'
        try:
            output = run_docker_cmd(self.docker_image, command.split())
            manifest = yaml.safe_load(output)
        except:
            return 2

        return manifest['object_type']

    def run(self):
        """
        support tmi and ttmi
        """
        for idx, task in enumerate(self.cfg.tasks):
            object_type = self.get_object_type()

            if idx > 0 and task == 'training':
                self.cfg.in_dir = osp.join(self.work_dir, self.task_id, 'pretrain', 'in')
                self.cfg.out_dir = osp.join(self.work_dir, self.task_id, 'pretrain', 'out')
            else:
                self.cfg.in_dir = osp.join(self.work_dir, self.task_id, task, 'in')
                self.cfg.out_dir = osp.join(self.work_dir, self.task_id, task, 'out')

            if object_type == 2:
                v = VerifierDetection(self.cfg)
            else:
                v = VerifierSegmentation(self.cfg)

            if idx == 0 and task == 'training':
                v.verify_task(docker_image_name=self.docker_image, task=task)
                new_weights_dir = osp.join(self.work_dir, self.task_id, task, 'models')

                # copy weights file to other path to mount in docker
                self.host_out_dir = self.cfg.out_dir
                host_weights_dir = self.get_host_path(self.env_config.output.models_dir)
                shutil.copytree(host_weights_dir, new_weights_dir)

                self.cfg.pretrain_weights_dir = new_weights_dir
            else:
                v.verify_task(docker_image_name=self.docker_image, task=task)
