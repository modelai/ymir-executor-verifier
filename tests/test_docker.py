import os.path as osp
import unittest

import docker


class TestDocker(unittest.TestCase):

    def __init__(self, methodName: str = ...):  # type: ignore
        super().__init__(methodName)

        self.ymir_in_dir = osp.abspath('data/voc_dog/in')
        self.ymir_out_dir = osp.abspath('data/voc_dog/out')
        self.ymir_model_dir = osp.abspath('data/configs')
        self.client = docker.from_env()

        self.docker_image_name = 'ubuntu:18.04'

    def test_multiple_mount(self):
        target_image = self.client.images.get(self.docker_image_name)

        command = 'ls /models'
        volumes = [f'{self.ymir_in_dir}:/in:ro', f'{self.ymir_out_dir}:/out:rw', f'{self.ymir_model_dir}:/models:ro']

        for detach in [True, False]:
            if detach:
                container = self.client.containers.run(
                    image=target_image,
                    command=command,
                    runtime='nvidia',
                    auto_remove=True,
                    volumes=volumes,
                    environment=['YMIR_VERSION=1.1.0'],  # support for ymir1.1.0/1.2.0/1.3.0/2.0.0
                    shm_size='64G',
                    detach=detach)

                print('use follow command to view docker logs')
                print(f'docker logs -f {container.short_id}')
                # container.start()
                stream = container.logs(stream=True, follow=True)
                for line in stream:
                    print(line.decode('utf-8'), end='')
                print('\n')
                container.wait()
            else:
                print('this task may take long time, view `docker ps` and `docker logs -f xxx` for process')
                run_result = self.client.containers.run(
                    image=target_image,
                    command=command,
                    runtime='nvidia',
                    auto_remove=True,
                    volumes=volumes,
                    environment=['YMIR_VERSION=1.1.0'],  # support for ymir1.1.0/1.2.0/1.3.0/2.0.0
                    shm_size='64G',
                    detach=detach,
                    stderr=True,
                    stdout=True)
                print(run_result.decode('utf-8'))

        self.assertTrue(True)
