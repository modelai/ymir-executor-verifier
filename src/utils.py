import os
import subprocess
import warnings
from pprint import pprint
from typing import Dict, List


def print_error(result: Dict):
    error_count = 0
    for key in result:
        if result.get('error'):
            error_count += 1
            pprint({key: result[key]})

    if error_count == 0:
        print('nice, no error found')


def append_binds(cmd: List[str], bind_path: str) -> None:
    if os.path.exists(bind_path):
        if os.path.islink(bind_path):
            actual_bind_path = os.path.abspath(os.readlink(bind_path))
        else:
            actual_bind_path = os.path.abspath(bind_path)

        cmd.append(f"-v{actual_bind_path}:{actual_bind_path}")
    else:
        warnings.warn(f'bind path {bind_path} not exist')


def run_docker_cmd(docker_image_name: str, command: List[str]) -> str:
    """simple run the docker command without mount volumes

    Parameters
    ----------
    docker_image_name : str
        eg: youdaoyzbx/ymir-executor:ymir2.1.0-mmyolo-cu113-tmi
    command : List[str]
        eg: cat /img-man/training-template.yaml

    Returns
    -------
    str
        eg: the training template config
    """
    cmd = 'docker run --rm'.split()
    cmd.append(docker_image_name)
    cmd += command

    return run_cmd(cmd, need_output=True)


def run_cmd(command: List[str], need_output: bool = False) -> str:
    cmd = ' '.join(command)
    print(f'run cmd: {cmd}')
    if not need_output:
        # wait until the command finished, the output will in stdin and stderr
        subprocess.run(command, check=True)
        return ''
    else:
        # run it with 60s timeout, fetch the output
        result = subprocess.check_output(command, timeout=60)
        return result.decode('utf-8')
