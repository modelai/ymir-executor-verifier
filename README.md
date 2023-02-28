# ymir-executor-verifier
check docker image for ymir

## pre-requiremnts (for linux)

- `nvidia-docker`
- `python3`

## How to use

1. clone code and download dataset
```
git clone https://github.com/modelai/ymir-executor-verifier.git
cd ymir-executor-verifier
pip install -e .
sudo apt install wget unzip
bash start.sh
```

then you will get follow dataset
```
tests/data/voc_dog
├── in
│   ├── annotations # txt annotations files
│   ├── assets # image files
│   ├── train-index.tsv
│   └── val-index.tsv
└── out
```

2. set up python environment
```
export PYTHONPATH=.
pip3 install -r requirements.txt
```

3. run your ymir docker images

use `youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi` as example
```
docker pull youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi

echo "youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi" > ymir_docker_images.txt
python3 tools/test_training.py
```

4. use with your own config

view `tests/configs/all-in-one.yaml` for example

```
ymir-verifier --help
ymir-verifier --config tests/configs/all-in-one.yaml
```

5. debug your docker images

view [docker image debug](https://github.com/modelai/ymir-executor-fork/blob/ymir-dev/docs/docker-image-debug.md) for detail.

```
docker run -it --gpus all --shm-size 128G -v $PWD/tests/data/voc_dog/in:/in -v $PWD/tests/data/voc_dog/out:/out -v $HOME/code:/code youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi bash
```
