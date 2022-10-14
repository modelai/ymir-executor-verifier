# ymir-executor-checker
check docker image for ymir

## pre-requiremnts (for linux)

- `nvidia-docker`
- `python3`

## How to use

1. download dataset
```
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

use `youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi` as example
```
docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi

echo "youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi" > ymir_docker_images.txt
python3 tools/test_training.py
```

## training
- training-template.yaml
```
# sample hyper-parameter
batch_size_per_gpu: 8
workers_per_gpu: 4
shm_size_per_gpu: 12G
image_size: 640
export_format: ark:raw
```
- monitor process
- tensorboard log
- result file with model weight and map
- load checkpoint
- resume checkpoint

## infer
- infer-template.yaml
- monitor process
- result file
- check bad image

## mining
- check bad image
- mining-template.yaml
- monitor process
- result file


# TODO
- multiple configs test, read from a config file, each option will be List[Any], modify the standard config and test the result.
