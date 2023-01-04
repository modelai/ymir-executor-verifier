#!/bin/bash

# init scripts
set -e

echo "download voc dog dataset"
mkdir -p tests/data
wget https://github.com/modelai/ymir-executor-fork/releases/download/dataset/voc_dog_debug_sample.zip -O tests/data/voc_dog_debug_sample.zip

echo "unzip voc dog dataset"
cd tests/data && unzip voc_dog_debug_sample.zip

echo "generate candidate-index.tsv for mining and infer"
cd voc_dog/in && cat val-index.tsv | awk '{print $1}' > candidate-index.tsv

echo "to test you docker image training"
echo "youdaoyzbx/ymir-executor:ymir2.0.0-yolov5-cu111-tmi" > ymir_docker_images.txt
echo "1. modify ./ymir_docker_images.txt and tools/test_training.py"
echo "2. run: python3 tools/test_training.py"

echo "\n\n"
echo "to test training/mining/infer"
echo "1. modify ./tests/data/configs/test-config.yaml and tools/test_tmi.py"
echo "2. run: python3 tools/test_tmi.py"
