#!/bin/bash

# init scripts
set -e

echo "download voc dog detection dataset"
mkdir -p tests/data
wget https://github.com/modelai/ymir-executor-fork/releases/download/dataset/voc_dog_debug_sample.zip -O tests/data/voc_dog_debug_sample.zip

echo "unzip voc dog detection dataset"
cd tests/data && unzip voc_dog_debug_sample.zip

echo "download eg100 segmentation dataset"
wget https://github.com/modelai/ymir-executor-verifier/releases/download/dataset-eg100/eg100.zip -O tests/data/eg100.zip

echo "unzip eg100 segmentation dataset"
cd tests/data && unzip eg100.zip

echo "generate candidate-index.tsv for mining and infer"
cd voc_dog/in && cat val-index.tsv | awk '{print $1}' > candidate-index.tsv

echo "to test you docker image training"
echo "run: python3 tools/test_training.py"

echo "\n\n"
echo "to test training/mining/infer"
echo "1. modify ./tests/data/configs/all-in-one.yaml"
echo "2. run: python3 tools/test_tmi.py"
