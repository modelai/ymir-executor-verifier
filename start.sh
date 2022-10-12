#!/bin/bash

# init scripts
set -e

echo "download voc dog dataset"
wget https://github.com/modelai/ymir-executor-fork/releases/download/dataset/voc_dog_debug_sample.zip -O tests/data/voc_dog_debug_sample.zip

cd tests/data && unzip voc_dog_debug_sample.zip
