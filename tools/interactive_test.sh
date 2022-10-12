#!/bin/bash

docker run -it --rm --gpus all --ipc host -v $PWD/tests/data/voc_dog/in:/in -v $PWD/tests/data/voc_dog/out:/out youdaoyzbx/ymir-executor:ymir1.1.0-yolov5-cu111-tmi bash
