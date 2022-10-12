# ymir-executor-checker
check docker image for ymir


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
