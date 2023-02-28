## 测试内容

## 训练检查内容

- /img-man/training-template.yaml
```
# sample hyper-parameter
batch_size_per_gpu: 8
workers_per_gpu: 4
shm_size_per_gpu: 12G
image_size: 640
export_format: ark:raw
ymir_saved_file_patterns: []
```
- monitor process
- tensorboard log
- result file with model weight and evaluation result
- load checkpoint
- resume checkpoint
- custom saved files, support python regular expression

## 推理检查内容

- /img-man/infer-template.yaml
- monitor process
- result file

## 挖掘检查内容

- /img-man/mining-template.yaml
- monitor process
- result file
