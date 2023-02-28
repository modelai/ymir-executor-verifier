# 配置文件说明

以 `tests/configs/all-in-one.yaml` 为例，它提供以下信息：

- docker_image: 待测试的镜像地址

- data_dir: 数据集根目录，需要符合 [ymir平台的数据集格式](https://ymir-executor-fork.readthedocs.io/zh/latest/overview/dataset-format/)

    - 其中通过 `start.sh` 下载的数据集 voc_dog 为 `det-ark:raw` 格式, eg_seg 为 `seg-coco:raw` 格式。

    - 测试工具会将data_dir下的 `assets` 与 `annotations` 目录软链接到镜像中的 `/in` 目录下

- work_dir: 测试结果输出根目录，会基于时间生成新的子目录并软链接到镜像中的 `/out` 目录下

- pretrain_weights_dir: 预训练模型存放目录，测试工具会将该目录下的文件名自动配置到镜像的 `/in/config.yaml` 文件中，并将该目录软链接到 `/in/models`。

- class_names: 数据集的类别信息

- tasks: 待测试的任务列表， 特别地，对于 `tasks = ['training', 'training']`， 第二个训练任务将利用第一个任务的模型输出结果，测试加载预训练模型。

- env_config: ymir环境信息，将输出到 `/in/env.yaml`

- param_config: 超参数配置信息，将覆盖镜像对应的默认参数
