# 快速开始

## 安装

- 参考ymir安装过程，安装好 `nvidia-docker`

- 安装好 `python3` 及 `wget`, `unzip`
```
sudo apt install -y python3 wget unzip
```

- 安装 `ymir-executor-verifier` 并下载准备好数据

```
git clone https://github.com/modelai/ymir-executor-verifier.git
cd ymir-executor-verifier
pip install -e .
bash start.sh
```

## 使用

- 准备测试镜像, 这里以 yolov5 镜像为例

```
docker pull youdaoyzbx/ymir-executor:ymir2.1.0-yolov5-v7.0-cu111-tmi
```

- 修改配置文件 `tests/configs/all-in-one.yaml`

- 进行测试

```
# 测试训练
python tools/test_training.py

# 测试训练/推理与挖掘
python tools/test_tmi.py

# 测试分割镜像
python tools/test_segmentation.py
```
