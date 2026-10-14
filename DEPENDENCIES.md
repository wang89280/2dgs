# 依赖规格

以下内容来自原代码快照的依赖声明；安装前按实际平台选择。

## code/sfm/requirements-base.txt

```text
numpy==2.2.6
scipy==1.15.3
Pillow==12.2.0
opencv-python==4.13.0.92
pycolmap==4.0.4

```

## code/sfm/requirements-cpu.txt

```text
-r requirements-base.txt
onnxruntime==1.22.0

```

## code/sfm/requirements-multiview.txt

```text
# Install a CUDA-compatible torch/torchvision pair for the local GPU separately.
# Versions below match the adapter validation environment.
omegaconf==2.3.0
loguru==0.7.3
einops
kornia
Pillow
numpy

```

## code/sfm/requirements.txt

```text
-r requirements-base.txt
onnxruntime-gpu==1.22.0

```
