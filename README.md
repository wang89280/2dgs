# 2DGS 前景稳定性修复版

本文件夹独立整理自本地 Git 提交 `85b1775bd49bdb468922c11b3838f8cacb89995f`（2026-10-14 14:43）。保留该提交的修复源码原字节，不混入后续 SfM 轮廓约束或相机精化模块。这里是代码整理包，不是新的训练结果。

## 目录与入口

- `code/2dgs_foreground_fix/`：完整的该次修复 Python 运行源码，含原许可证；训练入口为 `train.py`。
- `tools/test_foreground_fix.py`：该提交的裁剪与损失回归测试。
- `reference/historical_tools/`：原三方法实验的4个调度、评估、进度和报告脚本，原样存档。
- `reference/validation_at_commit/`：提交当时的协议、运行环境、验证记录及70张训练mask预览；不包含正式训练最终结果。
- `reference/original_2dgs/`：4个被修改文件的原版，便于逐文件比较。
- `reference/core_changes.patch`：4个修改文件和1个新增模块的差异。
- `SOURCE_MANIFEST.json`：每个导出文件在原提交中的路径及SHA256。
- `文件清单.md`：所有导出文件的来源映射。

## 实际修改

1. `utils/foreground_stability.py`：前景/背景透明度监督，排除轮廓3px区域；可靠前景中的单位法线一致性约束。
2. `scene/gaussian_model.py`：修复增密时屏幕半径统计提前清零；增密结束后继续清理异常高斯。
3. `train.py`：接入损失与裁剪；保存随机状态、剩余训练视角队列；输出训练诊断。
4. `scene/dataset_readers.py`：划分训练/评估后，仅加载训练mask。
5. `arguments/__init__.py`：新增稳定性开关与权重、调度参数。

原渲染器、CUDA扩展、ALIKED、LightGlue、SfM与BA没有在该提交中修改。这里不包含SfM完整工程。

## 运行条件

使用本机WSL Ubuntu-20.04及现有Python环境 `/home/wk/miniconda3/envs/surfel_splatting/bin/python`，依赖其已安装的Torch、CUDA光栅化扩展和其他依赖。此包不含环境、CUDA扩展源码/二进制、训练照片或模型，不能称为离线自包含安装包。

训练数据须包含 `images/`、`sparse/0/` 的COLMAP相机/图像记录与初始化点云，以及只对应训练视角的 `masks_train/<图像名>.png`。图像与mask尺寸须相同；原实验已预处理到1600px。默认 `--eval` 每8张留一张，因此80张图是70训练/10评估。

先在WSL进入本包的 `code/2dgs_foreground_fix` 目录，再执行以下模板。将路径占位符换为自己的数据目录和一个全新的输出目录：

```bash
/home/wk/miniconda3/envs/surfel_splatting/bin/python train.py \
  -s /absolute/path/to/dataset -m /absolute/path/to/NEW_model_output \
  --eval --resolution 1 --data_device cpu --iterations 30000 \
  --test_iterations 30000 --save_iterations 7000 15000 30000 \
  --autosave_iterations 5000 --port 0 \
  --use_masks --foreground_stability \
  --lambda_alpha_bg 0.1 --lambda_alpha_fg 0.01 --lambda_normal 0.05 \
  --normal_from_iter 15000 --normal_ramp_iters 5000 \
  --max_screen_radius_fraction 0.25
```

历史调度脚本含原工程目录与原WSL实验目录的固定引用，放在reference下仅供查阅，不能直接当成本包的一键入口运行，否则可能访问主工程或因依赖路径缺失而失败。这里没有启动训练或修改正在运行的实验。

## 验证与范围

整理时核对所有导出文件与源提交字节完全一致，并对Python文件做语法解析；没有重新执行GPU训练或回归测试。原提交的 `validation.json` 记载当时回归测试及20步冒烟通过，正式训练尚未完成。不要把后续提交的结果误归入本快照。

本次改动是2DGS训练稳定性工程修复，不是SfM精度提升证明。原许可证位于 `code/2dgs_foreground_fix/LICENSE.md`。
