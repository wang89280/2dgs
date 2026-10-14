# 本工程的 Keypt2Subpx 接入说明

来源：Shinjeong Kim、Marc Pollefeys、Daniel Barath，*Learning to Make Keypoints Sub-Pixel Accurate*，ECCV 2024。[论文](https://arxiv.org/abs/2407.11668) · [官方仓库](https://github.com/KimSinjeong/keypt2subpx)。原代码和权重遵循随附的 Apache-2.0 LICENSE。

本目录保存 commit `30598aeb3c7676cb1459248d780562bfc0e604a2` 的原始 `model.py`、`utils.py`、许可证、README、训练数据准备/评估脚本和 ALIKED 权重，均未改动。URL、大小、SHA256 和 Git blob 校验记录在 `SOURCE_MANIFEST.json`。原 README 中的大数据集下载和训练命令不属于本工程运行要求。

独立适配器在 `code/sfm/python/learned_refinement_predictor.py`，使用原 `AttnTuner`、原 ALIKED 权重及原描述子均值操作。不会重新训练，不会自行改变轨迹或执行 BA，也不读取测试文件。这里只保留开发用的可选推理模块，不表示已经启用或证实提升精度。

## 输入约定

`Predictor(device='cuda').predict_patches(...)` 接受成对排列的输入：

| 字段 | 形状与含义 |
|---|---|
| `keypoints0_native`, `keypoints1_native` | `(N,2)`，本工程完整图像坐标，保留前端的 `+0.5` |
| `descriptors0`, `descriptors1` | `(N,128)`，原 ALIKED-N32 单位化描述子 |
| `rawpatch0`, `rawpatch1` | `(N,1,11,11)`，RGB 按 `[0.299,0.587,0.114]` 转灰度，范围 `[0,1]` |
| `scorepatch0`, `scorepatch1` | `(N,1,11,11)`，原检测器 dense Sigmoid score map，不能用稀疏单点分数替代，也不再次归一化 |
| `offset_scale0_native`, `offset_scale1_native` | `(N,2)`，各点原 tile 的 `[(w-1)/w,(h-1)/h]` |

配对两端的描述子先取算术均值，**不再次归一化**。原网络输出乘 `2.5`，随后按各点比例变换回 native 偏移，再加回 native 原坐标。输出含 `offset0_native`、`offset1_native`、`refined0_native`、`refined1_native` 和模型溯源元数据。重复特征在不同图像对中得到的预测应由调用方统一融合，不能直接成为多个不一致的 SfM 观测。

## patch 的准确坐标

官方仓库绑定 GlueFactory commit `4a8283517fb77e7527e278e233df502f8b7a3266`。其中 ALIKED 的 DKD 归一化按 `[w-1,h-1]`，但输出恢复语句为 `wh * (keypoints + 1) / 2.0`，`wh=[w,h]`。因此本工程必须显式转换：

`official_xy = (tile_native_xy - 0.5) * [w/(w-1), h/(h-1)]`

若保有原 normalized ALIKED 输出，优先直接按官方恢复语句计算，避免坐标往返的 float32 舍入。使用原 `model.py` 的 5px 零 padding、整数截断、`utils.py` 的边界 clamp 和 11×11 整数采样。不能用任意中心 crop 或双线性 subpixel patch 冒充官方预处理。原 tile 位置与全图原 feature ID 必须追溯一致。

## 验证和边界

Windows 的 PyTorch `2.6.0+cu124`、RTX 3080 已完成实际权重加载和推理。原 checkpoint 附带 NumPy float64 元数据；适配器只显式允许该数字类型，仍使用 `weights_only=True`，没有退回任意 pickle 执行。其他 PyTorch 版本未验证。

`INFERENCE_VALIDATION.json` 保存合成接口测试：8 对包含图像边界，与官方 `Keypt2Subpx.forward` 的 native 坐标差不超过 `7.63e-6` px；分块 AttnTuner 推理也做过直接等价核对。该测试只证明接线与数值约定，不是定位精度或稀疏重建 benchmark。

**官方 ALIKED refiner 对应 N16，当前工程使用 N32。** 两者描述子同为 128 维并不证明其分布或坐标基一致。当前模块属于预训练网络的跨前端迁移实验，不能称原论文完整流程复现，不保证比经典方法更好，也不保证精度明显提升。项目增加的多视图融合与拒绝策略属于独立工程部分，需要单独评估。
