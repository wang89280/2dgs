# 改进 SfM 与 2DGS：仅源码和文档

分支：`codex/sfm-surface-init-7146030`。独立于 `codex/foreground-fix-85b1775` 分支。

本分支仅上传源代码、Markdown说明和CMake构建脚本，不上传模型权重、预编译程序、图片、数据集、训练模型、实验结果或缓存。分支从原main创建，提交历史也不引入之前准备的大文件包。

## 核心入口

- `code/sfm/surface_triangulation.py`：最新候选，基于原始观测信息矩阵的轮廓约束三角化与初始化点修正。
- `code/sfm/src/`：SfM原生几何、相机估计与轨迹处理源码。
- `code/sfm/frontend/`：ALIKED + LightGlue前端接口。
- `code/sfm/run.py`：SfM运行入口。
- `tools/prepare_surface_sfm.py`、`export_surface_native.py`：轮廓约束准备与导出。
- `tools/check_surface_sfm.py`：数值检查与消融准备。
- `code/2dgs_foreground_fix/`：前景稳定性修复版2DGS源码。
- `docs/轮廓约束SfM模块_20261014.md`：算法与实验协议说明。
- `DEPENDENCIES.md`：原依赖声明。
- `SOURCE_FILES.md`：文件来源与哈希。

## 运行边界

这是源码包，不是开箱即用的运行环境。需自行准备ALIKED/LightGlue权重、标定与配置、数据、CUDA扩展，并编译SfM原生程序。原运行器仍包含本机路径及config.json约定，配置文件未上传。轮廓约束批处理脚本绑定原2-1实验，依赖未上传的历史模型、mask和坐标缓存，不能直接复跑历史实验；可使用核心数组API适配新数据。

根目录完整流水线还有旧code/2dgs路径约定；修复版2DGS源码位于code/2dgs_foreground_fix。不要把普通run.ps1视为已适配的一键端到端入口。docs中的历史outputs链接所指结果未上传。

最新SfM候选保持前端、原始观测和相机固定，只修改点的位置与初始化筛选。源快照中同观测RMSE由1.544042升至1.575145px；当时完整2DGS对照未结束，不能据此宣称已提升。候选未设默认。第三方许可证随对应源码保留。
