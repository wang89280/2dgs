# 固定内参稀疏重建模块

从总包根目录运行 `run.ps1 -SparseOnly`。此模块使用同一最终代码，参数由根目录 `config.json` 和显式命令行覆盖决定。默认开启轨迹恢复；`-TrackRecovery:$false` 关闭图上轨迹重组、合并和观测补回，同时保留 BA 与常规异常点过滤。

执行顺序：`frontend_stage.py` 提取 ALIKED-N32、LightGlue 匹配、pycolmap 几何验证及静止点过滤；`backend_stage.py` 调用初始化、增量 SfM、轨迹重建、轨迹维护；`export_2dgs.py` 导出标准 PINHOLE 数据；`validate.py` 检查导出完整性。

`src/modules/` 是唯一一份共用几何源码，`src/tracks/` 是轨迹处理和可选图像精化 BA。`bin/` 包含四个原阶段与一个可选 `refine_image_observations.exe`，均属于同一套代码。`python/` 是模型读取、统计、过滤和审计工具；`frontend/` 是学习特征接口；`weights/` 是两份 ONNX 权重。

```powershell
# 在本目录编译并安装最终程序，需要 Visual Studio 2022 C++ 工具链
./build.ps1 -Install
```

依赖：GPU 前端使用 `requirements.txt`，CPU 前端使用 `requirements-cpu.txt`。独立 SfM 不调用 Metashape 或 COLMAP mapper，但前端验证和标准模型检查依赖 pycolmap。

图像精化由 `--image-refinement off|localization|weighted|learned` 控制，默认 off。`learned` 另需 CUDA torch 与 onnx，仅支持已验证的 4096 点分块协议，第三方权重和 Apache-2.0 许可证在 `third_party/keypt2subpx/`。全部分支固定内参，不增删观测；原测量与修正测量分开记录。当前实测尚未证明明显涨点，详见总包的图像精化实验报告。

`run.py` 和 `backend_stage.py` 均支持 `--track-recovery` / `--no-track-recovery`。原生 `rebuild_tracks.exe` 与 `maintain_tracks.exe` 也接受这两个开关。仅用于严格消融的 `maintain_tracks.exe --no-track-recovery --ba-rounds N` 可重放 ON 实际执行的 0–3 次有界 BA；正常 OFF 默认执行 3 次。配对实验入口与限制见总包 `docs/轨迹恢复消融.md`。
