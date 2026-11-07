# docrestore：本地文档图像修复与质量评估

这个仓库围绕真实拍摄/扫描文档的质量退化问题，实现了一个可配置的本地修复流水线。核心目标不是做 OCR，而是把透视变形、阴影、光照不均、反光、噪声、模糊、划痕、褪色、背景杂乱和多页混排等退化中的可恢复部分处理清楚，同时给出质量分数和失败警告。

## 快速开始

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e .
.venv/Scripts/python.exe -m docrestore run "path/to/input.jpg" -o results/runs/example
```

处理目录时，CLI 当前按单文件调用即可；批量实验脚本会自行遍历配对数据。若要保存光照场、反光掩膜等中间结果：

```powershell
.venv/Scripts/python.exe -m docrestore run input.jpg -o results/runs/debug --intermediates
```

只评估一张图：

```powershell
.venv/Scripts/python.exe -m docrestore assess input.jpg -o results/quality/example.json
```

## 流水线

1. 页面理解：检测页面候选、抑制小面积杂物、支持多页拆分；检测失败时回退整图并记录警告。
2. 几何修复：对四边形候选做透视矫正；不明显或不可靠的候选不会强行扭曲。
3. 光照修复：默认形态学闭运算估计纸面背景，另有 guided、contrast-guided、Retinex 和 water-filling 变体。
4. 反射率恢复：亮度增益与低饱和度区域的彩色增益分开，避免印章和彩色笔迹被整体漂移。
5. 画质增强：反光、噪声、模糊、褪色、划痕模块均可开关；默认关闭，避免无退化图像被过度处理。
6. 质量评估：输出背景均匀性、光照变化、边缘能量、Laplacian 方差、噪声、反光比例、可读性分数和警告。

## 主要结果

在 Water-Filling 仓库提供的 20 对 Jung 真实配对文档上，最终自主方法全图 PSNR 为 18.989 dB，SSIM 为 0.843；DocShadow 为 17.300 dB / 0.834，Doc-Image-Tool 为 15.918 dB / 0.810。完整表见 `results/tables/benchmark_summary.csv`，报告见 `docs/experiment_report.md`。

注意：SD7K 官方 OneDrive 链接需要授权，Hugging Face 上同名 `jsonjiao/sd7k` 实际是 Stable Diffusion 加速元数据，不能作为文档阴影数据集使用。因此本仓库没有伪称完成了完整 SD7K benchmark。

## 场景配置

```powershell
.venv/Scripts/python.exe -m docrestore run input.jpg -o results/runs/phone -c configs/scenarios/phone.yaml
.venv/Scripts/python.exe -m docrestore run scan.png -o results/runs/scanner -c configs/scenarios/scanner.yaml
.venv/Scripts/python.exe -m docrestore run contract.jpg -o results/runs/archive -c configs/scenarios/archive.yaml
```

`phone.yaml` 打开反光、去噪和锐化；`scanner.yaml` 关闭透视，保留温和去噪；`archive.yaml` 打开划痕修复并加强边缘保护；`accessibility.yaml` 更偏向低对比文档的可读性。这些预设不是自动模型选择，只是不同风险策略。

## 实验脚本

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe scripts/generate_synthetic_degradations.py
.venv/Scripts/python.exe scripts/run_benchmark.py --save-all --output results/benchmark_final
.venv/Scripts/python.exe scripts/run_estimator_comparison.py
.venv/Scripts/python.exe scripts/run_ablation.py
.venv/Scripts/python.exe scripts/run_param_sensitivity.py
.venv/Scripts/python.exe scripts/analyze_failures.py
```

## 测试

```powershell
.venv/Scripts/python.exe -m pytest
```

当前 16 个测试覆盖 Unicode 读写、五种光照估计器、阴影恢复、参考/区域指标、页面检测、多页拆分、反光保护和大面积反光拒修。

## 边界

- 主流程默认 CPU 可运行；PyTorch 只用于第三方对照模型，不在主包中强依赖。
- 本地文档不上传到在线 OCR 或云服务。
- `third_party/` 和 `data/` 不入库，第三方代码只用于复现对照，主包没有复制其模型架构。
- 未实现的完整能力包括深度学习 dewarping、训练新模型、正式 OCR 精度评测和大规模 SD7K 训练/测试。
