# 实验运行日志

## 2026-11-05 环境

- 工作目录：`C:\Users\Yukinoshita\Documents\ChatGPT\图像处理2`
- Python：项目环境 3.12.12；系统默认 3.14.4
- 关键库：OpenCV 5.0.0、NumPy 2.5.4、SciPy 1.18.1、scikit-image 0.26.0、torch 2.11.0+cu128
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB
- 可用 C 盘空间：约 70.6 GB
- Git：2.54.0.windows.1

## 调研与复现

1. 先用 GitHub API 查询仓库元数据时触发了匿名速率限制，改用浅克隆和论文合集仓库核对地址。
2. `ZZZH-jun/DocRes`、`sagnikdas/DewarpNet`、`cvlab-stonybrook/Inv3D`、`RoyalVane/DocDiff` 等地址失败；通过 `ZZZHANG-jx/Recommendations-Document-Image-Processing` 确认正确地址是 `ZZZHANG-jx/DocRes` 和 `Royalvice/DocDiff`。
3. Water-Filling 是 C++/OpenCV3 工程，本机没有 `cl`、`g++`、`msbuild`，因此原始 exe 未构建；改为阅读源码并自主实现向量化变体。
4. Doc-Image-Tool 第一次推理失败，原因是 OpenCV `imread` 不能读取包含中文的路径；改用 Unicode 安全读写后成功。
5. UVDoc 官方模型自带在仓库中，使用 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 后在 torch 2.11 下完成真实推理。
6. DocRes 权重从 Hugging Face Space 下载后，运行了 deshadowing、deblurring、appearance 三个官方任务。
7. DocShadow 官方 `SD7K.pth` 下载成功，我写了独立适配器加载权重，不复制官方模型代码到主包。
8. SD7K 数据集官方 OneDrive API 返回 401；Hugging Face 的同名仓库实际是 Stable Diffusion 加速元数据，不能使用。

## 实验迭代

1. 第一版默认 guided + 全部外观增强 + 0.72 边缘保护，在 20 对 Jung 数据上全图 PSNR/SSIM 为 17.049 / 0.795。
2. 消融发现关闭边缘保护和锐化更优；继续扫描 `edge_protection={0,0.25,0.50,0.72}` 与 `sharpen_amount={0,0.12,0.28}`，最优为两者均 0，PSNR/SSIM 为 18.350 / 0.827。
3. 关闭褪色、去噪、反光和锐化后，只做光照修复，guided 估计器达到 18.875 / 0.842。
4. 五种估计器对照显示形态学闭运算最优：18.989 / 0.843 / 0.127。最终默认配置改为 `morphological`，外观增强默认关闭。
5. 最终 benchmark：ours 18.989 / 0.843 / 0.127；DocShadow 17.300 / 0.834 / 0.148；Doc-Image-Tool 15.918 / 0.810 / 0.176；input 12.932 / 0.792 / 0.240。
6. 失败排序显示 `jung_06`、`jung_09`、`jung_10` 仍落后 DocShadow，主要涉及强阴影、复杂图表和密集版面。

## 测试

- `pytest`：16 passed。
- 单图 CPU 端到端：`third_party/Doc-Image-Tool/test/阴影.jpg`，`configs/scenarios/phone.yaml`，输出 `results/examples/real_shadow/`。
- 多页端到端：`data/synthetic/multipage.png`，检测到 2 页，输出 `results/examples/synthetic_multipage/`。
