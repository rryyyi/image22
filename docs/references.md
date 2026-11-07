# 参考仓库阅读与运行记录

以下记录只写实际发生的事情。没有运行推理的仓库会明确标出“未运行”，不会把阅读当作复现。

| 仓库 | commit / 来源 | 实际处理 | 观察与取舍 |
|---|---|---|---|
| [Water-Filling](https://github.com/seungjun45/Water-Filling) | `9e1fee1` | 浅克隆；阅读 C++ 实现；本机无 `cl`、`g++`、`msbuild`，原始 exe 未构建 | 原方法把 Cb/Cr 通道当地形面迭代注水，再用 0.85 系数恢复。我据此独立写了向量化 water-filling 估计器，但未复制 C++ 工程。 |
| [Doc-Image-Tool](https://github.com/jiangnanboy/Doc-Image-Tool) | `c7f3731` | 浅克隆；运行 `DocShadowRemoval.removeShadow` 于真实中文阴影样例 | 第一次被 Windows/OpenCV 中文路径读取问题卡住，后来主包统一用 `np.fromfile + cv2.imdecode`。它的局部背景思路有效，但会把部分纸面纹理压平。 |
| [DocShadow-SD7K](https://github.com/CXH-Research/DocShadow-SD7K) | `b9395ed` | 浅克隆；下载官方 `SD7K.pth`；写独立适配器加载官方权重并完成 20 对 Jung 推理 | 官方 SD7K OneDrive API 返回 401，未下载完整数据集。学习模型在部分样本很强，但在 jung_14、jung_15 等样例上出现明显色偏。 |
| [DocRes](https://github.com/ZZZHANG-jx/DocRes) | `d3e3a18` | 浅克隆；从 Hugging Face Space 获取 `docres.pkl` 和 `mbd.pkl`；运行官方 deshadowing、deblurring、appearance 推理 | 统一任务提示设计有启发性。旧 requirements 与本机不匹配，我用 torch 2.11/cu128 加载并记录兼容性风险。 |
| [ShaDocFormer](https://github.com/kilito777/ShaDocFormer) | `0ce9d0f` | 浅克隆；阅读检测/恢复结构 | README 没有给出可直接下载的预训练权重，未运行推理；没有把它的 Transformer 结构复制进主包。 |
| [Shadow-Contrast](https://github.com/yifan613/Shadow-Contrast) | `98a8a3a` | 浅克隆；阅读对比度引导思想 | 预训练权重需要 Google Drive，未运行完整模型；其“不依赖显式掩膜”的思想被简化为局部对比度引导估计器。 |
| [UVDoc](https://github.com/tanguymagne/UVDoc) | `4c9b82b` | 浅克隆；使用仓库内 `best_model.pkl`，对 DocTr++ 自带的 `25.jpg` 完成一次推理 | 能在本机 RTX 5060 + torch 2.11 加载，但需要 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。它适合弯曲页面的后续扩展，不作为阴影主基线。 |
| [DocTr++](https://github.com/fh2019ustc/DocTr-Plus) | `2b12e48` | 浅克隆；使用其样例图验证 UVDoc | 权重需另取，未运行官方推理。它的 unrestricted rectification 思路是后续几何模块方向。 |
| [PaperEdge](https://github.com/cvlab-stonybrook/PaperEdge) | `3647958` | 浅克隆；阅读 DIW 预训练和 demo 说明 | 权重在 Google Drive，未下载；没有作为本阶段 baseline。 |
| [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet) | `ead1d37` | 浅克隆；阅读几何矫正方法 | 未运行推理，只作为 dewarping 参考。 |
| [DocEnTR](https://github.com/dali92002/DocEnTR) | `004159e` | 浅克隆；阅读 DIBCO 训练/测试流程 | 它主要输出二值化，与“保留彩色/纹理”的目标不同，没有纳入主流程。 |
| [DocDiff](https://github.com/Royalvice/DocDiff) | `f422a3a` | 浅克隆；阅读残差扩散设计 | 对水印、印章和退化建模有参考价值，但扩散推理开销较大，未用于默认 CPU 流程。 |
| [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) | `32a8ec2` | 浅克隆；阅读版面检测能力 | 对后续表格/段落保护有用，本阶段页面检测使用轻量 OpenCV 方法。 |
| [DIQA_CNN](https://github.com/rjchern/DIQA_CNN) | `4e03e6d` | 浅克隆；阅读 SOC 质量评估结构 | 未运行训练；帮助确定质量报告需要 OCR 相关可读性而非只看 PSNR。 |
| [pero-quality](https://github.com/DCGM/pero-quality) | `03fa1e9` | 浅克隆；阅读 OCR 置信度热图方法 | 未运行；确认没有 OCR 时只能给出保守启发式可读性分数。 |
| [Recommendations-Document-Image-Processing](https://github.com/ZZZHANG-jx/Recommendations-Document-Image-Processing) | `4342d72` | 浅克隆；检索 DocRes 正确地址和论文线索 | 修正了我最初记错的 `ZZZH-jun/DocRes` 仓库地址。 |

## 数据与权重可获取性

- Water-Filling 仓库包含 20 对 Jung 原图/无阴影图，可直接用于本报告的公开复现。
- SD7K 官方 OneDrive 链接返回 401；OpenXLab 需要账号流程，未完成下载。
- Hugging Face 上 `jsonjiao/sd7k`、`jsonjiao/sd7k_v1` 的数据字段与 SD7K 文档阴影无关。
- DocRes 权重通过 Hugging Face Space `qubvel-hf/documents-restoration` 获取。
