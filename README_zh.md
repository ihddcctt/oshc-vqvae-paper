# OSHC-VQ-VAE 论文配套仓库

本版提供最终 **MNIST 与 STL-10** 实验的代码、四份模型配置、数值源数据及 K128 AES-GCM 软件回环验证。版本为 **v1.0.0**。公开地址：https://github.com/ihddcctt/oshc-vqvae-paper 。

十条实测光谱记录确定性生成冻结基础码本 B_s；共享可训练残差码本 R 补充量化残差。每个位置传输两个索引，错误接收状态保持索引不变，仅更换 B_s。MNIST 与 STL-10 分别训练网络，共用同一冻结码本库；D=64，K_B=K_R=128。

状态以 10 s 能量 Q₁₀（mJ）标注；算法将 P₁₀=Q₁₀/(10 s) 的 mW 数值写入 `power_mw` 后序列化。两个零能量记录分别对应两次扫描的参考测量。

| 数据集 | 正确状态 pooled MSE | 90 种错误状态 pooled MSE | wrong/correct |
|---|---:|---:|---:|
| MNIST，10,000 张测试图像 | 0.00175440 | 0.03261143 | 18.5884 |
| STL-10，8,000 张测试图像 | 0.00311294 | 0.03371626 | 10.8310 |

软件基线、分量诊断、SSIM/LPIPS 和置信区间见[完整结果](docs/final_results.md)。模型、公式和统计定义集中在[技术复现协议](docs/supporting_information.md)，安装及训练命令见[英文首页](README.md)和[复现说明](docs/reproducibility.md)。

无需任务权重即可核验已有数值：

```bash
python scripts/verify_release.py --require-manifest
python scripts/verify_aes_results.py
python scripts/summarize_results.py
```

本版不提供任务权重。读者可以精确再生种子/冻结码本、复算数值结果并独立训练；生成作者原 checkpoint 的推理图像需要对应权重。每个最终配置报告一次训练。图像数据从官方来源获取，代码许可证暂未设置。见[可用性说明](docs/data_and_code_availability.md)。
