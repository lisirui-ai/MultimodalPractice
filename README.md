<div align="center">

# MultimodalPractice

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20Diffusers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Models-ViT%20%7C%20CLIP%20%7C%20BLIP%20%7C%20SD%20%7C%20Qwen3VL-8A2BE2?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 PyTorch + HuggingFace 的多模态模型实战系列</p>
<p>以 <b>视觉基础模型 → 图文对齐 → 多模态理解与生成 → 扩散模型 → 多模态大模型</b> 为主线，循序渐进覆盖多模态核心技术栈</p>
<p>每个 Notebook 均配有详细中文注释，适合多模态入门与进阶学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览

| #   | Notebook                                                                                         | 核心技术                                                                                         | 亮点                                                               |
| --- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 01  | ViT 模型架构实现（HuggingFace）                                                                          | Vision Transformer · Patch Embedding · Position Embedding · CLS Token · Transformer Encoder | 从零手写 ViT 全部核心模块，可视化 Patch 划分与注意力机制                               |
| 02  | CLIP 模型架构实现（ViT 图像编码器 + Transformer 文本编码器）                                                       | CLIP · 对比学习 · ViT 图像编码器 · Transformer 文本编码器 · L2 归一化 · 余弦相似度                                | 官方 API 推理 + 手动实现 CLIP 完整架构，直观展示图文对齐原理                             |
| 03  | ALBEF 模型架构实现（ViT 图像编码器 + BERT 文本编码器）                                                             | ALBEF · ITC · ITM · MLM · Cross-Attention · Momentum Distillation                            | 完整手写 ITC 对比学习 + Cross-Attention 融合的双阶段图文对齐架构                      |
| 04  | BLIP 模型架构实现（ViT + BERT 编码器 + Causal BERT 解码器）                                                    | BLIP · MED · CapFilt · ITC · ITM · LM · Causal BERT Decoder · HuggingFace                   | 官方 API 实战（图像描述 / 视觉问答 / 特征提取 / 图文匹配）+ 手动实现统一编解码架构               |
| 05  | BLIP-2 模型架构实现（ViT + BERT Q-Former + OPT 解码器）                                                     | BLIP-2 · Q-Former · EVA-ViT-G/14 · OPT · Frozen 预训练 · Hook 机制                              | 官方 API 图像描述 + Hook 捕获 Q-Former 内部张量 + 完整层级结构注释                    |
| 06  | Stable Diffusion 模型架构可视化（PrintModel / TorchInfo / TorchViz）                                      | Stable Diffusion · VAE · CLIP Text Encoder · U-Net · torchinfo · torchviz · Graphviz         | 三种可视化工具对比：打印结构 / torchinfo 参数统计 / torchviz 计算图 SVG 导出           |
| 07  | Stable Diffusion 推理流程手动实现（DDIM 采样）                                                               | DDIM · U-Net · Cross-Attention · 时间步嵌入 · VAE 解码 · 去噪循环                                      | 不依赖 Diffusers Pipeline，从零手写完整 DDIM 采样推理流程                         |
| 08  | Stable Diffusion 文生图实战（HuggingFace Diffusers）                                                    | Stable Diffusion · Diffusers Pipeline · ModelScope · Text-to-Image                          | 使用 ModelScope 下载权重 + Diffusers Pipeline 一键文生图 + matplotlib 展示  |
| 09  | Qwen3-VL 模型架构实现（ViT 视觉编码器 + Qwen3 LLM）                                                          | Qwen3-VL · ViT · Patch Merger · DeepStack · 3D 卷积 · GQA · HuggingFace                       | 官方 API 图文推理 + 逐层打印结构 + 3D 卷积原理详解（图像 / 视频统一 Patch 嵌入）             |

---

## 📁 目录结构

```
MultimodalPractice/
├── 📓 01.VIT_ModelArchitecture_HuggingFace.ipynb
├── 📓 02.CLIP_ModelArchitecture_ViTImageEncoder_TransformerTextEncoder_HuggingFace.ipynb
├── 📓 03.ALBEF_ModelArchitecture_ViTImageEncoder_BERTTextEncoder.ipynb
├── 📓 04.BLIP_ModelArchitecture_ViTImageEncoder_BERTTextEncoder_CausalBERTDecoder_HuggingFace.ipynb
├── 📓 05.BLIP2_ModelArchitecture_ViTImageEncoder_BERTQFormer_OPTDecoder_HuggingFace.ipynb
├── 📓 06.1.StableDiffusion_ModelArchitectureVisualization_PrintModel.ipynb
├── 📓 06.2.StableDiffusion_ModelArchitectureVisualiztion_TorchInfo.ipynb
├── 📓 06.3.StableDiffusion_ModelArchitectureVisualization_TorchViz.ipynb
├── 📓 07.StableDiffusion_InferenceProcess_DDIM_CustomImplementation.ipynb
├── 📓 08.StableDiffusion_Text2Image_HuggingFace.ipynb
├── 📓 09.Qwen3VL_ModelArchitecture_ViTVisionEncoder_Qwen3LLM_HuggingFace.ipynb
│
├── 📂 model_cache/                    # 模型权重缓存目录（gitignored）
│   ├── ViT/                           # [自动生成] ViT 权重缓存（Notebook 01）
│   ├── CLIP/                          # [自动生成] CLIP 权重缓存（Notebook 02）
│   ├── bert/                          # [自动生成] BERT 权重缓存（Notebook 03）
│   ├── BLIP/                          # [自动生成] BLIP 权重缓存（Notebook 04）
│   ├── blip2/                         # [自动生成] BLIP-2 权重缓存（Notebook 05）
│   ├── SD/                            # [自动生成] Stable Diffusion 权重缓存（Notebook 06/07/08）
│   └── Qwen3VL/                       # [自动生成] Qwen3-VL 权重缓存（Notebook 09）
│
├── 📂 torchviz_output/                # torchviz 计算图 SVG/PNG 导出目录（gitignored）
│   ├── vae_encoder.svg / .png         # VAE 编码器计算图
│   ├── vae_decoder.svg                # VAE 解码器计算图
│   ├── clip_text_encoder.svg          # CLIP 文本编码器计算图
│   ├── clip_encoder_layer_0.svg       # CLIP 编码器单层计算图
│   ├── unet_time_embedding.svg        # U-Net 时间步嵌入计算图
│   ├── unet_resnet_block_with_temb.svg
│   ├── unet_basic_transformer_block.svg
│   ├── unet_down_block_last.svg
│   └── unet_up_block0.svg
│
├── 🖼 cat.png                         # 实验用图（Notebook 01/02）（gitignored）
├── 🖼 dog.jpg                         # 实验用图（Notebook 02）（gitignored）
├── 🖼 SD_generated.png                # Stable Diffusion 生成示例图（gitignored）
├── 📄 requirements.txt            # Python 依赖清单（Python 3.14.5）
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

---

## 📚 Notebook 介绍

### 01. ViT 模型架构实现

> `01.VIT_ModelArchitecture_HuggingFace.ipynb`

从零手写 **Vision Transformer（ViT）** 完整架构，涵盖 Patch Embedding、可学习 Position Embedding、CLS Token、Transformer Encoder（多头自注意力 + FFN）及分类头全部模块。同步演示官方 HuggingFace ViT API 推理，并通过可视化验证 Patch 划分效果与注意力机制。


| 章节                  | 内容                                                              |
| ------------------- | --------------------------------------------------------------- |
| 手写 ViT 模型定义与结构验证    | Patch Embedding · Position Embedding · CLS Token · 多头自注意力 · 分类头 |
| 图像 Patch 划分可视化      | 224×224 图像 → 16×16 Patch 网格划分 · 边界框绘制                          |
| 为何使用可学习位置编码？        | 与固定正弦编码对比 · 可学习参数优势分析                                          |
| HuggingFace 官方 API  | 加载 `google/vit-base-patch16-224` · 注意力热力图可视化                    |


> **ViT 核心原理** · 将图像切分为 16×16 像素的固定 Patch，经 Conv2d Patch Embedding 投影后展平为 Token 序列，拼接 CLS Token 并加 Position Embedding，送入标准 Transformer Encoder；CLS Token 的最终隐状态作为全局图像表示，接分类头输出预测。
>
> 参考论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*（Dosovitskiy et al., ICLR 2021）

---

### 02. CLIP 模型架构实现

> `02.CLIP_ModelArchitecture_ViTImageEncoder_TransformerTextEncoder_HuggingFace.ipynb`

以 **对比学习（Contrastive Learning）** 为目标，使用 4 亿图文对预训练：**ViT 图像编码器** 将图像映射至共享嵌入空间，**Transformer 文本编码器** 将文本 Prompt 映射至同一空间，通过相似度匹配实现零样本图像分类、跨模态检索等下游任务。官方 API 推理与手动实现全架构并行展示。


| 章节                  | 内容                                                              |
| ------------------- | --------------------------------------------------------------- |
| 官方 API 推理演示         | 加载 `openai/clip-vit-base-patch32` · 图文相似度计算 · 零样本分类              |
| 前向传播逐步拆解            | 文本编码 → 文本投影 → L2 归一化 · 图像编码 → 图像投影 → L2 归一化 · 温度缩放余弦相似度矩阵      |
| L2 归一化原理说明          | 公式推导 · 单位超球投影意义 · 在 CLIP 中的作用                                   |
| 手动实现简化版 CLIP 架构     | ViT 图像编码器（PatchEmbedding + MultiHeadAttention）· 文本编码器 · 对比学习损失  |


> **CLIP 对比学习损失** · batch 内 N 对图文互为正样本，N²-N 对互为负样本；图像编码器与文本编码器同时优化，使正对相似度最大、负对相似度最小，温度系数 τ 由可学习标量控制。
>
> 参考论文：*Learning Transferable Visual Models From Natural Language Supervision*（Radford et al., OpenAI, 2021）

---

### 03. ALBEF 模型架构实现

> `03.ALBEF_ModelArchitecture_ViTImageEncoder_BERTTextEncoder.ipynb`

以 **先对齐（ITC）后融合（Cross-Attention）** 的思路解决单模态编码器直接拼接导致的噪声融合问题。三阶段训练目标：**ITC**（图文对比学习，拉齐单模态表示空间）→ **ITM**（图文匹配二分类，细粒度对齐）→ **MLM**（Masked Language Modeling，利用视觉上下文补全遮蔽词）。


| 章节                      | 内容                                                                    |
| ----------------------- | --------------------------------------------------------------------- |
| 图像编码器（ImageEncoder）     | ViT-B/16 主干 · `[CLS]` Token 全局表示 · 196 个 Patch Token 作为 Cross-Attention Key/Value |
| 文本编码器（TextEncoder）      | BERT-base 文本编码 · Token ID → 上下文感知特征序列                                 |
| 多头交叉注意力（CrossAttention） | 文本 Token 为 Query · 图像 Patch 为 Key/Value · 实现视觉信息注入文本                   |
| 多模态融合层（MultimodalEncoder）| Pre-LN + 残差连接 · 文本自注意力 → 图像交叉注意力 → FFN 三阶段融合                         |
| ITC / ITM / MLM 损失      | 动量蒸馏队列 · ITM 困难负样本挖掘 · 三目标联合训练                                       |


> **ALBEF 核心创新** · 在多模态融合前先通过 ITC 对比学习将图文嵌入对齐至公共空间（Align），再通过 Cross-Attention 进行细粒度跨模态融合（Fuse），避免直接拼接时噪声干扰。动量编码器产生软标签用于蒸馏，缓解噪声图文对的影响。
>
> 参考论文：*Align before Fuse: Vision and Language Representation Learning with Momentum Distillation*（Li et al., NeurIPS 2021）

---

### 04. BLIP 模型架构实现

> `04.BLIP_ModelArchitecture_ViTImageEncoder_BERTTextEncoder_CausalBERTDecoder_HuggingFace.ipynb`

引入 **MED（Multimodal mixture of Encoder-Decoder）** 统一架构，同一 BERT 权重通过不同注意力掩码切换三种工作模式：单模态文本编码（ITC）、图文跨模态编码（ITM）、因果语言模型生成（LM），实现理解与生成任务统一。**CapFilt** 机制通过 Captioner 生成伪标题、Filter 过滤噪声数据完成自举式数据清洗。


| 章节              | 内容                                                                             |
| --------------- | ------------------------------------------------------------------------------ |
| 官方 API 实战       | 图像描述生成（Image Captioning）· 视觉问答（VQA）· 特征提取 · 图文匹配（ITM）                         |
| 官方结构展示          | `BlipForConditionalGeneration` 逐层打印 · ViT-B/16 视觉编码器 · MED 文本编码/解码器结构注释      |
| 手写 MED 架构       | 三模式切换（单模态编码 / 交叉注意力编码 / 因果解码）· Causal BERT Decoder · Cross-Attention 注入     |
| ITC + ITM + LM  | 三目标联合训练 · 动量编码器 · 困难负样本采样 · 因果 LM 损失                                         |


> **MED 三模式切换** · `text_encoder` 模式下屏蔽交叉注意力，仅做单模态编码（ITC）；`multimodal_encoder` 模式下开启交叉注意力融合图像（ITM）；`text_decoder` 模式下启用因果掩码做自回归生成（LM）。三种模式共享 BERT 参数，通过 `mode` 参数控制。
>
> 参考论文：*BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*（Li et al., ICML 2022）

---

### 05. BLIP-2 模型架构实现

> `05.BLIP2_ModelArchitecture_ViTImageEncoder_BERTQFormer_OPTDecoder_HuggingFace.ipynb`

引入轻量级 **Q-Former（Querying Transformer）** 桥接冻结的视觉编码器（EVA-ViT-G/14，1408 维，39 层）与冻结的大语言模型（OPT），以 32 个可学习 Query Token 从视觉特征中蒸馏语义信息后投影至 LLM 隐藏维度，极大降低视觉-语言对齐的训练代价。


| 章节                     | 内容                                                                            |
| ---------------------- | ----------------------------------------------------------------------------- |
| 官方 API 图像描述生成          | 加载 `Salesforce/blip2-opt-2.7b` · 图像描述生成示例                                    |
| Hook 机制捕获 Q-Former 内部张量 | `register_forward_hook` 钩取第 0 层自注意力输入/输出 · 验证 32 个 Query Token 张量形状          |
| 完整模型结构展示               | `Blip2ForConditionalGeneration` 三大模块逐层打印 · 每层 tensor shape 注释                |
| 手写简化版 Q-Former         | 可学习 Query Token 设计 · 自注意力（Query 间）+ 交叉注意力（Query 查询视觉 Key/Value）· 两阶段预训练策略说明 |


> **Q-Former 两阶段预训练** · 第一阶段仅解冻 Q-Former，以 ITC + ITM + ITG 三目标对齐图文表示；第二阶段将 Q-Former 输出接 LLM，以语言建模目标对齐语言生成能力。视觉编码器与 LLM 全程冻结，训练参数量极少（约 188M）。
>
> 参考论文：*BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*（Li et al., ICML 2023）

---

### 06. Stable Diffusion 模型架构可视化

本系列由三个子 Notebook 构成，分别采用三种工具对 **Stable Diffusion 2.1-base** 的四大核心模块（VAE、CLIP 文本编码器、CLIPTokenizer、U-Net）进行架构可视化，可横向对比三种工具的展示能力与适用场景。

> 💡 **三种工具的区别**：`print(model)` 以纯文本快速浏览层级结构；`torchinfo` 以表格呈现每层参数量与 tensor shape 变化；`torchviz` 以计算图 SVG 精确展示反向传播依赖关系，适合追踪梯度流动。

#### 06.1 打印模型结构

> `06.1.StableDiffusion_ModelArchitectureVisualization_PrintModel.ipynb`

使用 `print(model)` 直接打印 VAE、CLIP Text Encoder、Tokenizer 及 U-Net 四大模块完整结构，每个子模块均标注输入/输出 shape（如 `[B,3,512,512]→[B,4,64,64]`）、关键层参数（卷积核、隐藏维度等）及模块功能说明。

#### 06.2 TorchInfo 可视化

> `06.2.StableDiffusion_ModelArchitectureVisualiztion_TorchInfo.ipynb`

使用 `torchinfo.summary()` 对各模块进行参数统计与 tensor shape 追踪，以表格形式呈现每一层的输入/输出维度变化（空间分辨率 512→64）、可训练参数量、FLOPs 等信息，便于横向对比 DownEncoderBlock / UpDecoderBlock 等同类结构的计算量。模型通过 ModelScope `snapshot_download` 本地缓存加载。

#### 06.3 TorchViz 计算图可视化

> `06.3.StableDiffusion_ModelArchitectureVisualization_TorchViz.ipynb`

使用 `torchviz.make_dot()` 将各模块前向传播的 autograd 计算图渲染为 SVG/PNG 并导出至 `torchviz_output/`，精确展示每个算子节点（`ConvolutionBackward`、`AddmmBackward` 等）、叶子参数节点（绿色）及中间张量节点的依赖关系，可用于梯度流动分析与 bug 排查。


| 工具          | 展示维度         | 图形类型      | 适用场景          |
| ----------- | ------------ | --------- | ------------- |
| `print`     | 模块层级结构       | 纯文本树形     | 快速浏览 / 结构概览   |
| `torchinfo` | 参数量 · shape 变化 | 表格        | 参数统计 / 计算量分析  |
| `torchviz`  | autograd 依赖图 | SVG 有向无环图 | 梯度流分析 / 精确依赖图 |

---

### 07. Stable Diffusion 推理流程手动实现（DDIM 采样）

> `07.StableDiffusion_InferenceProcess_DDIM_CustomImplementation.ipynb`

**不依赖** HuggingFace Diffusers Pipeline，从零手写完整 Stable Diffusion 文生图推理流程，涵盖 CLIP 文本编码、DDIM 时间步调度、U-Net 去噪循环及 VAE 解码全链路，深入理解扩散模型推理原理。


| 章节                      | 内容                                                                            |
| ----------------------- | ----------------------------------------------------------------------------- |
| 时间步嵌入工具函数               | 正弦时间步嵌入 · 时间步 → 频率编码 · U-Net 时间条件注入                                          |
| 文本编码器（TextEncoder）      | CLIP 文本编码 · prompt → 条件嵌入 · 作为 U-Net Cross-Attention 的条件信号（condition）         |
| 交叉注意力（CrossAttention）   | Q 来自图像 latent · K/V 来自文本嵌入 · 公式：`Attention(Q,K,V) = softmax(QKᵀ/√d)V`      |
| U-Net 去噪网络              | Encoder-Decoder 对称结构 · 时间步嵌入条件 · 文本交叉注意力条件 · 残差跳跃连接                          |
| DDIM 采样循环               | 确定性采样 · `x_{t-1}` 由预测 `x_0` 和方向向量线性组合 · 相比 DDPM 大幅减少采样步数                    |
| VAE 解码                  | latent `[B,4,H/8,W/8]` → 像素 `[B,3,H,W]` · 解压缩还原 RGB 图像                      |


> **DDIM 与 DDPM 的区别** · DDPM 在每步加入随机噪声（随机过程），需 1000 步采样；DDIM 将逆扩散过程改写为确定性 ODE，在不降低生成质量的前提下可用 20-50 步完成采样，速度提升约 20 倍。
>
> 参考论文：*Denoising Diffusion Implicit Models*（Song et al., ICLR 2021）

---

### 08. Stable Diffusion 文生图实战

> `08.StableDiffusion_Text2Image_HuggingFace.ipynb`

基于 HuggingFace `diffusers` 库 + ModelScope 平台完成 Stable Diffusion 端到端文生图推理实战。涵盖 ModelScope `snapshot_download` 权重下载/本地缓存、Pipeline 加载与配置、文本 Prompt 输入、图像生成及 matplotlib 可视化全流程，生成结果保存为 `SD_generated.png`。


| 章节      | 内容                                                                    |
| ------- | --------------------------------------------------------------------- |
| 环境与依赖安装 | `diffusers` · `modelscope` · `accelerate` 安装验证                        |
| 模型加载    | ModelScope `snapshot_download` 下载缓存 · `StableDiffusionPipeline.from_pretrained` |
| 文生图推理   | 文本 Prompt 输入 · 推理步数 / 引导系数配置 · 图像生成 · 保存为 PNG                        |

---

### 09. Qwen3-VL 模型架构实现

> `09.Qwen3VL_ModelArchitecture_ViTVisionEncoder_Qwen3LLM_HuggingFace.ipynb`

使用 HuggingFace 加载 **Qwen3-VL-2B-Instruct**，展示官方 API 多模态推理与完整模型架构。核心创新点：**Patch Merger**（2×2 空间合并 + 跨模态投影）将相邻视觉 Patch 合并后映射至 LLM 隐藏空间（1024→2048）；**DeepStack** 中间层早期融合机制向 LLM 中间层注入多尺度视觉特征；**3D 卷积 Patch 嵌入** 统一处理图像（2D）与视频帧（3D）。


| 章节               | 内容                                                                                |
| ---------------- | --------------------------------------------------------------------------------- |
| 环境检查与依赖安装        | transformers · torch · accelerate 版本验证                                            |
| 模型加载与多模态推理       | 加载 `Qwen/Qwen3-VL-2B-Instruct` · 图文多轮对话推理 · 视觉问答示例                               |
| 完整模型结构展示         | `Qwen3VLForConditionalGeneration` 三大模块逐层打印 · 每层 tensor shape 与维度含义注释              |
| 3D 卷积原理详解        | 2D 卷积回顾（ViT Patch 嵌入） · 3D 卷积时空 Patch 嵌入 · 图像 vs 视频 patch 划分对比 · kernel 参数解析    |


> **Qwen3-VL 架构要点** · ViT 视觉编码器（24 层，hidden=1024）通过 3D Conv Patch Embedding 统一编码图像与视频；Patch Merger 将 2×2 相邻 Patch 合并后线性投影至 2048 维接入 LLM；Qwen3 语言模型主干（28 层，GQA + SwiGLU + RoPE）作为解码器。
>
> 参考论文：*Qwen3 Technical Report*（Qwen Team, Alibaba, 2025）

---

## 🛤 学习路径

```
视觉基础模型层
  01 ViT（Vision Transformer，图像 Patch + Transformer Encoder）
       ↓
图文对齐与多模态理解层
  02 CLIP（对比学习，图文零样本对齐）
       ↓
  03 ALBEF（ITC 对齐 + Cross-Attention 融合，先对齐后融合）
       ↓
  04 BLIP（MED 统一架构，理解 + 生成）
       ↓
  05 BLIP-2（Q-Former 桥接冻结 ViT + 冻结 LLM）
       ↓
扩散生成模型层
  06 Stable Diffusion 架构可视化
     ├── 06.1 print(model) 文本结构树
     ├── 06.2 torchinfo 参数表格
     └── 06.3 torchviz 计算图 SVG
       ↓
  07 DDIM 手动推理实现
     └─ 不依赖 Diffusers Pipeline
        从零实现：文本编码 → DDIM 采样 → U-Net 去噪 → VAE 解码
       ↓
  08 Stable Diffusion 文生图（HuggingFace Diffusers 一键推理）
       ↓
多模态大模型层
  09 Qwen3-VL（ViT + Patch Merger + DeepStack + Qwen3 LLM）
```

建议按编号顺序学习：先掌握 ViT 图像 Patch 化基础（01），再通过对比学习理解图文对齐机制（02-03），进阶到统一理解与生成框架（04-05），然后深入扩散生成模型（06-08），最后学习现代多模态大模型架构（09）。**06-08 系列建议连续学习：先通过三种工具吃透 SD 架构（06），再手写推理流程理解 DDIM 原理（07），最后用 Diffusers 一键验证（08）。**

---

## ⚙️ 环境依赖

> **说明**：所有 Notebook 共用同一套环境，按需安装对应模块依赖即可。

**核心依赖** · Python 3.14.5

| 包名                     | 版本           | 用途                                                          |
| ---------------------- | ------------ | ----------------------------------------------------------- |
| `torch`                | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2 加速）                                     |
| `torchvision`          | 0.27.0+cu132 | 图像预处理与视觉数据集工具                                               |
| `transformers`         | 5.14.1       | HuggingFace 模型加载与推理（ViT / CLIP / BLIP / Qwen3-VL）          |
| `diffusers`            | 0.39.0       | Stable Diffusion Pipeline（Notebook 08）                     |
| `accelerate`           | 1.14.0       | 混合精度与设备管理加速                                                 |
| `modelscope`           | 1.38.1       | ModelScope 模型下载与缓存（Notebook 06/07/08）                      |
| `sentence-transformers` | 5.6.0        | 句向量编码与语义相似度计算                                               |
| `torchinfo`            | 1.8.0        | 模型参数统计与 tensor shape 可视化（Notebook 06.2）                    |
| `torchviz`             | 0.0.3        | autograd 计算图 SVG 导出（Notebook 06.3）                        |
| `matplotlib`           | 3.10.9       | 图像展示与可视化                                                    |
| `numpy`                | 2.4.6        | 数值计算基础库                                                     |
| `pillow`               | 12.2.0       | 图像读取、格式转换与预处理                                               |

> ⚠️ Notebook 06.3（torchviz）额外需要系统级 **Graphviz**：
> ```bash
> # 推荐方式（自动配置 PATH）
> conda install graphviz
> # 或手动下载：https://graphviz.org/download/，将 bin/ 目录添加至系统 PATH
> ```

---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/MultimodalPractice.git
cd MultimodalPractice
```

**2. 安装依赖**

```bash
# --extra-index-url 在 PyPI 基础上追加 PyTorch 官方 CUDA wheel 源
# 使 torch==2.12.0+cu132 / torchvision==0.27.0+cu132 与其余包可一次性安装
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

**3. 准备模型权重**

所有 Notebook 均通过 HuggingFace Hub 或 ModelScope 在首次运行时**自动下载**并缓存至 `model_cache/` 对应子目录，无需手动下载。

- **Notebook 01**：自动下载 `google/vit-base-patch16-224` 缓存至 `model_cache/ViT/`
- **Notebook 02**：自动下载 `openai/clip-vit-base-patch32` 缓存至 `model_cache/CLIP/`
- **Notebook 03**：自动下载 `bert-base-uncased` 缓存至 `model_cache/bert/`
- **Notebook 04**：自动下载 `Salesforce/blip-image-captioning-base` 缓存至 `model_cache/BLIP/`
- **Notebook 05**：自动下载 `Salesforce/blip2-opt-2.7b` 缓存至 `model_cache/blip2/`
- **Notebook 06/07/08**：通过 ModelScope 自动下载 Stable Diffusion 2.1-base 至 `model_cache/SD/`
- **Notebook 09**：自动下载 `Qwen/Qwen3-VL-2B-Instruct` 缓存至 `model_cache/Qwen3VL/`

**4. 启动 Jupyter**

```bash
jupyter notebook
```

按编号顺序打开 Notebook 开始学习即可。

---

## 📄 License

[MIT](LICENSE) © 2026
