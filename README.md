# 脑肿瘤进展预测（Brain Metastasis Progression Prediction）

基于 **3D 潜空间扩散模型（Latent Diffusion Model, LDM）** 的脑转移瘤（Brain Metastasis）MRI 进展预测项目。输入单次随访的 T1 增强 MRI 体数据，即可：

- 预测未来任意天数（如 Day 0 ~ Day 100）的肿瘤进展序列；
- 在保持脑部解剖结构、纹理细节的前提下模拟局部肿瘤扩散；
- 附带独立的四分类模型，识别肿瘤类型（Type_1 ~ Type_4）。

> ⚠️ 本项目仅用于学术研究 / 算法演示，**不构成任何临床诊断或治疗建议**。

88fbbfbd-86b8-4f9e-9203-9ab4adfd54db.png
---

## 目录

- [数据集](#数据集-dataset)
- [算法原理](#算法原理-algorithm)
- [模型结构](#模型结构-model-architecture)
- [使用方法](#使用方法-usage)
- [评估指标](#评估指标-evaluation)
- [目录结构](#目录结构)
- [依赖环境](#依赖环境)
- [已知限制](#已知限制)

---

## 数据集 (Dataset)

### BrainMets 脑转移瘤 MRI 数据集

- **模态**：`t1_gd`（T1 加权增强），每个病例由一组 2D PNG 切片按 `_z` 序号堆叠为 3D 体数据。
- **尺寸**：所有体数据统一重采样为 **64 × 64 × 64**（三线性插值）。
- **划分**：训练集 **105** 例，测试集 **51** 例。
- **标签**：每个病例目录下含 `seg` 分割标签，但本项目为**无监督重建/生成模型**，训练与评估**不直接使用** seg 标签计算 Dice。
- **分类标签**：四分类任务需要额外的 `tumor_labels.csv`（列：`case, type`，如 `Mets_001,Type_1`），供独立的肿瘤类型分类器使用。

### 数据目录结构

```
BRAINMETASTASIS/
├── train/
│   └── Mets_XXX/
│       ├── t1_gd/
│       │   ├── slice_z0.png
│       │   ├── slice_z1.png
│       │   └── ...
│       └── seg/
│           └── ...
├── test/
│   └── Mets_XXX/ ...
└── tumor_labels.csv        # 可选，四分类标签
```

### 数据获取

BrainMets / BraTS 类医学影像数据通常需要登录授权，项目**未内置公开下载地址**。请将有权访问的直链填入 `DATA_URL`，调用 `download_brain_dataset()` 即可自动下载并解压（支持 zip / tar / gz）。

```python
DATA_URL = '你的授权下载直链'
DATA_ROOT = download_brain_dataset()
```

---

## 算法原理 (Algorithm)

整体流程：**VAE 将 3D 体数据压缩到潜空间 → 条件潜空间扩散模型（LDM）在潜空间做去噪演化 → 解码回体空间 → 脑结构保持后处理**。

### 1. 潜空间扩散模型（LDM）

借鉴 Stable Diffusion 的思路，先训练一个 3D VAE 将 64³ 的体数据压缩为 **4 × 16 × 16 × 16** 的潜变量，再在低维潜空间执行扩散过程，显著降低 3D 数据的高昂计算成本。

- **前向扩散**（加噪）：`z_t = √(ᾱ_t) · z_0 + √(1 − ᾱ_t) · ε`
- **β 线性调度**：β 从 `1e-4` 线性增长到 `0.02`，总步数 `T = 200`
- **训练目标**：UNet 预测噪声 `ε_θ(z_t, t, cond)`，损失为 MSE，并做梯度裁剪（max_norm=1.0）

### 2. AdaNI 自适应噪声注入（Adaptive Noise Injection）

根据**预测随访天数差 Δd** 自适应选择扩散起始噪声步 `t_start`，天数差越大，注入噪声越多，演化幅度越大：

| 随访天数差 Δd | 起始噪声步 t_start |
|---|---|
| |Δd| ≤ 3 | 20% × T |
| 3 < |Δd| ≤ 7 | 50% × T |
| 否则（长期随访） | 80% × T |

### 3. 两阶段推理

- **阶段一（演化生成）**：从 AdaNI 选定的 `t_start` 开始逐步去噪，生成肿瘤扩散的整体变化。
- **阶段二（低噪声细化）**：从 15% × T 开始再次去噪，用于消除阶段一引入的伪影、保持脑解剖结构稳定。

### 4. 纹理感知 VAE 重建损失

普通 VAE 重建容易过度平滑、丢失脑部细节，因此训练损失中引入**三方向梯度损失**：

```
L = MSE + 0.5 · L1 + 0.25 · GradientLoss(3D)
```

其中 GradientLoss 约束 x/y/z 三个轴方向的局部梯度（L1 差分），迫使重建保留高频纹理。

### 5. 脑结构保持后处理

扩散生成结果解码后，进一步做约束融合，确保“脑还是那个脑，只有肿瘤在变”：

- **去噪 + 纹理回填**：按 `DENOISE_STRENGTH` 平滑去噪，再用 `SHARPEN_STRENGTH` 回填输入的高频脑纹路细节；
- **病变区曝光降低**：对病灶候选区降低 `LESION_EXPOSURE_REDUCTION` 的过度增强；
- **残差截断**：生成与输入的差异截断在 ±8%（`MAX_GENERATED_CHANGE`），防止解剖结构被大幅篡改；
- **轮廓锁定**：用输入脑掩膜（>0.035）锁住外轮廓，避免边界变薄或断裂。

### 6. 肿瘤候选掩膜

用「局部均值 + 局部细节」双通道软掩膜（sigmoid 加权：0.65×高信号 + 0.35×局部热点）自动定位病灶区域，供后处理约束与可视化红框标注使用。

---

## 模型结构 (Model Architecture)

| 模块 | 说明 |
|---|---|
| `SimpleVAE3D` | 3D VAE。编码器：Conv3d 1→16→32→4；解码器：ConvTranspose3d 4→32→16→1 + Sigmoid |
| `LatentUNet3D` | 条件潜空间 UNet。时间嵌入（1→32→32）+ 脑全局条件（潜变量全局池化 → 4 维），U 型编解码结构 |
| `TumorClassifier3D` | 独立四分类器（Conv3d 1→16→32→64 + 自适应池化 + Linear→4），与扩散模型解耦 |
| `LDMScheduler3D` | 扩散调度器：β 线性调度、前向加噪、AdaNI 起点选择、两阶段采样 |

条件注入方式：时间步 `t` 归一化到 [0,1] 后经 MLP 嵌入，与脑全局条件向量相加，广播到整个潜空间后与带噪潜变量拼接。

---

## 使用方法 (Usage)

### 1. 环境准备与数据

```bash
pip install torch nibabel numpy matplotlib pillow
```

按上文“数据获取”准备 `BRAINMETASTASIS/train` 与 `BRAINMETASTASIS/test`。

### 2. 数据体检

```python
train_report = validate_brainmets_dataset(DATA_ROOT / 'train')
test_report = validate_brainmets_dataset(DATA_ROOT / 'test')
```

### 3. 训练（VAE 预训练 → LDM 训练 → 测试集重建评估）

```python
vae, unet, history, test_metrics = train(TRAIN_ROOT, TEST_ROOT)
```

- VAE 预训练 3 个 epoch（纹理感知损失），随后冻结；
- LDM UNet 训练 5 个 epoch（噪声预测 MSE + 梯度裁剪）；
- 保存权重至 `brain_tumor_ldm.pt`，并在测试集上输出 MSE / MAE / PSNR。

### 4. 四分类肿瘤类型识别（可选，独立分支）

```python
train_tumor_classifier()          # 需先准备 tumor_labels.csv
result = classify_tumor_case(test_case_path)
# → {'type': 'Type_2', 'confidence': 0.93, 'probabilities': {...}}
```

### 5. 推理：未来进展预测

```python
PREDICT_DAYS = 100   # 生成 Day 0 ~ Day 100
sequence, metrics = predict_tumor_progression(
    input_path=TEST_ROOT / 'Mets_001',   # 支持 PNG 病例目录或 .nii/.nii.gz
    predict_delta_days=PREDICT_DAYS,
    change_strength=0.35,
)
plot_full_progression(sequence)         # 三方向（轴/矢/冠）多时间点可视化，红色小框标注病灶热点
```

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DENOISE_STRENGTH` | 0.35 | 去噪强度 |
| `SHARPEN_STRENGTH` | 0.24 | 纹理锐化回填强度 |
| `LESION_EXPOSURE_REDUCTION` | 0.22 | 病变区曝光降低幅度 |
| `MAX_GENERATED_CHANGE` | 0.08 | 允许的最大生成变化（防结构篡改） |
| `change_strength` | 0.35 | 推理时病灶变化幅度 |

### 6. 自检

```python
smoke_test()   # 前向传播形状校验
```

---

## 评估指标 (Evaluation)

- **重建质量**（测试集）：MSE、MAE、PSNR（dB）
- **结构保持**（推理时，`compare_structure_quality`）：
  - **脑轮廓 IoU** > 0.95：生成前后脑掩膜重合度；
  - **高频纹理相关性** > 0.90：去均值后的纹理细节相关性；
  - **病变区均值变化**：病灶区曝光差异（模拟进展的体现）。

---

## 目录结构

```
脑肿瘤预测.ipynb
├── 0. 全局配置与设备检测
├── 1. 2D 切片堆叠为 3D 数据集（BrainMets 加载/下载/体检）
├── 2. 3D SimpleVAE（保留脑部纹理）
├── 3. 3D 条件 Latent-UNet（扩散去噪网络）
├── 4. LDM 调度器 + AdaNI 自适应噪声注入
├── 5. 训练流程（VAE 预训练 → LDM 训练 → 重建评估）
├── 6. 推理与可视化（两阶段采样 + 结构保持 + 序列生成）
├── 7. 运行前快速自检（smoke test）
└── 8. 三方向多步骤可视化
```

---

## 依赖环境

- Python 3.8+
- PyTorch（CPU / CUDA / MPS 均可，自动检测）
- nibabel、numpy、matplotlib、Pillow

---

## 已知限制

1. **数据授权**：BrainMets 等医学影像数据需登录/授权获取，未内置公开直链；
2. **无监督性质**：模型不做病灶分割/Dice 评估，seg 标签仅作参考；扩散结果需人工复核；
3. **临床边界**：预测序列为算法模拟，不能替代医生判断与随访影像；
4. **计算资源**：3D 扩散模型显存/内存占用较高，`BATCH_SIZE` 默认 1，长序列生成（如 Day 100）耗时随帧数线性增长。

---

## License

仅供学术研究与算法学习使用。数据集遵循其原始授权条款，使用前请确认。
