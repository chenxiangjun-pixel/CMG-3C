# DMC–EAB/SCAG–TMD for LiTS2017

## 概述
本项目实现了针对 LiTS2017 肝脏/肿瘤分割的三阶段协同框架 **DMC–EAB/SCAG–TMD**：

1. **Dynamic Multi-scale Context (DMC)**：在编码器每一级引入多尺度并行分支，通过可学习门控自适应聚合不同感受野，同时在损失中加入熵正则防止塌陷。
2. **Edge-Aware Skip Gating (EAB/SCAG)**：从浅层特征中显式提取边缘响应，通过门控与语义对齐抑制噪声，并引入 Boundary Dice/HD surrogate 边界损失。
3. **Transformer Mask Decoder (TMD)**：引入查询驱动的掩码迭代精化模块，与轻量 CNN 解码器并联，利用前景引导的交叉注意力提升细节与全局一致性。

该框架遵循“召回→对齐→判别”流程：DMC 捕获多尺度上下文以召回潜在区域；EAB/SCAG 利用边界信息校准跳连；TMD 结合 transformer 查询与 CNN 精细判别。项目默认使用 PyTorch 2.3+，可选启用 MONAI 数据增强。

## 环境搭建

### 使用 pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 使用 Conda（可选）
```bash
conda create -n dmc-tmd python=3.10 -y
conda activate dmc-tmd
pip install -r requirements.txt
```

确保系统安装 CUDA 11.8+ 以及与 PyTorch 匹配的驱动。如仅使用 CPU，训练脚本会自动调节批大小为 1。

## 数据准备
1. 从 LiTS2017 官方获取数据并解压到：`project/data/raw/LiTS2017/`，应包含 `volume-*.nii` 与 `segmentation-*.nii`。
2. 运行预处理脚本：
   ```bash
   python scripts/prepare_lits.py
   ```
   该脚本会完成：
   - 三线性/最近邻重采样至 `1.0×1.0×1.0 mm`；
   - HU 窗口裁剪至 `[-200, 250]` 并做 Z-score 标准化；
   - 生成训练/验证/测试划分（70/15/15）并保存至 `data/processed/splits.json`；
   - 将处理后体积与标签保存至 `data/processed/{train,val,test}/`。

若需自定义划分，可编辑 `scripts/prepare_lits.py` 顶部常量或通过参数覆盖。

## 训练

### 最小复现（单卡 GPU 或 CPU）
```bash
python train.py
```
默认配置位于 `config/default.yaml`，其中设置了小批量（GPU: 2，CPU: 1）与 patch 大小 `96×160×160`，适用于显存 16GB 左右的 GPU。如遇 OOM，可在配置中调低 `patch_size` 或 `batch_size`。

### 标准训练
```bash
python scripts/train.py --config config/default.yaml --amp True --grad_accum 2
```
训练日志与 TensorBoard 文件写入 `logs/`，最佳模型与最近 checkpoint 存储在 `weights/`。脚本自动记录配置、指标（肝脏/肿瘤 Dice、IoU、HD95、ASD）并支持断点续训。

## 消融实验
运行以下脚本可依次禁用各模块并记录结果：
```bash
python scripts/train.py --config config/ablation_no_dmc.yaml
python scripts/train.py --config config/ablation_no_eab.yaml
python scripts/train.py --config config/ablation_no_tmd.yaml
```
可在 `logs/ablation_results.csv` 中整理结果，模板如下：

| 配置 | Liver Dice | Tumor Dice | Avg Dice | Liver HD95 | Tumor HD95 | Avg ASD |
|------|------------|------------|----------|------------|------------|---------|
| default |            |            |          |            |            |         |
| no DMC |            |            |          |            |            |         |
| no EAB |            |            |          |            |            |         |
| no TMD |            |            |          |            |            |         |

## 评估与可视化
```bash
python scripts/evaluate.py --config config/default.yaml --split val
python scripts/evaluate.py --config config/default.yaml --split test
```
评估脚本会输出整体指标与逐病例 CSV 至 `logs/metrics/`，并在 `logs/vis/` 保存若干病例的切片可视化与注意力热力图。

单病例推理：
```bash
python scripts/infer_case.py --config config/default.yaml --case volume-0.nii.gz
```
将导出预测掩码与 PNG 切片到 `logs/infer/volume-0/`。

## 常见问题
- **CUDA OOM**：降低 `batch_size`、`patch_size` 或关闭 `tmd.use_transformer`。可启用梯度累计与检查点。
- **NIfTI 朝向不一致**：预处理脚本会标准化到 LPS 方位并写入 affine。若仍报错，请确认原始数据是否损坏。
- **Spacing 不一致**：脚本会自动重采样并断言 spacing 与标签匹配；若失败，请检查源文件 Header。
- **训练不收敛**：尝试调低学习率、增加 warmup 步数，或减少数据增强强度。

## 许可证与引用
本项目采用 [MIT License](LICENSE)。若使用本仓库，请引用 LiTS2017 数据集与相关工作。
