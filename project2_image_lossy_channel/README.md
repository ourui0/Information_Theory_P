# 项目二：图像有损信源编码与 BSC/BEC 简单信道编码

Information Theory Course · Project 2
**主题**：Lossy source coding for image and simple channel coding over BSC and BEC.

本项目在统一的仿真框架下实现了一条完整的"图像 → 有损信源编码 → 信道编码 →
BSC/BEC 仿真传输 → 信道解码 → 信源解码 → 恢复图"的处理链路，
并按照课程要求从 **accuracy / algorithm complexity / PSNR** 三个维度评估。

| 模块 | 方法 |
| --- | --- |
| 有损信源编码 | 8×8 块 DCT + JPEG 亮度量化矩阵（quality 可调）<br>均匀标量量化（1–8 bit/像素，baseline） |
| 信道编码 | Hamming(7,4) 汉明码<br>(n,1) 重复码 (n 为奇数)<br>无编码（baseline） |
| 信道模型 | 二进制对称信道 BSC(p)<br>二进制擦除信道 BEC(p) |
| 评估 | PSNR、MSE、像素精度（@0, @±5）、BER（编码前/解码后）、编解码耗时、信源压缩比 |

---

## 目录结构

```
project2_image_lossy_channel/
├── main.py                      # CLI：单张图跑一条 pipeline
├── experiment.py                # 完整扫描实验（生成 CSV + PNG 图表）
├── requirements.txt
├── src/
│   ├── bitio.py                 # 位级打包/解包工具
│   ├── metrics.py               # PSNR / MSE / BER / accuracy
│   ├── pipeline.py              # 端到端管线与 Report
│   ├── source/
│   │   ├── quantize.py          # 均匀标量量化（1..8 bit/像素）
│   │   └── dct_codec.py         # 块 DCT + JPEG 量化 + 自适应位宽
│   └── channel/
│       ├── bsc.py               # BSC(p) 仿真
│       ├── bec.py               # BEC(p) 仿真
│       ├── repetition.py        # (n,1) 重复码 + 多数表决
│       └── hamming.py           # Hamming(7,4) 综合症解码 / 擦除解码
├── samples/
│   ├── generate_samples.py      # 生成 5 张 128×128 合成测试图
│   └── *.png
├── tests/                       # 59 个单元 / 集成测试（stdlib unittest）
│   ├── run_all.py               # 一键运行所有测试
│   ├── test_bitio.py
│   ├── test_metrics.py
│   ├── test_source.py
│   ├── test_channel.py
│   └── test_pipeline.py
└── report/                      # experiment.py 的输出
    ├── metrics.csv
    ├── rate_distortion.png
    ├── channel_sweep_bsc.png
    ├── channel_sweep_bec.png
    ├── complexity.png
    └── images/                  # 各参数下恢复图像 PNG
```

---

## 快速开始

```powershell
cd project2_image_lossy_channel
python -m pip install -r requirements.txt

# 1) 生成 5 张合成测试图
python samples/generate_samples.py

# 2) 单张图跑一条完整 pipeline
python main.py run --input samples/sinusoid.png ^
                   --source dct --source-param 60 ^
                   --channel-code hamming ^
                   --channel bsc --channel-p 0.01 ^
                   --output report/example.png

# 3) 跑完整实验（rate-distortion + BSC/BEC 扫描）
python experiment.py

# 4) 运行单元测试（59 个测试，仅用 Python 标准库）
python -m tests.run_all
```

实验结束后，`report/metrics.csv` 包含全部原始数据；
`rate_distortion.png`、`channel_sweep_bsc.png`、`channel_sweep_bec.png`
是最终报告的核心图表；`report/images/` 下是各配置的恢复图，可以直接贴进报告。

---

## 算法要点

### 1) 有损信源编码

**DCT codec（JPEG-like）** — 见 `src/source/dct_codec.py`

1. 图像居中到 `[-128, 127]`，按 8×8 分块，不足的边界像素做复制填充。
2. 每块做 2D 正交 DCT-II（通过预计算 DCT 矩阵 `D` 完成：`Y = D X Dᵀ`）。
3. 除以标准 JPEG 亮度量化矩阵 `Q_50` 的缩放版：
   $$Q(q) = \operatorname{clip}\!\Big(\big\lfloor (Q_{50}\cdot s(q) + 50)/100\big\rfloor,\,1,\,255\Big),$$
   其中
   $s(q) = 5000/q$（q < 50），$s(q) = 200 - 2q$（q ≥ 50）。
4. 按 Zigzag 顺序展平，低频在前、高频在后（便于后续扩展熵编码）。
5. **自适应位宽**：取所有量化后系数绝对值的最大值，选择最少的 `b` 位两补码能够容纳，
   写入 4 bit 头字段。这样 quality 越低、量化越粗、`b` 越小，码率自然下降。

解码做反向：解包位宽 → 按 Zigzag 还原 → 乘回 Q → 反 DCT → 加 128 → 裁剪到 `[0, 255]`。

**均匀标量量化器** — 见 `src/source/quantize.py`

把每个像素映射到 `k` bit 索引（`k ∈ {1..8}`），解码回量化 bin 的中点。
作为 baseline，用来说明"直接在像素域量化"与"在频域量化"的率失真差距。

### 2) 信道编码

**Hamming(7,4)** — `src/channel/hamming.py`

系统形式 `G = [I₄ | P]`，`H = [Pᵀ | I₃]`。速率 4/7。
- BSC：综合症 `s = H cᵀ` 直接查表定位单比特错误并翻转。
- BEC：若一个 7 比特码字中有 `e` 个擦除，枚举 `2^e` 种填充，找出满足零综合症的那个。
  对 `e ≤ 2` 总能唯一确定；`e = 3` 若对应 `H` 的 3 列线性无关也能确定；
  否则退化为随机填充。

**(n,1) 重复码** — `src/channel/repetition.py`

每个信息比特重复 `n` 次（`n` 为奇数）。速率 `1/n`。
- BSC：组内多数表决。
- BEC：在非擦除副本中做多数表决；若全部擦除则随机猜测。

### 3) 信道模型

- BSC(p)：每比特独立以概率 `p` 翻转；
- BEC(p)：每比特独立以概率 `p` 变为擦除符号 `2`（在 uint8 {0,1,2} 上表示）。

---

## 评估指标

对原图 `D`、恢复图 `D'`（均为 `uint8`，最大值 255）：

- **MSE**：$\text{MSE} = \frac{1}{HW}\sum_{i,j}(D_{ij}-D'_{ij})^2$
- **PSNR**：$\text{PSNR} = 10\log_{10}\!\left(\dfrac{255^2}{\text{MSE}}\right)$ dB
- **Accuracy @0**：`mean(D == D')`（逐像素精确匹配率）
- **Accuracy @±5**：`mean(|D - D'| ≤ 5)`（感知意义下的匹配率）
- **BER_raw**：信道输入 vs 信道输出的比特错误率（BEC 下为擦除率）
- **BER_decoded**：经信道解码后与原始信息比特的差异率
- **算法复杂度**：逐配置测量 `encode_time_s` 与 `decode_time_s`
- **压缩比**：`8·H·W / n_info_bits`（信源编码输出的信息比特数）

---

## 预期结论（在 `report/*.png` 中可视化）

1. **率失真曲线**：DCT codec 在每种图像上都明显优于标量量化器；同样的 bits-per-pixel
   下 DCT 的 PSNR 高出 ≥ 10 dB。
2. **BSC 扫描**：原始 BER 随 `p` 线性增长；Hamming(7,4) 在 `p ≤ 3%` 区域给出约 1–2 个
   数量级的解码 BER 下降；重复码 `n=5` 更稳健但以 5× 带宽为代价。
3. **BEC 扫描**：Hamming(7,4) 的擦除译码效果优异，在 `p ≤ 10%` 几乎无误码；
   `p ≥ 20%` 后重复码的"鲁棒随机回退"更可靠。
4. **复杂度**：DCT 单张 128×128 图的编解码总耗时 ≲ 10 ms；
   信道编解码是主要开销（重复码译码 ≈ O(nN)，Hamming 擦除枚举 ≈ O(N·2ᵉ)）。
5. **视觉**：`report/images/` 下同一张图在不同 q、不同 p 下的恢复结果按文件名排列，
   可直接排版进最终报告。

---

## 复现

```powershell
python -m pip install -r requirements.txt
python samples/generate_samples.py
python experiment.py              # ~1 分钟：产出 CSV 与所有 PNG
```

所有随机过程都使用 `np.random.default_rng(seed)`，默认 seed=0，结果可完全复现。

---

## 测试

测试套件覆盖了每一个模块。全部使用 Python 标准库 `unittest`，**无需额外依赖**。

| 文件 | 测试内容 |
| --- | --- |
| `tests/test_bitio.py` | bytes ↔ bits、signed int ↔ bits 往返、零填充、参数校验 |
| `tests/test_metrics.py` | MSE / PSNR（含已知值）、BER、像素精度 |
| `tests/test_source.py` | 标量量化器（k=8 无损、码率、PSNR 单调）+ DCT（头域往返、DCT 矩阵正交性、Zigzag 置换正确性、高质量 PSNR>40dB、码率随质量增大） |
| `tests/test_channel.py` | BSC（p=0/p=1、经验翻转率）+ BEC（p=0/p=1、擦除率、非擦除保真）+ 重复码（多数表决、BEC 全擦除→随机）+ Hamming（G·Hᵀ=0、单比特翻转必纠正、两擦除必纠正、编码增益） |
| `tests/test_pipeline.py` | 无噪声所有组合 → BER=0；Hamming 在 BSC(0.01) 下必降低 BER 并提升 PSNR；Rep(5) 在 BEC(0.1) 下必带来 ≥5dB 增益；相同 seed 产出逐比特一致 |

运行方式：

```powershell
python -m tests.run_all       # 或 python tests/run_all.py
# 或单独跑一个文件：
python -m unittest tests.test_channel -v
```

期望输出：

```
Ran 59 tests in 0.2s
OK
```
