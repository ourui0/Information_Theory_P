# 项目一：文本的无损信源编码与验证

Information Theory Course · Project 1  
**主题**：Design a kind of lossless source coding for text, program it and verify its coding efficiency.

本项目从零实现了三种经典的无损信源编码算法，并在统一的框架下对它们的
**信源熵 H(X)**、**平均码长 L̄**、**编码效率 η**、**冗余度 ρ** 和
**总体压缩比 CR** 进行数值验证：

| 方法 | 类型 | 核心思想 |
| --- | --- | --- |
| **Huffman** | 最优前缀码 (i.i.d.) | 贪心合并最低频率节点构造二叉树 |
| **Shannon-Fano** | 近似前缀码 (i.i.d.) | 按概率递归二分 |
| **LZW** | 通用字典编码 | 可变宽度编码不断增长的字符串字典 |

全部编码器都通过 `decode(encode(D)) == D` 的比特级往返测试，
即 `D' = D`，保证是真正的无损压缩。

---

## 目录结构

```
project1_source_coding/
├── main.py              # 命令行入口：encode / decode / verify
├── verify.py            # 完整实验脚本（生成 CSV 与图表）
├── requirements.txt
├── src/
│   ├── entropy.py       # 熵、平均码长、效率等信息论度量
│   ├── bitio.py         # 位级 I/O（MSB first）
│   ├── huffman.py       # Huffman 编解码 + 自描述文件格式 HUF1
│   ├── shannon_fano.py  # Shannon-Fano 编解码 + 文件格式 SF01
│   └── lzw.py           # 可变宽度 LZW + 文件格式 LZW1
├── samples/
│   ├── english.txt
│   ├── chinese.txt
│   ├── source_code.txt
│   └── dna.txt
└── report/              # verify.py 的输出（CSV + PNG）
```

---

## 快速开始

```powershell
cd project1_source_coding
python -m pip install -r requirements.txt

# 单文件验证：对比三种方法的编码效率
python main.py verify --input samples/english.txt

# 编码 / 解码一份文本
python main.py encode --method huffman --input samples/english.txt --output out.huf
python main.py decode --method huffman --input out.huf            --output recovered.txt

# 跑完整实验（遍历 samples/，写 report/metrics.csv 与 PNG 图表）
python verify.py
```

---

## 算法要点

### 1. Huffman 编码 (`src/huffman.py`)

1. 统计字符频率 `f_i`。
2. 用最小堆反复合并两个频率最低的节点，得到 Huffman 树。
3. 左边=`0`、右边=`1`，得到前缀码表 `{x_i ↦ c_i}`。
4. 按 **HUF1** 文件格式写入：magic、符号数、总字符数、payload 位长、
   符号表（UTF-8 + 频率）、比特流。
5. 解码时重建相同的树，按比特遍历还原字符。

**理论保证**：对任意 i.i.d. 信源，Huffman 码满足

$$H(X) \le \bar L < H(X) + 1.$$

`verify.py` 会显式打印这一不等式的数值成立情况。

### 2. Shannon-Fano 编码 (`src/shannon_fano.py`)

1. 按概率降序排列符号。
2. 在使左右子集概率和之差最小的位置切分；左子集追加 `0`，右子集追加 `1`。
3. 递归到单元素为止。

Shannon-Fano 生成前缀码但不保证最优，实验中其平均码长 ≥ Huffman，
正好与理论一致。

### 3. LZW 编码 (`src/lzw.py`)

1. 字典初始化为 0–255 的单字节。
2. 贪心地把能继续匹配的前缀扩展到 `w`；一旦 `w + c` 不在字典中，
   输出 `code(w)`，把 `w + c` 加入字典，`w ← c`。
3. 码宽从 9 bit 起按字典增长到 16 bit。
4. 格式为 **LZW1**：magic、payload 位长、变宽比特流。

LZW 能利用字符间的长程相关性，对重复性强的文本（如 `dna.txt`、
`source_code.txt`）通常给出最高压缩比。

---

## 验证指标

对 alphabet 为 `{x_i}`、概率为 `p_i`、码长为 `l_i` 的编码方案：

- **信源熵**：$H(X) = -\sum_i p_i \log_2 p_i$  
- **平均码长**：$\bar L = \sum_i p_i\, l_i$  
- **编码效率**：$\eta = H(X) / \bar L$  
- **冗余度**：$\rho = 1 - \eta$  
- **压缩比**：原始 UTF-8 字节数 / 输出字节数（含所有头信息）

对 LZW，由于它不是定长符号码，我们用 `L̄ = 8 · 输出字节数 / 字符数`
作为其每符号代价。

---

## 预期结论（英文样本 `english.txt`）

- 字符表大约 60 个符号，一阶熵 H(X) ≈ 4.4 bit/符号。
- Huffman：L̄ ≈ 4.43，η ≈ 99.3%，满足 4.4 ≤ L̄ < 5.4。
- Shannon-Fano：L̄ 略大于 Huffman，效率略低。
- LZW：利用了冗余的单词、空格，压缩比通常高于 Huffman。
- 所有算法 `decode(encode(D)) == D` 都通过。

详见运行 `python verify.py` 后生成的 `report/metrics.csv` 与图表。
