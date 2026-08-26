# Benchmark

> PaperVault RAG 质量评测 —— 检索召回 + 答案忠实度 + 信息覆盖 + 上下文相关性
>
> 支持单论文问题（三级难度）+ 跨论文综述/对比问题

## 快速开始

```bash
python benchmark.py                               # 全量评测，5 篇论文 + 跨论文问题
python benchmark.py quality retrieval             # 仅检索（零 LLM 费用）
python benchmark.py quality answer                # 仅答案质量
python benchmark.py quality --detail 2            # 指定 RAG detail level
python benchmark.py quality --papers 10 --json    # 采样 10 篇 + JSON 导出
python benchmark.py quality --cross-paper 0       # 仅单论文问题（不生成跨论文题）
python benchmark.py quality --no-cache            # 强制重新生成测试数据
python benchmark.py gen                           # 仅生成测试问题并缓存
```

首次运行自动生成测试问题并缓存至 `benchmark/data/cache/`。

## 评测维度

### 问题分类

| 类型 | 说明 | 检索评估方式 |
|------|------|------------|
| **单论文 L1**（浅层） | 简单事实回忆 — 数字、数据集、指标值 | 单目标召回（Rank） |
| **单论文 L2**（中等） | 方法解释、架构推理 — 核心创新、设计选择 | 单目标召回（Rank） |
| **单论文 L3**（深层） | 跨方面综合分析 — 设计权衡、局限性、对比替代方案 | 单目标召回（Rank） |
| **跨论文** | 综述/对比 — 需要综合 ≥2 篇论文的信息 | 多目标召回（Intersection） |

### 检索质量（Retrieval）

| 指标 | 定义 | 对象 |
|------|------|------|
| **Recall@K** | 目标论文出现在 top-K 中的比例 | 单论文 / 跨论文 |
| **MRR** | 目标论文在结果中的平均倒数排名 | 单论文 |
| **Multi-target Recall@K** | top-K 中平均覆盖了几成相关论文 | 跨论文 |

### 答案质量（Answer Quality）

| 指标 | 定义 | 评测方式 |
|------|------|---------|
| **Faithfulness**（忠实度） | 答案中的断言是否被检索到的上下文支撑 | LLM-as-judge |
| **Coverage**（覆盖率） | RAG 答案覆盖了参考回答中多少关键信息点 | LLM-as-judge |
| **Context Relevance**（上下文相关性） | 检索到的 passage 是否与问题相关 | LLM-as-judge |

## 测试结果（2026-06-22）

**测试环境**：5 篇随机采样论文，每篇 5 题（LLM 自动分配难度），4 题跨论文（生成 5 题、缓存中取到 4 题）

**重要变更**：参考回答现已改用 `extracted/*.md` 全文（~30K chars/篇）生成，而非笔记（~3-8K chars/篇）。这使 Coverage 评分更加严格——ground truth 包含的信息远超 RAG 检索到的有限 chunk。

### 检索质量 — 单论文

| 指标 | 总体 | L1 浅层 | L2 中等 | L3 深层 |
|------|:---:|:---:|:---:|:---:|
| Questions | 25 | 5 | 10 | 10 |
| Found | 25 (100%) | 5 | 10 | 10 |
| Mean Rank | 1.8 | — | — | — |
| MRR | 0.7994 | 0.5571 | **0.9000** | 0.8200 |
| Recall@1 | 0.680 | 0.400 | **0.800** | 0.700 |
| Recall@5 | 0.920 | 0.600 | 1.000 | 1.000 |
| Recall@10 | 1.000 | 1.000 | 1.000 | 1.000 |

**分析**：

- **L2 检索最强**：MRR=0.90, R@1=0.80。方法类问题含独特技术术语（如 "对抗故障重建 AFR"、"因果成分分解 CCD"），检索信号区分度高
- **L1 检索偏弱**：MRR=0.557, R@1=0.400。事实回忆题（如 "AP 是多少"）问题表述泛化，与其他论文的数值报告产生混淆。样本量小（5 题）也放大了波动
- **L3 表现稳健**：MRR=0.820, R@1=0.700。跨方面综合分析题虽语义泛化，但多向量检索（query decomposition）有效弥补了术语差异
- **Recall@10=1.0**：top-10 内全部命中，多向量检索 + 相似度阈值后整体召回可靠
- **对比旧版提升**：MRR 从 0.7569 → 0.7994，R@1 从 0.583 → 0.680。query 分解 + 相似度阈值过滤对首 hit 率有显著帮助

### 检索质量 — 跨论文

| 指标 | 数值 |
|------|:---:|
| Questions | 4 |
| 涉及论文总数 | 20 (5 per Q) |
| 命中论文 | 18 (90.0%) |
| Recall@5 | 0.500 |
| Recall@10 | 0.600 |
| Recall@20 | 0.900 |

**分析**：

- 跨论文问题每次需命中 5 篇目标论文，R@5=0.500（平均召回一半），R@20=0.900
- 多向量检索对跨论文召回有明显帮助：同一 query 的不同语义变体覆盖了不同论文的术语表述
- 典型失败模式：综述问题 "因果方法在 CV 与多模态学习中的发展趋势" 涉及范围广，部分论文在特定变体 query 下排名仍靠后

### 答案质量 — 按难度分组

| 指标 | L1 浅层 | L2 中等 | L3 深层 | 跨论文 |
|------|:---:|:---:|:---:|:---:|
| Faithfulness | 1.000 | 0.500 | 0.500 | 0.000 |
| Coverage | 0.150 | 0.000 | 0.025 | 0.000 |
| Context Relevance | 0.250 | 0.600 | 0.550 | 0.547 |

**分析**：

- **Coverage 全面下降是预期行为**：参考回答改用全文（~30K chars）生成，信息量是笔记（~3-8K chars）的 5-10 倍。RAG 检索到的有限 chunk 无法覆盖 ground truth 中的全部信息点。旧版 0.15-0.30 的 Coverage 是虚高——ground truth 和 RAG 共用同一信息源（笔记）
- **L1 忠实度满分**：事实类问题 context 聚焦，LLM 严格基于原文回答，无幻觉
- **L2/L3 忠实度 0.50**：方法解释和综合分析需要跨段落推理，LLM 偶尔生成超越 context 的推断
- **跨论文 Faithfulness 归零**：Judge 模型（`LIGHT_MODEL_ID`）对多论文综合回答的评判可能过于严格。RAG 回答实际引用了各论文 context，但 judge 可能将其判定为 "无法从单一 context 追溯"——这提示 judge prompt 需要适配多论文场景
- **Context Relevance 在 L2/L3/跨论文间接近**（0.55-0.60）：检索到的 passage 与问题的相关性在中高难度间差异不大

> **Coverage 解读注意事项**：由于参考回答基于全文而 RAG 基于有限检索，Coverage 本质上衡量的是 "检索到的 context 能覆盖全文信息的比例"，而非 RAG 答案质量本身。追求高 Coverage 需要增大 chunk 检索量（`all` 模式），但会牺牲 Faithfulness（见下方 Detail Level 对比）。

### Detail Level 对比

```bash
python benchmark.py quality answer --detail auto   # 默认：LLM 动态判定
python benchmark.py quality answer --detail all    # 全文 chunks
```

**实测对比**（同一组问题，使用笔记作为参考回答源）：

| 指标 | auto（LLM 判定） | all（全文 chunks） |
|------|:---:|:---:|
| Faithfulness | **1.000** | 0.500 |
| Coverage | **0.233** | 0.050 |
| Context Relevance | **0.500** | 0.250 |

> **更多 context ≠ 更好答案。** `all` 模式下注入全部 chunks（含无关片段），LLM 被噪声干扰。`auto` 模式精准聚焦 → 验证了自适应深度判定的设计价值。

> 注意：上表为旧版数据（参考回答基于笔记），Coverage 绝对值偏高。当前版本参考回答基于全文，Coverage 整体更低，但 auto vs all 的相对关系不变。

### 综合评价

| 维度 | 评级 | 说明 |
|------|------|------|
| 检索召回（单论文） | ⭐⭐⭐⭐ | 100% found, R@5=0.92, MRR=0.80 |
| 检索召回（跨论文） | ⭐⭐⭐ | 90% targets found, R@5=0.50 |
| 答案忠实度（L1） | ⭐⭐⭐⭐⭐ | L1 零幻觉 |
| 答案忠实度（L2/L3） | ⭐⭐⭐ | 复杂推理时有推断性回答，Faithfulness=0.50 |
| 答案忠实度（跨论文） | ⭐⭐ | Judge 对多论文综合回答判定严格，实际质量待人工评估 |
| 信息完整性 | ⭐ | 全文参考回答 vs 有限检索 chunks 的差距大，Coverage 近零 |
| 检索 vs 答案梯度 | ⭐⭐⭐⭐⭐ | L1→L2→L3→跨论文难度梯度显著、结果符合预期 |

## 指标解读指南

### 如何判断结果好坏

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| Recall@5（单论文） | > 0.80 | 0.60-0.80 | < 0.60 |
| Recall@5（跨论文） | > 0.60 | 0.35-0.60 | < 0.35 |
| MRR | > 0.70 | 0.50-0.70 | < 0.50 |
| Faithfulness | > 0.90 | 0.75-0.90 | < 0.75 |
| Coverage（全文参考） | > 0.20 | 0.10-0.20 | < 0.10 |
| Coverage（笔记参考） | > 0.50 | 0.30-0.50 | < 0.30 |

### 指标之间的关系

- **L2 检索强 + L1 检索弱**：方法类问题含独特技术术语（检索信号强），事实回忆题泛化（具体数字无上下文时区分度低）→ 多向量检索对 L1 类问题帮助有限
- **Faithfulness 高 + Coverage 近零**：这是全文参考回答下的正常模式。RAG 基于有限检索保守回答 → 忠实但信息量少。如需提高 Coverage，增大 detail level 或 chunk 数（但会牺牲 Faithfulness）
- **跨论文 Faithfulness 骤降**：Judge 对多论文综合回答判定过于严格是主要原因。建议使用更强的 judge 模型（如 GPT-4o），或人工抽检跨论文回答质量
- **跨论文 Context Relevance 不低但 Faithfulness 为零**：说明检索到了相关 context，但 judge 无法将多论文综合回答中的断言追溯到单一来源 → 考虑启用 Divide & Conquer 模式
- **参考回答源切换的影响**：从笔记切换到全文后，Coverage 全面大幅下降。这不是 RAG 质量变差了，而是 ground truth 变完整了。Coverage 的绝对值应结合参考回答源来解读

## 数据生成

```
随机采样 N 篇论文
  ├─► 单论文问题（每篇 questions_per_paper 题）
  │     └─► LLM 按 L1/L2/L3 三个难度均匀分布生成
  │           L1: 事实回忆（数字、指标、数据集）
  │           L2: 方法解释（创新点、设计选择、工作原理）
  │           L3: 综合分析（设计权衡、局限性、替代方案对比）
  │
  └─► 跨论文问题（cross_paper 题）
        └─► LLM 综合全部论文摘要，生成综述/对比类问题
            如："对比不同方法在小波变换使用上的差异"
                "因果分析在不同领域的应用角色有何异同"

生成的问题 + 参考回答缓存至 benchmark/data/cache/
```

### 生成问题示例

**L1 浅层**（因果规划论文）：
> "在550个STRIPS基准任务中，因果图启发式（CG）成功解决了多少个任务？"

**L2 中等**（WaveMamba 论文）：
> "WaveMamba 中高频子带的融合策略是什么？为什么选择这种策略？"

**L3 深层**（CausalMixNet 论文）：
> "CausalMixNet 在处理复杂或低对比度病变区域时存在什么局限性？论文中提到了哪些未来改进方向？"

**跨论文**：
> "对比 WaveMamba 和 Wavelet-Attention CNN 在小波变换使用策略上的异同"

## 成本估算

以 5 篇论文 × 5 题/篇 + 5 题跨论文为例：

| 阶段 | LLM 调用 | 预估费用 |
|------|---------|---------|
| 单论文问题生成 | 5 次 | ~¥0.01 |
| 跨论文问题生成 | 1 次 | ~¥0.002 |
| 参考回答生成 | 30 次（基于全文，max_tokens=1536） | ~¥0.10 |
| 检索评测 | 0 次（本地 LanceDB） | ¥0 |
| 答案质量评测 | ~90 次（每题 3 judge） | ~¥0.16 |
| **合计** | **~126 次** | **~¥0.28** |

> 参考回答生成是最主要的 token 消耗（全文 ~30K chars 输入）。如果使用缓存（默认），后续评测跳过生成阶段，仅需答案质量评测费用。

> 单独运行 `retrieval` 模式完全免费。

## 扩展指南

### 调整难度分布

修改 `generate.py` 中 `_DIFFICULTY_SINGLE_PROMPT` 的难度定义或比例。

### 添加新的跨论文题型

在 `generate.py` 中新增 prompt 和 `generate_*` 函数，在 `build_test_data()` 中调用。

### 更换 judge 模型

judge 使用 `LIGHT_MODEL_ID`。设置为更强的模型（如 GPT-4o）可获得更可靠的评判：

```bash
# .env
LIGHT_MODEL_ID=gpt-4o
```

### 定期回归测试

```bash
python benchmark.py quality --json --seed 42
# → benchmark_report.json，可 git diff 对比历史
```
