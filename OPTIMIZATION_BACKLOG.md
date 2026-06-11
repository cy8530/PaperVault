# Optimization Backlog

> 待优化项，按优先级排列。已完成项已移除。

## P1 — 高优先级

### Token 预算控制

**现状**：notes（5篇 × 6000 chars）+ chunks（5篇 × N chunks × 800 chars）+ prompt 直接拼入 context，无 token 计数。若超出模型上下文窗口会被静默截断。

**方案**：
- 设定 context token 预算（如 12000 tokens，为 answer 预留 2000）
- Notes 优先保证完整性（截断代价大）
- Chunks 按相似度贪心填充剩余预算
- 超出时动态减少 n_papers 或 chunks_per_paper
- 预算不足时打印提示

### 问题分类前置

**现状**：judge 已升级为三级（1/2/3），但所有问题都看到全部筛选后的论文笔记才判定。聚焦型问题（如"公式 X 的含义"）其实只需看 top 2-3 篇。

**方案**：问题分类前置（关键词匹配，零成本）：
- 聚焦型（含公式名/具体术语）→ 传 top 3 篇给 judge
- 对比型（含 对比/比较/vs/difference）→ 传 top 5 篇
- 探索型（含 哪些/综述/summary/survey）→ 传 top 5 篇

### 无测试覆盖

**现状**：整个项目没有任何单元测试或集成测试。

**方案**：优先覆盖核心模块：
- `indexer/chunker.py`：chunk 边界、章节识别、中文正则匹配
- `rag/qa.py`：`_gather_context` 各分支（无结果、notes only、chunks + section match）
- `indexer/store.py`：CRUD 操作的 mock LanceDB 测试
- `parser/extractor.py`：PDF 文本提取正确性

---

## P2 — 中优先级

### 论文检索相关性筛选可升级为图结构

**现状**：`_search_and_filter()` 已实现（扩大召回 2x + LLM 筛选），但本质上是"事后补救"——向量检索只看语义相似度，不知道论文间的引用、主题关联等结构关系。

**为什么关系图谱能根本优化**：
- 引用关系：论文 A 引用论文 B → 两者强相关，检索 A 时应提升 B 的权重
- 主题聚类：同一研究方向的论文形成簇，用户提问时可沿簇召回
- 关键词共现：两篇论文共享多个关键词 → 即使摘要语义差异大，也有主题关联
- 图谱可在导入时预计算（离线），检索时零 API 调用、零延迟

**方案**：
- 第二期实现论文关系图谱后，用图谱扩充向量检索：`graph_score * α + vector_similarity * (1-α)`
- 图谱模块提供"相关论文"API，供 RAG 检索时扩展候选集
- 可替代或增强当前 LLM 筛选，减少 API 调用量

### Context 中 notes/chunks 信息冗余

**现状**：notes 包含高度概括的方法结论，chunks 包含展开细节，两者在最终 context 中大量重叠。Prompt 写了"优先参考 [Detail:]"，但 token 已经花出去了。

**方案**：judge 判定需要 chunks 后，从 notes 中只保留 metadata 行（标题、作者、年份）和结论段，删去方法/实验段的详细描述（因 chunks 会覆盖），减少冗余。

### CLI 搜索按 chunk 展示而非按论文去重

**现状**：`search_cmd.py` 搜索 chunks 索引，同一篇论文的多个 chunk 重复展示为独立结果。用户看到 top-5 结果可能全部来自同一篇论文。Web 端 `/api/search` 已做 paper_id 去重，CLI 尚未同步。

### chunker 首个标题不更新 section 字段

**现状**：`chunker.py` 中，当第一个 section 标题是论文的首段内容（`current` 为空时），`current_section` 不会被更新。导致首个标题对应的所有文本块的 `section` 字段为空字符串。

**方案**：在 `is_header and not current` 分支中同步更新 `current_section = para`。

### usage_tracker 线程安全

**现状**：`usage.py` 中全局 `tracker` 在 Web 导入时被多个线程同时调用 `add()`。Python int 的 `+=` 非原子操作，并发下可能丢失计数。

**方案**：加 `threading.Lock` 或使用 `threading.local`。

### `get_note` 全表扫描

**现状**：`web/app.py` 中 `get_note` 无 `filename` 参数时，拉取整个 notes_index 表遍历查找 paper_id（O(n)）。

**方案**：使用 LanceDB filter 直接查询，或确保前端始终传递 `filename` 参数（当前前端已传，此为兜底路径优化）。

---

## P3 — 低优先级

### 多轮对话 Session

**现状**：每个 `ask()` 是无状态的独立调用。无法追问、无法指代前文（如"第二篇论文的方法呢？"）。

**方案**：
- Session 对象维护跨轮状态：`papers_mentioned`（已引用论文 ID 列表）+ `history`（每轮 Q&A 摘要）
- 每轮重新检索（追问可能转向不同论文）
- 历史以摘要形式传入 prompt（< 500 tokens）
- CLI 增加交互模式

### 向量检索去重

**现状**：`search_chunks_for_papers` 逐篇调用时，同一 chunk 可能被重复检索到。多个 paper_ids 的检索结果合并后无去重。

### notes 重名处理

**现状**：`_make_note_filename` 在多篇论文同名（如同一标题的不同版本）时会产生文件名冲突，后导入的覆盖先导入的。

**方案**：检测文件已存在时追加尾部序号（如 `Title_Venue_2024_2.md`）。

### SQL 拼接安全

**现状**：`search_chunks_for_papers` 中 paper_id 和 section 名称通过字符串拼接构造 SQL `IN` 子句。`_escape_sql` 已做单引号转义，但参数化查询更安全。

**方案**：考虑用 LanceDB 的 filter API 替代字符串拼接。

### extract_text 路径遍历风险

**现状**：`cache_path = cache_dir / f"{pdf_path.stem}.md"`，若 PDF 文件名为恶意构造（如 `../../etc/foo`），缓存路径可能写到 vault 目录外。低风险（需本地恶意文件），但作为开源项目应防范。

**方案**：校验 `pdf_path.stem` 不包含 `..` 或路径分隔符。

### embedder 线程安全

**现状**：`_get_model()` 的全局变量读写不是原子操作，多线程并发首次调用时可能重复加载模型。

### 无结构化日志

**现状**：所有输出通过 `print()`。生产环境需要 `logging` 模块支持级别过滤和日志持久化。

### `_SKIP_WORDS` 机构名硬编码 → ✅ 已移除

`_SKIP_WORDS` 随启发式元数据提取器（`_heuristic_extract`、`_extract_authors_from_block`、`is_plausible_title`）一同删除。元数据提取已统一为纯 LLM。

---

*最后更新：2026-06-11（标记已完成项）*
