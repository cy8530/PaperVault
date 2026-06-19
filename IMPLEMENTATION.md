# Implementation Notes

> 技术实现文档，记录各模块的具体方法、设计决策、已知问题与优化方向。

## 1. PDF 文本提取

**选型：** PyMuPDF (fitz)

**为什么不用 Marker：**
- Marker 下载 1.35GB 模型 + 每篇论文推理 10-17 分钟（Apple MPS）
- PyMuPDF 直接解析 PDF 文字流，毫秒级，无需 GPU
- 输出文本量差异仅 ~8%（~39K vs ~42K chars）

**实现：** `paper_vault/parser/extractor.py::extract_text()`

- 按页遍历，`page.get_text("dict")` 获取块→行→span 结构
- 只取 `type == 0`（文本块），跳过图片
- 首次提取后缓存为 `extracted/{stem}.md`，后续直接从缓存读取
- 用户可手动编辑缓存文件修正提取错误


## 2. 笔记生成

**实现：** `paper_vault/notes/generator.py::generate_note()`

- Prompt 模板 → `prompts.py::NOTE_PROMPT`
- 全文传入 LLM，不做截断（由模型自身上下文窗口限制）
- 参数由 `config.NOTE_GEN_TEMPERATURE` / `config.NOTE_GEN_MAX_TOKENS` 控制（默认 0.3 / 4096），支持环境变量覆盖
- 输出：中文结构化 Markdown（标题/作者/问题/方法/结论/公式/表格）
- LLM 客户端通过 `utils.get_llm_client()` 懒加载单例获取，全项目共享
- 输出自动剥离 ``` 代码围栏（`utils.strip_code_fences`）

**笔记文件命名：** `importer.py::_make_note_filename()`

- 从元数据构建文件名：`Title_Venue_Year.md`
- 使用 `utils.sanitize_part()` 移除非法字符（`/\:*?"<>|`），空白替换为下划线
- venue/year 缺失时自动省略，仅保留 title
- title 缺失时回退到 PDF 文件名
- 文件名长度限制由 `config.NOTE_FILENAME_MAX_LEN` 控制（默认 150）

**Prompt 要求：**
1. 中文输出
2. 提取元数据（标题、作者、发表信息）
3. 1-2 句核心问题 + 3-5 条方法要点
4. 2-3 条关键发现 + 重要公式及通俗解释
5. 描述文中的表格内容

**已知限制：**
- 当前只用提取文本调 LLM，无图表/公式的视觉信息，因此依赖于PyMuPDF提取结果
- 后续混合模式：渲染 PDF 中图表/公式密集区为图片，追加到 Prompt

## 3. 文本分块

**实现：** `paper_vault/indexer/chunker.py::chunk_text()`

**策略：** 基于段落/章节边界而非机械截断

1. 空行拆段落
2. 逐段累加，遇章节标题（正则匹配覆盖中英文学术论文常见章节）→ 立即输出前块、开始新块
3. 或累加超出 800 chars → 输出前块、保留 100 chars 重叠开始新块
4. 尾块输出

**参数：** 由 `config.CHUNK_SIZE` / `config.CHUNK_OVERLAP` 控制（默认 800 / 100 chars），支持环境变量 `PAPER_VAULT_CHUNK_SIZE` / `PAPER_VAULT_CHUNK_OVERLAP` 覆盖

**效果：** 9 页论文约 50-60 个 chunks

**章节标签：** 每个 chunk 附带 `section` 字段（所属章节标题），如 "3. Method"、"实验评估"。章节正则 `_SECTION_RE` 覆盖英文 + 中文论文学术论文常见章节类型：

- 英文：Abstract, Introduction, Method, Experiments, Conclusion 等
- 中文：摘要、引言、相关工作、方法论、实验、结论、消融 等

## 4. 向量 Embedding

**实现：** `paper_vault/indexer/embedder.py`

**模型：** 通过 `EMBEDDING_MODEL` 环境变量配置，默认 `intfloat/multilingual-e5-base`（768 维，~500MB）

- 懒加载单例：首次调用自动从 HuggingFace 下载并缓存，之后复用
- `normalize_embeddings=True`：L2 归一化，后续点积 = 余弦相似度
- 批量编码：`model.encode(texts, ...)` 一次处理所有 chunks
- `embedding_dim()` 动态获取模型输出维度，LanceDB schema 据此适配 vector 列宽
- E5 系列需要 query/passage 前缀：`embed_texts()` 通过 `is_query` 参数自动添加

**为什么选 multilingual-e5-base：**
- 768 维，中英文检索质量与速度的平衡点
- 多语言（100+ 语言），跨语言检索（中文问题→英文 chunks）优于 384 维
- 500MB 体积，内存占用 ~1GB，本地完全可跑
- 通过 `.env` 可切换为 `intfloat/multilingual-e5-small`（384 维，200MB，更快）或 `BAAI/bge-m3`（1024 维，1.5GB，更强）

**部署注意事项：**
- 首次运行需下载 500MB 模型文件，缓存于 `vault/models/huggingface/`（由 `HF_HOME` 环境变量控制）
- 国内用户可通过 `HF_ENDPOINT=https://hf-mirror.com` 加速下载
- 切换模型后需重建索引（`python pv.py import --force`）

## 5. 向量存储

**实现：** `paper_vault/indexer/store.py`

**数据库：** LanceDB（嵌入式，`vault/vectors/`）

**Schema：**
| 字段 | 类型 | 说明 |
|---|---|---|
| `paper_id` | string | 论文标识（文件名去后缀） |
| `chunk_idx` | int32 | 块序号 |
| `text` | string | 原始文本 |
| `section` | string | 所属章节标题（如 "3. Method"） |
| `vector` | float32[N] | 嵌入向量（维度由 embedding 模型决定） |
| `title` | string | 论文标题 |
| `authors` | string | 作者列表（逗号分隔） |
| `year` | int32 | 发表年份（0 表示未知） |
| `keywords` | string | 关键词（逗号分隔） |

元数据字段直接存储在 chunks 表中（非关联表），使 LanceDB 的 `where` 子句能直接过滤。`section` 列支持章节级别的定向检索。

**notes_index 补充字段：**
- `sections` (string): JSON 数组，记录论文章节结构及 chunk 区间，如 `[{"heading": "3. Method", "chunk_start": 6, "chunk_end": 25}, ...]`。用于 RAG 章节定向检索。

**`rebuild_chunks_index()` / `rebuild_notes_index()` 逻辑：**
- 连接 `vault/vectors/` 下的 LanceDB
- 接收 `meta` dict（LLM 提取的元数据），写入每行
- 如表存在 → `DELETE WHERE paper_id = 'xxx'`（删旧数据）
- `ADD` 新行
- 如表不存在 → `CREATE TABLE` + 写入

**`search_chunks()` 逻辑：**
- `table.search(query_vector).where(where_clause).limit(top_k).to_list()`
- `where` 子句支持 SQL：`year >= 2024`, `authors LIKE '%name%'`
- LanceDB 默认 IVF-PQ 近似搜索，非暴力扫描
- 返回 `{paper_id, chunk_idx, text, title, authors, year, _distance}`

**为什么选 LanceDB：**
- 嵌入式零运维（vs Qdrant 需 Docker）
- 列存格式读写快（vs ChromaDB SQLite 单写者）
- 原生 SQL where 过滤（vs FAISS 无持久化、无过滤）

**已知问题：**
- 元数据存于每一行（冗余），但避免了 LanceDB 不支持 join 的限制

## 5.5 元数据提取

**实现：** `paper_vault/notes/generator.py::extract_paper_metadata()`

**数据源：提取后的文本，非原始 PDF**

```
PDF → PyMuPDF → extracted/*.md → 取前 N 字符 → LLM 提取
```

**纯 LLM 策略（2026-06 重构）：** 彻底移除启发式提取器，元数据提取统一走 LLM。

**为什么移除启发式：**
- 真实 PDF 布局千差万别（期刊头部、arxiv 格式、中文论文、Elsevier/Springer 模板），靠正则追赶排版多样性永远追不上
- 每次遇到新格式就加检测规则，本质是打补丁，代码膨胀且维护成本高
- 元数据是整个系统的地基（影响搜索、RAG 引用、笔记文件名、Web UI 展示），省下的 API 成本远小于出错代价
- 轻量 LLM 调用一次约 $0.0003，100 篇论文的元数据提取成本可忽略

**实现细节：**
- 取文本前 `config.METADATA_SNIPPET_CHARS` 字符（默认 6000），调 `LIGHT_MODEL_ID`
- Prompt (`META_PROMPT`) 要求返回纯 JSON：`{title, authors, year, venue, keywords}`
- JSON 解析通过 `utils.parse_llm_json()` 统一处理（自动剥离 ``` 围栏）
- 解析失败 → 以 temperature=0.3 重试一次 → 仍失败则打印 `[WARN]` 日志并返回空 dict
- `max_tokens` 由 `config.META_EXTRACT_MAX_TOKENS` 控制（默认 512）

**fix-metadata 命令：** `pv.py fix-metadata <paper_id> [--all]`

重新提取元数据并原地更新 LanceDB 双表（chunks + notes_index），同时重命名笔记文件（当 title 变化导致新文件名与旧文件名不同时）。无需重新导入 PDF。根据 paper_id 反向查找 extracted 缓存文件，支持精确匹配 → 规范化匹配 → 子串匹配三级查找。

**设计决策：**
- LLM 天然理解各种排版（Elsevier、Springer、arxiv、中文期刊），中英文同一套逻辑
- 代码量从 ~145 行缩减到 ~40 行（移除 `_heuristic_extract`、`_extract_authors_from_block`、`_SKIP_WORDS`、`is_plausible_title`）
- 元数据提取不再区分 "heuristic" / "llm" 方法，统一为 "llm"

### 5.6 增量导入与内容去重

**实现：** `paper_vault/importer.py::import_pdfs()` + `store.py`

**文件名去重：**
- 导入前查询 notes_index 中已索引的 paper_id 集合
- PDF 文件名（stem）已在集合中 → 自动跳过
- `--force` 参数强制重新处理所有论文

**内容哈希去重（2026-06 新增）：**
- `import_one()` 提取文本后，计算 `text[:config.DEDUP_HASH_CHARS]` 的 MD5 作为 `content_hash`
- 导入前查询已索引的 content_hash 集合（`store.get_indexed_hashes()`）
- 相同内容的 PDF 以不同文件名导入时自动跳过（避免 `2311.05316v1` 和 `ABIGX_...` 重复索引同一论文）
- `content_hash` 写入 notes_index 的 `content_hash` 列

**文件大小限制：**
- `MAX_PDF_SIZE_MB`（环境变量 `PAPER_VAULT_MAX_PDF_MB`，默认不限）：CLI 和 Web 导入路径均在 `import_one()` 入口统一校验，超限抛出 `ValueError`
- `MAX_UPLOAD_MB`（环境变量 `PAPER_VAULT_MAX_UPLOAD_MB`，默认 100MB）：Web 上传端点检查 `Content-Length` 请求头，超限返回 HTTP 413

### 5.7 两级模型配置

**实现：** `config.py`

任务按复杂度分层，使用不同模型：

| 配置项 | 用途 | 特点 |
|---|---|---|
| `LLM_MODEL` | Note 生成、RAG 回答 | 质量优先，全文推理 |
| `LIGHT_MODEL_ID` | Metadata 提取、RAG Judge、章节匹配 | 速度优先，< 512 tokens 输出 |

环境变量为 `LLM_MODEL`（旧文档可能称为 `MODEL_ID`）。`LIGHT_MODEL_ID` 未设置时回退到 `LLM_MODEL`。推荐轻量模型：`Qwen/Qwen2.5-7B-Instruct`（速度快，适合简单结构化任务）。

### 5.8 章节结构映射

**实现：** `paper_vault/importer.py::_build_section_map()`

- chunker 为每个 chunk 标注所属章节后，`_build_section_map()` 扫描所有 chunks 构建章节→chunk 区间 JSON
- 存入 notes_index 的 `sections` 字段
- RAG 检索时，`_match_sections_batch()` 用轻量 LLM 匹配问题 → 相关章节标题
- 章节过滤器传入 `search_chunks_for_papers(sections=...)` 实现定向检索

### 5.9 导入错误容错

**实现：** `paper_vault/importer.py` + try/except

- 单篇论文的处理逻辑为独立函数 `import_one()`（原 `_import_one`，现为公开 API），`import_pdfs()` 循环中逐篇 try/except 调用
- 任一篇论文的 LLM 超时或 API 报错不影响其他论文
- 失败的论文记录文件名，结束时汇总：`X succeeded, Y failed: file1.pdf, file2.pdf`
- `import_one()` 同时被 CLI 和 Web 层复用，无代码重复

## 6. 语义检索

**实现：** `paper_vault/retriever/__init__.py::search_papers()`

支持两级索引：
- `notes_index` — paper-level，整篇笔记一条记录，用于快速定位论文
- `chunks` — passage-level，全文分块，用于检索技术细节

**流程：**
```
query → embed_texts([query]) → search_chunks(vector, top_k, where) → [{paper_id, chunk_idx, text, title, year, ...}]
```

**过滤条件构建：** `store.py::build_where_clause()` 根据参数构建 SQL where 子句：
- `year_from=2024` → `year >= 2024`
- `year_to=2023` → `year <= 2023`
- `author="name"` → `authors LIKE '%name%'`
- 多个条件 AND 连接

**距离度量：** 点积（等价余弦相似度，因向量已 L2 归一化），`_distance` 越小越相关。

### 6.1 RAG 问答（三级自适应 + 章节定向检索 + 用户覆盖）

**实现：** `paper_vault/rag/qa.py`

**架构（2026-06 重构）：** 核心管线拆分为明确的 stage 函数，不再使用生成器 hack 统一 streaming/non-streaming：

| 函数 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `_gather_context()` | Stage 1-3 统一入口：嵌入→搜索→过滤→判定→检索 | question, n_papers, ... | (status_msgs, context, answer_tokens) |
| `_search_and_filter()` | 嵌入 + 召回 + LLM 筛选 | question, n_papers, where [, query_vec] | notes_results |
| `_determine_detail()` | 判定细节深度 | question, notes_results, detail | (level, full_text) |
| `_match_sections_batch()` | 批量章节匹配 | question, notes_results | {paper_id: [sections]} |
| `_answer_tokens()` | token 预算计算 | n_papers, max_tokens | answer_tokens |
| `_format_context()` | 统一上下文拼装 | items, prefix, text_key | context_string |
| `_call_light_llm()` | 轻量 LLM 调用封装 | prompt, label, max_tokens | raw_response |

**Public API：**
| 函数 | 用途 | 返回方式 |
|------|------|----------|
| `ask()` | CLI 同步 Q&A | 直接 return 答案字符串 |
| `ask_stream()` | Web SSE 流式 Q&A | yield SSE 事件 |

两者均调用 `_gather_context()` 获取上下文，然后各自完成 LLM 调用。避免了原来 `_run_rag` 生成器中 `Streaming=False` 时靠 `StopIteration.value` 传回返回值的隐式行为。

**多轮对话支持（2026-06 新增）：**
- `ask()` / `ask_stream()` 接受可选 `session_id` 参数
- 有 session 时：加载会话 → 用轻量 LLM 改写问题（解析代词、补充上下文）→ 将压缩后的历史注入 QA_PROMPT 的 `{history}` 占位符 → 回答完成后自动记录本轮对话
- 历史注入前检查 token 预算（`CONTEXT_HISTORY_MAX_TOKENS`，默认 2000），超限时截断保留最后约 30 行

**Embedding 复用：** 问题嵌入在 `_gather_context()` 开头执行一次，通过 `query_vec` 参数传递给 `_search_and_filter()` 和 chunk 检索循环，避免原来两次独立调用。

**完整流程：**
```
用户提问
  → _gather_context()
       → embed_texts([question])  (一次)
       → _search_and_filter(query_vec=...)
            → search_notes(top_k=max(n*2, 10)) → 扩大召回
            → _call_light_llm(FILTER_PAPERS_PROMPT) → 筛选相关论文 (≤ n 篇)
       → _determine_detail() / 用户 -d 参数覆盖
            1 = notes 足够 → 直接用 notes 生成答案
            2 = 需要关键细节 → per_paper = max(5, ceil(chunk_count/8))
            3 = 需要大量细节 → per_paper = max(15, ceil(chunk_count/3))
            all = 全量检索   → per_paper = chunk_count (所有 chunks)
       → _match_sections_batch() → 批量章节匹配 (一次轻量 LLM 调用)
       → search_chunks_for_papers(sections=matched) → 定向检索 (复用 query_vec)
       → _format_context() → 统一拼装 notes + chunks context
       → 返回 (status_msgs, context, answer_tokens)
  → ask() / ask_stream()
       → LLM(QA_PROMPT) → 回答 (同步返回 / SSE 流式)
```

**Judge 三级机制：**
- `NEED_DETAILS_PROMPT`：将问题 + notes 内容发给 LLM，评估需要多少额外细节
- 返回 1（notes 足够）、2（中等）、3（大量）
- 使用 `LIGHT_MODEL_ID`，`max_tokens` 由 `config.RAG_JUDGE_MAX_TOKENS` 控制（默认 10）
- 解析失败时默认回退到 2

**用户 detail 覆盖（CLI: `-d`/`--detail`）：**
- `-d 1`：跳过 judge，仅用 notes（最省 token）
- `-d 2`：跳过 judge，中等 chunks
- `-d 3`：跳过 judge，大量 chunks
- `-d all`：跳过 judge 和章节匹配，拉取全部 chunks（全文检索）
- 默认 `auto`：LLM judge 自动判定

**论文相关性筛选（`_search_and_filter`）：**
- 初始检索 2x n_papers（至少 10 篇）→ 轻量 LLM 选择相关论文 → 保留 ≤ n_papers 篇
- 解决向量检索 top-k 截断导致遗漏的问题（例如问题覆盖 10 篇论文但 top-5 只召回部分）
- LLM JSON 解析失败时回退到原始 top-n（通过 `parse_llm_json` 工具函数统一处理）

**chunk 分配公式（参数由 config 集中管理）：**
| 等级 | 公式 | 60-chunk 论文 | 含义 | 配置项 |
|------|------|-------------|------|--------|
| 1 | 0 | 0 | notes only | — |
| 2 | max(MIN, ceil(N/DIV)) | ~8 | 关键细节 | `RAG_DETAIL_MODERATE_MIN`=5, `RAG_DETAIL_MODERATE_DIVISOR`=8 |
| 3 | max(MIN, ceil(N/DIV)) | ~20 | 大量细节 | `RAG_DETAIL_EXTENSIVE_MIN`=15, `RAG_DETAIL_EXTENSIVE_DIVISOR`=3 |
| all | N | 60 | 全量 chunks | — |

**章节定向检索：**
- 每篇笔记的 `sections` 字段存储章节→chunk 区间映射
- `_match_sections_batch()` 用一次轻量 LLM 调用为所有论文匹配章节（避免 N 次串行调用）
- 匹配成功 → 只检索相关章节的 chunks
- 匹配失败或返回空 → 回退全文 chunk 检索
- `-d all` 时跳过章节匹配，直接拉全部 chunks
- 日志区分：`[RAG route: notes + chunks (8 detail chunks, 8/paper [moderate], section-targeted)]`

**Context 拼装（`_format_context` 统一处理）：**
- 检索结果按 (paper_id, chunk_idx) 排序，保证同一论文的 chunks 出现在一起且按原文顺序
- Notes 标注 `[Source: notes/{paper_id} | Paper: ...]`
- Chunks 标注 `[Detail: {paper_id}/chunk_{idx} | Paper: ... | Section: ...]`

**LLM 调用链路分析：**

一次 RAG 问答（Web UI 多轮对话，detail=auto，level≥2）涉及以下调用：

| # | 调用 | 类型 | 模型 | 耗时 | 触发条件 |
|---|------|------|------|------|----------|
| 1 | `_preprocess_question` 问题改写 + 搜索词扩展 | LLM API | 轻量 | 较短 | 有 session 时 |
| 2 | `paper_filter` 论文相关性筛选 | LLM API | 轻量 | 较短 | 候选论文 > n_papers 时 |
| 3 | `detail_judge` 深度判断 | LLM API | 轻量 | 中等 | detail=auto 时（默认） |
| 4 | `section_match` 章节匹配 | LLM API | 轻量 | 中等 | level≥2 且非 full_text 时 |
| 5 | `answer_generation` 答案生成 | LLM API | 主模型 | **较长** | 总是 |
| 6 | `round_summary` 本轮摘要 | LLM API | 轻量 | 较短 | 有 session 时 |
| 7 | `history_compact` 历史压缩 | LLM API | 轻量 | 中等 | 完整轮次 > SESSION_KEEP_FULL_ROUNDS 时（偶发） |
| — | `embed_texts` 问题向量化 | 本地模型 | — | 很短 | 总是（CPU，零 API 费用） |
| — | `search_notes` / `search_chunks` | 本地向量检索 | — | 很短 | 总是（LanceDB，零 API 费用） |

其中 #3 和 #4 通过后台线程**并行执行**（#4 在 Stage 1 结束后立即启动，与 #3 的 LLM 调用同时进行），#4 的实际耗时被 #3 覆盖。LLM API 调用有效串行阶段为 5 个（#1 → #2 → #3∥#4 → #5 → #6）。

**各场景调用数：**

| 场景 | LLM API 调用数 | 跳过项 |
|------|---------------|--------|
| Web UI 多轮，detail=auto，level≥2 | 6（#7 偶发 +1） | — |
| Web UI 多轮，detail=1（notes only） | 4 | 跳过 #4 |
| CLI ask，无 session | 4 | 跳过 #1 #6 #7 |
| 首轮对话，无历史 | 4 | 跳过 #1 #6 #7（#7 永不触发） |

**调用类型说明：**

| 类型 | 说明 | 费用 |
|------|------|------|
| LLM API | 通过 OpenAI 兼容 API 调用远程大语言模型，延迟取决于模型速度和网络 | 按 token 计费 |
| 本地模型 | SentenceTransformer 在本地 CPU/GPU 运行 embedding，单次 ~100ms | 零 |
| 本地向量检索 | LanceDB 嵌入式向量数据库，近似搜索 | 零 |

**首 token 延迟瓶颈：** 用户发送问题后，需等待 #1 → #2 → #3∥#4 全部完成后，才开始 #5 的流式输出。其中 #3（detail_judge）因将全部 notes 文本送入 LLM（~2000 input tokens），是预处理阶段最重的单次调用。整个预处理阶段约贡献总延迟的 20-25%，#5 答案生成为 70-75%。

**QA Prompt 优化：**
- 区分 [Paper:]（背景理解）和 [Detail:]（事实依据）的权重
- 对比类问题建议用表格
- 显式要求说"不知道"：`当前知识库中的论文未涉及此问题的足够信息`
- 引用格式：`【Title (Year)】`
- 回答末尾附 Sources 段落
- 自检要求：每条断言必须能在 excerpts 中找到依据

**动态参数：**
- `n_papers`（CLI: `-n`/`--notes`）：notes 检索数量，默认 5
- `chunks_per_paper`（CLI: `--chunks`）：每篇 chunk 上限，None 时按等级自动计算
- `detail`（CLI: `-d`/`--detail`）：`auto`(默认)、`1`、`2`、`3`、`all`
- `max_tokens`（CLI: `--max-tokens`）：LLM 回答最大 token 数，默认 auto = 1篇→1024, 2篇→2048, 3+篇→3072

**共享工具（`paper_vault/utils.py`）：**
- `get_llm_client()` — 懒加载 OpenAI 客户端单例，全项目共享
- `safe_format()` — 模板格式化，用 `str.replace` 替代 `str.format`，避免 LaTeX 大括号冲突
- `parse_llm_json()` — LLM 返回 JSON 解析，统一处理 ``` 标记剥离和 `json.JSONDecodeError`
- `strip_code_fences()` — 通用 ``` 围栏剥离，同时用于 JSON 解析和笔记生成
- `sanitize_part()` — 文件名安全化处理（原 CLI 私有函数 `_sanitize_part`，提升为公共工具）
- `normalize_id()` — paper_id 标准化（截断、去非法字符、去重下划线）
- `emit_sse()` — SSE 事件格式化，CLI 和 Web 共用同一格式
- `get_llm_client()` — 内置 `httpx.Timeout(120.0, connect=15.0)`，防止 API 不可达时无限挂起（之前默认 600s 无超时导致 RAG 在 "Searching papers" 阶段卡死）

### 6.2 多轮对话会话管理（session.py）

**实现：** `paper_vault/rag/session.py`

**为什么需要会话管理：**
- 单轮 Q&A 无法处理追问（"那第二篇呢？"、"对比一下这两篇的方法"），LLM 缺少上下文无法解析代词
- 多轮对话历史会持续膨胀，需要压缩机制控制上下文长度，避免超出 LLM 窗口或 token 预算

**Turn/Session 数据结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `Turn.role` | str | "user" / "assistant" |
| `Turn.question` | str | 用户原始问题 |
| `Turn.rewritten_question` | str | 经 LLM 改写后的自包含问题 |
| `Turn.answer` | str | LLM 回答（压缩后为空字符串） |
| `Turn.summary` | str | 本轮对话 LLM 摘要（压缩后填充完整回答的摘要） |
| `Turn.cited_papers` | list[str] | 引用的论文 ID |
| `Turn.timestamp` | float | Unix 时间戳 |
| `Session.id` | str | 时间戳字符串（`session_{int(time.time())}`） |
| `Session.name` | str | 会话名称 |
| `Session.created_at` | float | 创建时间（Unix 时间戳） |
| `Session.updated_at` | float | 最后更新时间（Unix 时间戳） |
| `Session.turns` | list[Turn] | 对话轮次列表 |
| `Session.compact_count` | int | 已执行压缩的次数 |

**三级渐进式压缩策略：**

| 级别 | 触发条件 | 操作 | 说明 |
|------|----------|------|------|
| 1 — 滑动窗口 | 轮数 > `SESSION_KEEP_FULL_ROUNDS` | 保留最近 N 轮完整，更早轮次的 assistant 回答被压缩为 LLM 摘要 | 最轻量，大部分场景触发此级 |
| 2 — 全量压缩 | Level 1 后 token 仍超预算 | 所有旧轮次合并为一段 LLM 摘要 | 中等压缩 |
| 3 — 预算预检 | 压缩后 token 仍超 `CONTEXT_HISTORY_MAX_TOKENS` | 截断保留最后 ~30 行 | 硬截断，兜底保护 |

**压缩判定逻辑（`_is_compacted`）：**
- 检测模式：`role == 'assistant' and answer == '' and summary != ''`
- 正常轮次：answer 和 summary 均有内容
- 已压缩轮次：answer 清空，summary 填充 LLM 生成的摘要
- 优点：自然区分压缩/未压缩轮次，无需额外标记字段或 magic string

**问题改写 + 搜索词扩展（`_preprocess_question` in qa.py）：**
- `qa.py` 使用 `_preprocess_question()` 合并改写 + 搜索词提取为单次 LLM 调用，替换旧版 `rewrite_question`
- 将最近 6 轮对话（3 个 Q&A 对）上下文发给轻量 LLM
- LLM 将代词（"它"、"第二篇"、"该方法"）解析为明确指代，同时提取 3-5 个搜索关键词
- Prompt 模板：`QUESTION_PREPROCESS_PROMPT`
- 改写失败时回退到原始问题
- `session.py` 中 `rewrite_question` 为独立函数，仅用于 session 层面改写（非 RAG 管道内）

**历史注入（`build_history_for_prompt`）：**
- 格式化为三段：`## Earlier conversation (summarized)` → 已压缩旧轮次 → `## User` / `## Assistant` → 最近完整轮次
- 注入前估算 token 数（`estimate_tokens = max(len(text) // 3, 1)`），超 `CONTEXT_HISTORY_MAX_TOKENS` 时截断

**回答后处理（`after_answer`）：**
- 对本轮 assistant 回答调用 LLM 生成 round summary（`ROUND_SUMMARY_PROMPT`）
- 追加 user turn + assistant turn（含 summary）到 session
- 触发 `_compress_to_summary()` 检查是否需要压缩
- 持久化保存到 `vault/sessions/{session_id}.json`

**持久化：** 每个 session 保存为独立 JSON 文件，CRUD 操作均为文件系统级别，不依赖数据库。

**新增 Prompt 模板：**

| Prompt | 用途 | 调用模型 |
|--------|------|----------|
| `QUESTION_REWRITE_PROMPT` | 将追问改写为自包含问题 | 轻量模型 |
| `ROUND_SUMMARY_PROMPT` | 为单轮回答生成摘要 | 轻量模型 |
| `HISTORY_FULL_COMPACT_PROMPT` | 将多轮历史合并为一段摘要 | 轻量模型 |

### 6.3 导入逻辑抽取（importer.py）

**实现：** `paper_vault/importer.py`

将原来耦合在 `cli/import_cmd.py` 中的核心导入逻辑抽取为独立 service 层模块，CLI 和 Web 层共同引用：

| 函数 | 可见性 | 用途 |
|------|--------|------|
| `import_one()` | public | 单篇 PDF 导入（文本提取 → 元数据 → 笔记 → 索引，含内容哈希去重） |
| `import_pdfs()` | public | 批量导入入口（文件名去重 + 内容哈希去重、错误隔离、汇总） |
| `_make_note_filename()` | private | 笔记文件名生成（信任 LLM 输出的 title，venue 简单长度校验） |
| `_build_section_map()` | private | 章节→chunk 区间映射 |

`cli/import_cmd.py` 缩减为一行转发：`from ..importer import import_pdfs`。`web/app.py` 从 `..importer` 导入 `import_one`。消除了 Web→CLI 反向依赖。

## 7. CLI

**实现：** `paper_vault/cli/`

```
pv.py import [paths] [--no-llm] [--no-index] [--force]
pv.py search <query> [-k N] [--year-from] [--year-to] [--author]
pv.py ask <question> [-n N] [-d auto|1|2|3|all] [--chunks N] [--max-tokens N] [--year-from] [--year-to] [--author] [--session ID] [--continue]
pv.py serve [--host] [-p PORT] [--no-open]
pv.py list
pv.py fix-metadata <paper_ids> [--all]
pv.py remove <paper_id>
pv.py session list
pv.py session new [--name NAME]
pv.py session delete <session_id>
pv.py session show <session_id>
```

- `--force`：强制重新导入已索引论文
- `-n`/`--notes`：RAG 检索论文数（默认 5）
- `--chunks`：每篇论文最大 chunk 数（默认 auto）
- `--session`/`--continue`：指定会话 ID 或继续最近会话，启用多轮对话
- `paths` 为空时自动扫描 `PAPER_VAULT_IMPORT_DIRS`
- `fix-metadata`：纯 LLM 重新提取标题/作者/年份，原地更新 LanceDB 索引 + 同步重命名笔记文件，无需重新导入 PDF。指定 paper_id 或 `--all` 修复全部论文
- `session` 子命令：管理多轮对话会话（list/new/delete/show）

## 8. Token 用量跟踪

**实现：** `paper_vault/usage.py::UsageTracker`

- 全局单例 tracker，每次进程启动独立计数
- 所有 LLM 调用（note 生成、元数据提取、RAG judge、RAG 回答）均自动上报
- `summary()` 输出：调用次数 + prompt tokens + completion tokens
- 不做费用估算（不同 API 价格不同）

## 9. 配置

**实现：** `paper_vault/config.py::Config`

- `.env` 只存放 API 密钥和模型配置（不存路径配置）
- 路径有合理默认值（`./vault`, `./papers`），可通过环境变量覆盖
- 首次实例化时自动创建所需目录

**完整配置项：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_BASE_URL` | (必填) | API endpoint |
| `OPENAI_API_KEY` | (必填) | API 密钥 |
| `LLM_MODEL` | (必填) | 主模型（笔记生成、RAG 回答） |
| `LIGHT_MODEL_ID` | 同 LLM_MODEL | 轻量模型（元数据提取、Judge、章节匹配） |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | 本地 Embedding 模型 |
| `PAPER_VAULT_DIR` | `./vault` | 数据存储根目录 |
| `PAPER_VAULT_IMPORT_DIRS` | (环境变量或 settings.json) | 默认导入路径（`:` 分隔） |
| `PAPER_VAULT_MAX_PDF_MB` | 不限 | CLI/Web 导入 PDF 大小上限 (MB) |
| `PAPER_VAULT_MAX_UPLOAD_MB` | 100 | Web 上传大小上限 (MB) |
| `PAPER_VAULT_CHUNK_SIZE` | 800 | 文本分块大小 (chars) |
| `PAPER_VAULT_CHUNK_OVERLAP` | 100 | 块间重叠 (chars) |
| `PAPER_VAULT_METADATA_SNIPPET_CHARS` | 6000 | 元数据提取送入 LLM 的文本长度 |
| `PAPER_VAULT_META_EXTRACT_MAX_TOKENS` | 512 | 元数据 LLM 调用 max_tokens |
| `PAPER_VAULT_NOTE_GEN_TEMPERATURE` | 0.3 | 笔记生成 temperature |
| `PAPER_VAULT_NOTE_GEN_MAX_TOKENS` | 4096 | 笔记生成 max_tokens |
| `PAPER_VAULT_NOTE_FILENAME_MAX_LEN` | 150 | 笔记文件名最大长度 |
| `PAPER_VAULT_DEDUP_HASH_CHARS` | 5000 | 内容去重哈希所用文本前缀长度 |
| `PAPER_VAULT_RAG_QA_TEMPERATURE` | 0.3 | RAG 回答 temperature |
| `PAPER_VAULT_RAG_FILTER_MAX_TOKENS` | 256 | 论文筛选 LLM max_tokens |
| `PAPER_VAULT_RAG_JUDGE_MAX_TOKENS` | 10 | 细节判定 LLM max_tokens |
| `PAPER_VAULT_RAG_SEARCH_BREADTH_MIN` | 10 | 检索扩召最小候选数 |
| `PAPER_VAULT_RAG_DETAIL_MODERATE_DIVISOR` | 8 | 中等细节 chunk 除数 |
| `PAPER_VAULT_RAG_DETAIL_EXTENSIVE_DIVISOR` | 3 | 大量细节 chunk 除数 |
| `PAPER_VAULT_RAG_DETAIL_MODERATE_MIN` | 5 | 中等细节最小 chunk 数 |
| `PAPER_VAULT_RAG_DETAIL_EXTENSIVE_MIN` | 15 | 大量细节最小 chunk 数 |
| `PAPER_VAULT_RAG_DEFAULT_CHUNK_COUNT` | 50 | 未知 chunk_count 时的默认值 |
| `PAPER_VAULT_ANSWER_TOKENS_1` | 1024 | 单篇回答 token 预算 |
| `PAPER_VAULT_ANSWER_TOKENS_2` | 2048 | 两篇回答 token 预算 |
| `PAPER_VAULT_ANSWER_TOKENS_3` | 3072 | 三篇及以上回答 token 预算 |
| `PAPER_VAULT_SESSION_KEEP_FULL` | 3 | 会话保留完整内容的最近轮数 |
| `PAPER_VAULT_HISTORY_MAX_TOKENS` | 2000 | 注入 QA prompt 的历史 token 预算上限 |

**Web 设置持久化：** 可编辑配置项通过 Web UI 设置页面修改后保存至 `vault/settings.json`，下次启动自动加载。环境变量优先级高于 settings.json。

**废弃字段：** `PDFS_DIR` — 已移除（从未使用）。 `is_plausible_title()` — 已移除（随启发式元数据提取器一同删除）。

## 10. 项目打包

**实现：** `pyproject.toml`

- `pip install -e .` 一键安装所有依赖
- 注册 CLI 入口 `pv` → `paper_vault.cli.main:main`
- 依赖版本约束：pymupdf>=1.23, openai>=1.0, sentence-transformers>=3.0, lancedb>=0.6, pyarrow>=14, numpy>=1.24
- 已移除 pandas 依赖（全部改用 LanceDB 原生 API）
- Python >= 3.10

## 11. Web UI

**实现：** `paper_vault/web/app.py` (FastAPI) + `index.html` (单页应用)

**启动：**
```
python -m uvicorn paper_vault.web.app:app --host 127.0.0.1 --port 8080
# 或
python pv.py serve [--host 127.0.0.1] [-p 8080]
```

**安装：** `pip install -e ".[web]"`（额外安装 fastapi + uvicorn + python-multipart）

**架构：** 一个 HTML 文件包含全部前端逻辑（Vanilla JS），不依赖 npm/React。

**前端渲染：**
- marked.js (CDN) — Markdown → HTML
- KaTeX (CDN) — LaTeX 公式渲染（`$...$` / `$$...$$`）

**页面布局：**
- 左侧 280px 侧边栏：笔记目录、PDF 拖拽上传、RAG 问答面板
- 右侧主区域：Markdown 笔记阅读 / 问答结果显示

**API 端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/papers` | 已索引论文列表（仅元数据，不含内容） |
| GET | `/api/notes/{id}` | 论文笔记内容（前端按需加载，支持 `?filename=` 直接查找） |
| DELETE | `/api/papers/{paper_id}` | 删除论文（索引 + 笔记文件 + 提取文件） |
| PUT | `/api/papers/{paper_id}` | 更新论文标题 |
| PUT | `/api/papers/{paper_id}/note` | 保存笔记内容并重新 embedding |
| POST | `/api/papers/{paper_id}/reindex-note` | 从磁盘重新读取笔记并重新 embedding |
| POST | `/api/import` | 上传 PDF → 提取 + 笔记 + 索引（SSE 进度流，含上传大小限制） |
| POST | `/api/import-scan` | 扫描配置的导入目录并导入全部 PDF（SSE 进度流） |
| POST | `/api/search` | 语义搜索（按 paper_id 去重） |
| POST | `/api/ask` | RAG 问答（SSE 流式，支持 `session_id` 多轮对话） |
| GET | `/api/settings` | 获取当前配置（只读项 + 可编辑项） |
| POST | `/api/settings` | 保存可编辑配置到 `vault/settings.json` |
| GET | `/api/sessions` | 列出所有会话（id, name, turns, created_at, updated_at, compact_count） |
| POST | `/api/sessions` | 创建新会话（name 可选） |
| GET | `/api/sessions/{id}` | 获取会话详情（含全部 turns） |
| PUT | `/api/sessions/{id}` | 重命名会话 |
| DELETE | `/api/sessions/{id}` | 删除指定会话 |
| GET | `/api/browse` | 浏览目录（文件夹选择器用） |
| POST | `/api/browse-native` | 打开操作系统原生文件夹选择器 |
| GET | `/api/prompts` | 获取用户可编辑的 prompt 模板 |
| POST | `/api/prompts` | 保存 prompt 覆盖到 `vault/prompts.json` |
| POST | `/api/cmd` | 执行 CLI 风格命令（import/search/ask/list/fix-metadata/remove） |
| POST | `/api/clean-duplicates` | 清理 LanceDB 中的重复论文条目 |
| POST | `/api/reindex-orphans` | 重新索引磁盘上存在但不在 LanceDB 中的笔记文件 |

**性能优化（2026-06）：**
- `/api/papers` 不再支持 `include_content`，始终只返回元数据。前端点击论文时通过 `/api/notes/{id}?filename=...` 按需加载单篇内容，避免全量加载导致 OOM
- `_build_note_map()` 增加模块级缓存，导入完成后自动失效
- SSE 导入进度流包含 keepalive 注释，防止长连接超时
- 所有业务逻辑直接调用 `paper_vault.*` 模块，零重复代码

**Tab 多面板系统（2026-06 重构）：**
- 原有的单一 `#view-area` 替换为 `#tab-container`（`#tab-bar` + `#tab-panels`）
- 每个 tab 对应一个持久 DOM panel（`createElement` / `appendChild`），切换 tab 仅切换 CSS `.active` 类，从不重建 DOM
- 核心函数：
  - `createTab(type, title, opts)` — 创建 tab，笔记类按 paperId 去重（重复则切换到已有 tab），创建持久 panel 并填入内容
  - `closeTab(id)` — 移除 panel DOM，切换到相邻 tab
  - `switchTab(id)` — 仅切换 `.active` 类，无 DOM 重建
  - `renderTabBar()` — 仅更新 tab 栏 UI
- 设计动机：之前用 `innerHTML` 全文重建 `#view-area`，导致无法同时打开多篇笔记，也无法在 Q&A 等待时查看其他笔记。持久 panel 解决了这些问题，SSE 流式 token 可在后台 tab 正常接收

**Prompt 暴露控制（2026-06）：**
- Web 设置页面仅暴露用户关心的 2 个 prompt：`note_prompt`（笔记生成）和 `qa_prompt`（RAG 回答）
- 内部 pipeline prompt 不暴露：过滤、判定、章节匹配、问题改写、摘要生成等 7 个 prompt 均为只读
- 后端通过 `_USER_PROMPT_KEYS = {"note_prompt", "qa_prompt"}` 白名单控制 `/api/prompts` GET/POST

**API 密钥安全：**
- 设置页面不暴露 `LLM_API_KEY`，仅显示引导文字提示用户在 `.env` 中配置
- `LLM_BASE_URL` 同样不通过 Web UI 暴露

**会话聊天 UI（2026-06 新增）：**
- 侧边栏 Session 下拉选择器 + 新建按钮 + 自动刷新轮次计数
- 点击会话 → 打开 session 类型 tab，展示完整对话历史（聊天气泡样式：用户消息靠右蓝色，助理消息靠左深色，已压缩历史显示为灰色摘要）
- 每个 session tab 底部有内联输入框（Ctrl+Enter 发送），支持在该 session 内连续追问
- 侧边栏 Q&A 面板提问时，若当前有活跃 session tab，自动追加到聊天窗口而非新建 QA tab
- Command 输入框 ask 命令同样自动路由到活跃 session tab
- Tab 关闭时自动清除 session 选中状态，切换 session tab 时同步选择器

**设置页面（2026-06 新增）：**
- 侧边栏标题旁 ⚙ 按钮进入设置界面
- 7 个设置分组：Paths（4 个独立路径）、Models（LLM_MODEL / LIGHT_MODEL_ID / EMBEDDING_MODEL）、Import Limits、Indexing、Metadata Extraction、Note Generation、RAG Q&A、Answer Token Budget
- 每项参数附带描述说明（小字），解释该参数的具体影响和生效方式
- 路径字段旁提供 Browse 按钮，弹出文件夹选择器（`/api/browse`），支持目录导航选取
- 保存后写入 `vault/settings.json`，调用 `config.reload_settings()` 立即生效
- 点击论文或 Q&A 时自动退出设置视图

**配置架构（2026-06 重构）：**
- 模块级 `_get(key, env, default)` helper：先查 `settings.json`（null-safe，key 存在且 value 不为 None 才使用），再查环境变量，最后用默认值
- `_get_int` / `_get_float` 包装器，涵盖全部 30+ 可配字段
- Config 类体属性 + `reload_settings()` 均通过 `_get` 模式读取，避免 `dict.get()` 对 JSON null 的语义差异（key 存在但值为 null 时不使用默认值）
- 硬件路径（`VAULT_DIR`, `EXTRACTED_DIR`, `NOTES_DIR`, `VECTORS_DIR`, `MODELS_DIR`）各自独立配置，可通过 Browse 选取

**后端复用：** 所有 API 直接调用 `paper_vault.*` 模块，零业务逻辑重复。SSE 事件格式化通过 `utils.emit_sse()` 统一，`web/app.py` 和 `rag/qa.py` 共用同一格式。

---

*最后更新：2026-06-16（LLM 调用链路分析、#1+#2 合并/并行 #4、context 去重、空检索兜底、多轮语义扩展、笔记编辑与 reindex、Web UI settings toggle/detail 选择器/参数自定义输入）*
