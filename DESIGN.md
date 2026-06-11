# Paper Vault - Design Document

> 个人论文/笔记管理与阅读助手 | 本地优先 · 终端驱动 · AI 赋能

## 1. 项目定位

### 1.1 一句话描述

一个终端驱动的本地论文知识库管理工具，支持 PDF 解析为 Markdown 笔记、向量化存储、RAG 问答，以及基于关键词和摘要的论文关系图谱。

### 1.2 核心痛点

- **读过就忘**：三个月前精读的论文，想不起来核心思路和关联
- **论文间的关系靠人脑记**：缺少自动化的关联和检索手段
- **管理分散**：PDF 在文件夹、笔记在 Obsidian、问答靠 ChatGPT，三者割裂
- **通用 AI 工具不持久**：Claude/GPT 对话结束上下文就丢失，不记住你的知识库

### 1.3 与通用 AI 工具的区别

| | Claude Code / ChatGPT | Paper Vault |
|---|---|---|
| 论文库持久化 | 对话结束即丢失 | 永久存储，增量积累 |
| 结构化元数据 | 无 | 标题、作者、年份、关键词、摘要 |
| 知识图谱 | 无 | 论文间关联关系可查可浏览 |
| 批量处理 | 逐篇对话 | 批量导入、批量索引 |
| 数据可迁移 | 无 | 纯文件存储，工具无关 |

### 1.4 目标用户

面向所有人，但作为个人首个开源项目，设计上追求简单实用，不过度设计用户系统和多租户。

---

## 2. MVP 功能范围

### 2.1 第一期（MVP）

1. **PDF 导入与解析**
   - 单篇 PDF 拖入 → 自动解析为 Markdown 笔记，存入笔记目录
   - 支持多 PDF 批量导入
   - 自动提取元数据（标题、作者、年份、摘要、关键词）

2. **向量化存储与检索**
   - Markdown 笔记分块 → 本地 Embedding → 向量数据库
   - 支持关键词检索 + 语义相似度检索

3. **RAG 问答**
   - 基于论文库内容的问答（单篇 / 跨文档）
   - 云端 LLM 主力 + 本地 LLM 兜底

4. **CLI 核心 + 轻量 Web UI**
   - CLI：导入、检索、管理操作
   - Web UI：笔记阅读 + 问答交互

### 2.2 第二期

5. **论文关系图谱**
   - 基于摘要和关键词的论文相似度关联
   - 引用关系提取
   - 简单的关系图谱可视化

### 2.3 远期（探索性）

6. 网络检索集成（arXiv API / Semantic Scholar）
7. 文献调研辅助（给定方向 → 搜索 + 筛选 + 汇总）
8. 阅读进度追踪与笔记标注

---

## 3. 技术架构

### 3.1 本地/云端分层策略

| 组件 | 部署位置 | 选型 | 理由 |
|------|---------|------|------|
| PDF 解析 | **本地** | Marker / PyMuPDF | 一次解析永久使用，离线可用 |
| Embedding 模型 | **本地** | BGE-small / GTE-small | 免费、离线、够用 |
| 向量数据库 | **本地** | LanceDB / ChromaDB | 嵌入式零运维 |
| LLM 推理 | **云端为主 + 本地兜底** | Claude API / Qwen2.5-7B 本地 | 云端质量最高，本地保证离线基础可用 |

### 3.2 技术栈（建议）

```
语言：        Python >= 3.10
PDF 解析：     Marker (首选) / PyMuPDF (降级)
Embedding：   sentence-transformers (BGE/GTE)
向量数据库：   LanceDB
LLM：         anthropic SDK / openai SDK / ollama (本地)
Web UI：      FastAPI + 极简 HTML/JS 或 Streamlit
图存储：      JSON 文件 (MVP) → KùzuDB (后期)
包管理：      uv / pdm
```

### 3.3 系统架构图

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Web UI                      │
├─────────────────────────────────────────────────────┤
│  PDF 导入模块  │  检索模块  │  RAG 问答  │  图谱模块  │
├─────────────────────────────────────────────────────┤
│  PDF 解析器    │  Embedding  │  LLM 调用  │  图计算    │
│  (Marker)     │  (BGE)     │  (Claude) │  (NetworkX)│
├─────────────────────────────────────────────────────┤
│  文件系统 (Markdown 笔记)  │  LanceDB (向量索引)      │
└─────────────────────────────────────────────────────┘
```

---

## 4. 数据组织

### 4.1 目录结构

```
~/paper-vault/
├── notes/                  # Markdown 笔记，一篇论文一个 .md 文件
│   ├── 2017-attention-is-all-you-need.md
│   └── 2023-llama-open-and-efficient.md
├── pdfs/                   # 原始 PDF（可选保留）
│   ├── attention.pdf
│   └── llama.pdf
├── vectors/                # LanceDB 向量索引数据
├── graph.json              # 论文关系图谱数据（第二期）
└── config.yaml             # 用户配置（LLM API key、模型选择等）
```

### 4.2 Markdown 笔记格式

```markdown
---
title: "Attention Is All You Need"
authors: ["Vaswani, Ashish", "Shazeer, Noam", ...]
year: 2017
arxiv_id: "1706.03762"
keywords: ["transformer", "attention", "sequence-to-sequence"]
source_pdf: "attention.pdf"
imported_at: "2026-06-09"
---

# Attention Is All You Need

## Abstract
...

## 1. Introduction
...
```

### 4.3 设计原则

- **数据透明**：所有笔记以纯 Markdown 文件存储在磁盘，不锁定在专有数据库
- **工具无关**：即使用户不再使用本工具，笔记依然可用任何编辑器打开
- **增量可迁移**：向量索引和图谱可重建（源数据是 Markdown 文件）

---

## 5. 模块划分

### 5.1 核心模块

```
paper_vault/
├── cli/                    # CLI 命令入口
│   ├── __init__.py
│   ├── main.py            # 主命令组
│   ├── import_cmd.py      # 导入 PDF 命令
│   ├── search_cmd.py      # 检索命令
│   └── ask_cmd.py         # 问答命令
├── parser/                 # PDF 解析
│   ├── __init__.py
│   └── extractor.py       # PyMuPDF 文本提取
├── notes/                  # 笔记生成与元数据提取
│   ├── __init__.py
│   ├── generator.py       # LLM 笔记生成 + 元数据提取
│   └── prompts.py         # Prompt 模板
├── indexer/                # 向量化索引
│   ├── __init__.py
│   ├── chunker.py         # 文本分块
│   ├── embedder.py        # Embedding 生成
│   └── store.py           # 向量数据库操作 (LanceDB)
├── retriever/              # 检索
│   └── __init__.py        # 语义检索 (embed + search)
├── rag/                    # RAG 问答
│   ├── __init__.py
│   ├── qa.py              # 核心管线 + 步骤函数
│   └── prompts.py         # QA/判定/匹配 Prompt 模板
├── web/                    # Web UI
│   ├── __init__.py
│   ├── app.py             # FastAPI 应用 (SSE 流式)
│   └── index.html         # 单页前端 (Vanilla JS)
├── utils.py                # 共享工具 (safe_format, parse_llm_json, emit_sse)
├── config.py               # 配置管理
└── usage.py                # Token 用量跟踪
```

---

## 6. 关键设计决策（待确认）

### 6.1 PDF 解析策略

- **首选**：Marker（输出质量好，对公式/表格/双栏友好）
- **降级**：PyMuPDF + 清洗脚本（轻量，但需要较多后处理）
- **需验证**：中文论文的解析效果

### 6.2 文本分块策略

- 论文有天然的分段结构（章节），应优先按章节边界分块
- 块大小建议 500-1000 tokens，重叠 100 tokens
- 保留章节标题作为块的上下文元数据

### 6.3 知识图谱粒度

- **MVP（第二期）**：论文-论文关系（相似度 + 引用），不涉及概念节点
- 相似度：基于摘要 + 关键词的 Embedding 余弦相似度
- 边权重阈值：可配置，低于阈值的关联不展示

### 6.4 Embedding 模型选择

- 英文论文：BAAI/bge-small-en-v1.5 (384 维，轻量)
- 中文/多语言：BAAI/bge-small-zh-v1.5 或 thenlper/gte-small

### 6.5 LLM 配置

- 默认使用云端 API（Claude / GPT），用户配置自己的 API Key
- 可选配置本地 Ollama 模型作为离线备选
- RAG Prompt 需要包含论文元数据（标题、作者）以增加上下文

---

## 7. 风险与边界

### 7.1 不做的事

- **不做实时协作**：单用户本地工具
- **不做插件系统**：MVP 阶段不扩展
- **不做移动端**：桌面/终端优先
- **不做网页剪藏/高亮标注**：聚焦论文处理和管理
- **不做复杂的用户系统**：单用户本地运行

### 7.2 技术风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| PDF 解析质量不稳定 | 笔记质量差，用户体验差 | Marker 作为首选，但准备 PyMuPDF 降级方案 |
| 本地 Embedding 中文效果差 | 中文论文检索不准 | 选用多语言 Embedding 模型 |
| LLM 幻觉导致图谱关系错误 | 知识图谱不可信 | 图谱基于确定性数据（Embedding 相似度），不依赖 LLM 抽取 |
| 向量数据库文件损坏 | 检索失败 | 源数据是 Markdown 文件，可随时重建索引 |

---

## 8. 开发计划（建议）

### Phase 1：基础设施（第 1-2 周）
- 项目脚手架、配置管理
- PDF 解析 + Markdown 笔记生成
- 元数据提取

### Phase 2：向量索引 + 检索（第 2-3 周）
- 文本分块 + Embedding
- 向量数据库集成
- 关键词 + 语义检索

### Phase 3：RAG 问答 + Web UI（第 3-4 周）
- LLM 调用封装
- RAG Prompt 设计
- 极简 Web UI

### Phase 4：关系图谱（第 5-6 周）
- 论文相似度计算
- 图谱数据生成
- 图形可视化

### Phase 5：打磨与发布
- 文档、README、示例
- 打包发布（pip installable）
- GitHub Release

---

## 9. 命名备选

- **Paper Vault** — 简洁，暗示存储与管理
- **PaperLink** — 侧重关联图谱
- **ReadVault** — 侧重阅读管理
- **BibBrain** — 轻松有趣
- **PaperStack** — 技术感

> 当前暂用 Paper Vault，最终命名待定。

---

*最后更新：2026-06-11 | 状态：开发阶段*
