# 图灵知识图谱工程

本项目围绕 Alan Turing 构建专题知识图谱，目标不是单次脚本演示，而是形成一条可复现、可审计、可替换数据源的工程化流水线。当前主线已经覆盖结构化种子库、非结构化语料预处理、CRF mention 识别、实体链接、规则共指、关系候选生成、事实抽取与最终图谱导出。

当前数据仍是样例批次，后续可以直接替换 `data/raw/` 与 `configs/` 中的数据源配置，保持下游数据契约不变。

## 一、项目目标 

围绕图灵相关人物、机构、地点、作品、机器、概念、奖项与事件，构建一个带证据来源的知识图谱。图谱同时保留两类信息：

| 层级 | 说明 | 主要产物 |
| --- | --- | --- |
| 结构化事实层 | 来自 Wikidata / Wikipedia 的高置信实体、别名、声明和事件候选 | `data/processed/structured/` |
| 文本证据层 | 从 HTML / PDF / 本地文档中抽取 mention、链接实体、生成关系候选并聚合事实 | `data/processed/mentions/`、`linking/`、`relations/`、`facts/` |
| 可视化导出层 | 合并结构化事实、事件和文本证据，导出 Graph JSON、Gephi CSV、Neo4j CSV 和离线 HTML | `data/processed/visualization/` |

## 二、项目结构

```text
knowledge_graph/
├─ configs/                         # 配置中心
│  ├─ seed_entities.json             # 结构化抓取种子，当前为 Alan Turing(Q7251)
│  ├─ structured_fetch_config.json   # Wikidata / Wikipedia 抓取配置
│  ├─ unstructured_sources.yaml      # HTML / PDF 非结构化语料清单
│  ├─ relation_patterns.yaml         # 事实抽取规则触发词
│  └─ relation_training_config.json  # 关系模型训练配置
├─ knowledge/
│  └─ ontology.json                  # 实体、关系、事件和约束定义
├─ scripts/
│  └─ turing_kg.py                   # 统一命令行入口
├─ src/
│  ├─ kg_core/                       # 路径、IO、schema、实体目录、报告等公共能力
│  ├─ structured_seed/               # Wikidata / Wikipedia 结构化种子库构建
│  ├─ unstructured_preprocess/       # HTML / PDF 清洗、文档归一化、句子切分
│  ├─ mention_crf/                   # 弱标注、特征工程、CRF 训练与预测
│  ├─ entity_linking/                # 候选生成、打分、消歧、断链分析
│  ├─ coreference/                   # 规则共指与 anchor 传播
│  ├─ relation_extraction/           # 实体对候选、远程监督、关系模型训练/预测
│  ├─ fact_extraction/               # 规则 + DS + 可选 LLM 校验的事实抽取
│  └─ visualization_export/          # 图谱 JSON、Gephi、Neo4j、HTML 导出
├─ data/
│  ├─ raw/                           # 原始数据，当前被 gitignore 忽略
│  └─ processed/                     # 流水线中间层与最终产物
├─ tests/                            # 单元测试与契约测试
└─ requirements.txt                  # Python 依赖
```

## 三、系统流程

```text
结构化种子实体
  -> Wikidata / Wikipedia 抓取
  -> entities / aliases / claims / event_candidates

非结构化语料
  -> HTML / PDF 文档清洗
  -> 句子切分与时间表达识别
  -> tokenized sentences

Mention 识别
  -> 弱监督标注
  -> CRF 训练
  -> mention 预测

实体链接与共指
  -> 候选召回
  -> 局部特征 + 文档上下文打分
  -> REVIEW / LINKED / NIL 决策
  -> 规则共指补全

关系与事实抽取
  -> resolved mention 生成实体对候选
  -> 结构化 claims 远程监督对齐
  -> 规则触发词生成 fact candidates
  -> 离线或 LLM 校验
  -> 聚合最终 facts

图谱导出
  -> 合并结构化 facts、event candidates、文本 facts
  -> 导出 graph.json / graph.html / Gephi CSV / Neo4j CSV
```

## 四、核心模块说明

### 4.1 本体层

本体文件位于 `knowledge/ontology.json`，当前定义：

| 类型 | 内容 |
| --- | --- |
| 实体类 | `Person`、`Organization`、`Place`、`Work`、`Concept`、`Machine`、`Award` |
| 事件类 | `BirthEvent`、`EducationEvent`、`EmploymentEvent`、`PublicationEvent`、`DeathEvent`、`ProposalEvent`、`DesignEvent`、`HonorEvent` |
| 关系类 | `BORN_IN`、`DIED_IN`、`STUDIED_AT`、`WORKED_AT`、`AUTHORED`、`PROPOSED`、`DESIGNED`、`AWARDED`、`LOCATED_IN` 等 |

本体的作用是统一实体类型、关系 domain/range、事件角色和基础约束，避免每个模块各自发明字段。

### 4.2 结构化种子库

入口模块：`src/structured_seed/`

当前以 `configs/seed_entities.json` 中的 Alan Turing `Q7251` 为种子，从 Wikidata 抓取实体基础信息、别名和关系声明，并继续回填一跳、两跳对象实体。Wikipedia summary 用于补充实体描述。

主要输出：

| 文件 | 说明 |
| --- | --- |
| `data/processed/structured/structured_kg.db` | SQLite 结构化底座 |
| `data/processed/structured/csv/entities.csv` | 实体表 |
| `data/processed/structured/csv/aliases.csv` | 别名表 |
| `data/processed/structured/csv/claims.csv` | 结构化事实声明 |
| `data/processed/structured/csv/event_candidates.csv` | 由声明派生的事件候选 |
| `data/processed/structured/structured_fetch.log` | 抓取日志 |

当前样例规模：

| 指标 | 数量 |
| --- | ---: |
| 实体 | 33 |
| 别名 | 114 |
| 结构化关系事实 | 30 |
| 事件候选 | 24 |

### 4.3 非结构化预处理

入口模块：`src/unstructured_preprocess/`

数据源由 `configs/unstructured_sources.yaml` 管理，当前包含 Bletchley Park、King's College Cambridge、Princeton、University of Manchester、Royal Society、Wikipedia、Britannica 等 HTML/PDF 样例来源。

预处理流程：

1. 读取 source registry。
2. 按 `source_type` 调用 HTML 或 PDF extractor。
3. 生成统一文档记录。
4. 清理文本并切分句子。
5. 抽取基础时间表达，为事件和事实 qualifiers 做准备。

主要输出：

| 文件 | 说明 |
| --- | --- |
| `data/processed/unstructured/documents.jsonl` | 文档级记录 |
| `data/processed/unstructured/documents.report.json` | 文档处理报告 |
| `data/processed/unstructured/sentences.jsonl` | 句子级记录 |
| `data/processed/unstructured/sentences.report.json` | 句子切分报告 |

当前样例规模：

| 指标 | 数量 |
| --- | ---: |
| 文档 | 11 |
| 句子 | 2383 |
| 处理错误 | 0 |

### 4.4 Mention 识别

入口模块：`src/mention_crf/`

当前 mention 识别不是直接套用 spaCy，而是走“弱监督标注 + 字典特征 + CRF”的可审计路线：

1. `prepare`：把句子转成 token 序列，并抽样生成金标模板。
2. `weak-label`：结合结构化实体和别名资源生成弱标注样本。
3. `train`：训练 CRF 模型。
4. `predict`：对全量句子预测 mention。

识别类型对齐本体层，覆盖 `PERSON`、`ORGANIZATION`、`PLACE`、`MACHINE`、`CONCEPT`、`WORK`、`AWARD` 等。

主要输出：

| 文件 | 说明 |
| --- | --- |
| `data/processed/mentions/tokenized_sentences.jsonl` | token 化句子 |
| `data/processed/mentions/weak_labeled.jsonl` | 弱标注训练样本 |
| `data/processed/mentions/weak_label_review_queue.jsonl` | 需要人工 review 的弱标注样本 |
| `data/processed/mentions/test_gold_template.jsonl` | 人工金标模板 |
| `data/processed/mentions/mentions.jsonl` | CRF 预测 mention，若已运行 predict |

### 4.5 实体链接与共指

入口模块：

- `src/entity_linking/`
- `src/coreference/`

实体链接以结构化实体表和别名表为候选空间，综合 exact alias、normalized alias、surname alias、abbreviation、TF-IDF recall、文档上下文等信号，输出 `LINKED`、`REVIEW`、`NIL` 三类决策。共指模块在链接结果基础上做规则解析和 anchor 传播。

当前 linking 样例结果：

| 指标 | 数量 |
| --- | ---: |
| mention 总数 | 2252 |
| LINKED | 228 |
| REVIEW | 806 |
| NIL | 1218 |
| 平均候选数 | 4.9947 |

主要输出：

| 文件 | 说明 |
| --- | --- |
| `data/processed/linking/linked_mentions.jsonl` | 实体链接结果 |
| `data/processed/linking/linking_review.jsonl` | 需要人工复核的链接 |
| `data/processed/linking/linking_gap_report.json` | 高频断链 mention 报告 |
| `data/processed/coreference/resolved_mentions.jsonl` | 共指解析后的 mention |
| `data/processed/coreference/coreference_report.json` | 共指报告 |

### 4.6 关系候选与事实抽取

入口模块：

- `src/relation_extraction/`
- `src/fact_extraction/`

关系候选先基于 resolved mention、句子上下文、实体类型和结构化 claims 生成实体对候选；事实抽取再使用关系触发词、远程监督信号、链接质量和可选 LLM 校验进行打分，最后按事实粒度聚合。

当前支持的文本事实关系主要包括：

| 关系 | 含义 |
| --- | --- |
| `BORN_IN` | 出生地 |
| `STUDIED_AT` | 就读机构 |
| `WORKED_AT` | 工作/任职机构 |
| `AUTHORED` | 作者-作品 |
| `PROPOSED` | 提出概念 |
| `LOCATED_IN` | 机构/地点归属 |

当前样例结果：

| 阶段 | 数量 |
| --- | ---: |
| 关系候选 | 115 |
| fact candidates | 91 |
| pattern 命中 | 22 |
| Wikidata 对齐候选 | 48 |
| 最终文本事实 | 10 |
| 冲突事实 | 0 |

主要输出：

| 文件 | 说明 |
| --- | --- |
| `data/processed/relations/pair_candidates.jsonl` | 实体对关系候选 |
| `data/processed/relations/distant_labeled.jsonl` | 远程监督弱标签，若运行 weak-label |
| `data/processed/facts/fact_candidates.jsonl` | 事实候选 |
| `data/processed/facts/fact_verified.jsonl` | 已打分/校验事实 |
| `data/processed/facts/facts_final.jsonl` | 最终文本事实 |
| `data/processed/facts/fact_conflicts.jsonl` | 冲突记录 |

### 4.7 图谱导出与可视化

入口模块：`src/visualization_export/`

导出层会合并三类边：

| 来源层 | 说明 |
| --- | --- |
| `facts` | Wikidata 结构化事实 |
| `event` | 从结构化声明派生的事件节点与事件论元边 |
| `evidence` | 从文本中抽取并聚合的最终事实 |

当前样例图谱规模：

| 指标 | 数量 |
| --- | ---: |
| 节点 | 66 |
| 边 | 103 |
| facts 边 | 30 |
| event 边 | 63 |
| evidence 边 | 10 |

主要输出：

| 文件 | 用途 |
| --- | --- |
| `data/processed/visualization/graph.json` | 前端或其他程序消费的图谱 JSON |
| `data/processed/visualization/graph.html` | 离线交互可视化页面 |
| `data/processed/visualization/nodes.csv` | Gephi 节点表 |
| `data/processed/visualization/edges.csv` | Gephi 边表 |
| `data/processed/visualization/neo4j/nodes.csv` | Neo4j 导入节点 |
| `data/processed/visualization/neo4j/relationships.csv` | Neo4j 导入关系 |
| `data/processed/visualization/neo4j/load_csv.cypher` | Neo4j 导入脚本 |
| `data/processed/visualization/neo4j/browser_queries.cypher` | Neo4j Browser 查询样例 |

## 五、运行环境

项目 Python 环境使用 conda 的 `knowgraph` 环境。执行 Python 命令前先确认解释器：

```powershell
conda run -n knowgraph python -c "import sys; print(sys.executable)"
```

当前已验证解释器路径示例：

```text
G:\anaconda3\envs\knowgraph\python.exe
```

安装依赖：

```powershell
conda run -n knowgraph python -m pip install -r knowledge_graph\requirements.txt
```

主要依赖：

| 依赖 | 用途 |
| --- | --- |
| `PyYAML` | 配置解析 |
| `beautifulsoup4` | HTML 清洗 |
| `pypdf` | PDF 文本抽取 |
| `joblib` | 模型持久化 |
| `sklearn-crfsuite` | CRF mention 识别 |
| `numpy`、`nltk`、`torch` | 关系抽取模型与文本处理 |

## 六、运行方式

统一入口：

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py --help
```

PowerShell 在中文输出下可能出现 GBK 编码问题，可临时指定 UTF-8：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py --help
```

### 6.1 结构化种子库

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py structured init-db
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py structured build
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py structured validate
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py structured export-csv
```

### 6.2 非结构化预处理

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py unstructured preprocess
```

### 6.3 Mention 识别

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py mentions prepare
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py mentions weak-label
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py mentions train
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py mentions predict
```

说明：`mentions weak-label` 支持传入 `--api-key`、`--base-url`、`--model-name` 调用外部弱监督接口；不传时按当前实现走默认配置。

### 6.4 实体链接与共指

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py linking link
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py linking mine-gaps
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py coreference resolve
```

### 6.5 关系候选与事实抽取

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py relations prepare
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py relations weak-label
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py facts run
```

如需训练关系模型，可继续执行：

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py relations train
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py relations predict
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py relations evaluate
```

### 6.6 图谱导出

```powershell
conda run -n knowgraph python knowledge_graph\scripts\turing_kg.py visualization export
```

导出后可直接打开：

```text
knowledge_graph\data\processed\visualization\graph.html
```

## 七、数据契约示例

### 7.1 实体记录

```csv
entity_id,canonical_name,label_en,label_zh,description_en,entity_type,confidence
Q7251,艾倫·圖靈,Alan Turing,艾倫·圖靈,English computer scientist (1912-1954),Person,0.95
```

### 7.2 弱标注 mention 记录

```json
{
  "sentence_id": "sent_000017",
  "text": "You can see a letter here from Alan Turing ...",
  "tokens": ["You", "can", "see", "a", "letter", "from", "Alan", "Turing"],
  "labels": ["O", "O", "O", "O", "O", "O", "B-PER", "I-PER"],
  "label_source": "llm_weak_supervision",
  "weak_label_confidence": 0.92
}
```

### 7.3 关系候选记录

```json
{
  "candidate_id": "relcand_000002",
  "sentence_id": "sent_000021",
  "predicate": "STUDIED_AT",
  "subject_text": "Turing",
  "subject_entity_id": "Q7251",
  "object_text": "Princeton University",
  "object_entity_id": "Q21578",
  "exact_claim_match": true
}
```

### 7.4 最终事实记录

```json
{
  "fact_id": "fact_000002",
  "subject_id": "Q7251",
  "predicate": "STUDIED_AT",
  "object_id": "Q21578",
  "confidence": 0.85,
  "extractor": "pattern+ds+llm_verify",
  "status": "FINAL"
}
```

## 八、数据替换指南

后续替换真实数据时，优先改配置，不改代码：

| 要替换的内容 | 修改位置 |
| --- | --- |
| 核心种子实体 | `configs/seed_entities.json` |
| 结构化抓取关系范围 | `configs/structured_fetch_config.json` 的 `relations` |
| HTML / PDF 原始语料 | `data/raw/` |
| 非结构化数据源清单 | `configs/unstructured_sources.yaml` |
| 关系触发词 | `configs/relation_patterns.yaml` |
| 本体实体/关系/事件定义 | `knowledge/ontology.json` |

替换后推荐按“结构化 -> 非结构化 -> mention -> linking -> coreference -> relations/facts -> visualization”的顺序重跑，避免下游产物混用旧数据。

## 九、当前进度

| 模块 | 状态 |
| --- | --- |
| 本体定义 | 已完成 V2 |
| 结构化种子库 | 已打通，当前单种子 Alan Turing |
| 非结构化预处理 | 已打通 HTML / PDF |
| CRF mention 识别 | 已打通弱标注、训练、预测链路 |
| 实体链接 | 已打通候选召回、打分、review/gap 报告 |
| 共指解析 | 已有规则入口，当前样例未产生额外 coref 链接 |
| 关系候选 | 已打通 resolved mention 到 pair candidates |
| 事实抽取 | 已打通规则 + DS + 离线校验 + 聚合 |
| 可视化导出 | 已打通 Graph JSON、HTML、Gephi、Neo4j CSV |

## 十、后续重点

1. 扩充种子实体：从单中心 Alan Turing 扩展到 Turing Machine、Turing Test、Bletchley Park、Turing Award 等多中心。
2. 增加人工金标：补齐 `test_gold.jsonl`、`linking_dev_gold.jsonl`、`relation_gold.jsonl`，让 CRF、linking、relation 都能稳定评估。
3. 提升 linking 召回：针对 Bletchley Park、Enigma、Manchester Mark 1 等高频 NIL/REVIEW mention 补充结构化实体和别名。
4. 强化事实校验：将当前离线校验升级为可配置 LLM verifier，并保留人工复核队列。
5. 固化演示视图：围绕 Alan Turing 两跳网络、事件链路和文本证据层准备课堂展示查询。
