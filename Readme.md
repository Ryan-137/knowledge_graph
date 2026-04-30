# 图灵知识图谱课程项目

本项目以 Alan Turing 为核心对象，围绕人物、机构、地点、作品、机器、概念、奖项与事件等知识单元，构建一个可溯源、可复现、可导出的专题知识图谱。项目覆盖知识工程中的主要流程：数据采集与预处理、本体建模、命名实体识别、实体链接、共指解析、关系抽取、事件抽取、事实融合与图谱可视化。

项目不仅生成最终图谱，也保留了每个阶段的中间数据和处理报告，便于检查数据来源、抽取方法和结果质量。

## 一、总体技术路线

```text
本体设计
  -> 结构化种子库构建
  -> 非结构化语料预处理
  -> Mention 识别
  -> 实体链接与共指解析
  -> 关系候选生成与远程监督
  -> 关系模型训练与预测
  -> 事实抽取、校验与聚合
  -> 图谱融合与可视化导出
```

各阶段使用的方法如下：

| 阶段 | 使用方法 | 说明 |
| --- | --- | --- |
| 本体建模 | 领域本体设计 | 统一实体类型、关系类型、事件类型和约束 |
| 结构化知识获取 | Wikidata API + Wikipedia summary | 以 Alan Turing 的 QID 为种子获取实体、别名和声明 |
| 文本预处理 | HTML/PDF 抽取 + 文本清洗 + 分句 | 将不同来源转换为统一的句子级数据 |
| Mention 识别 | 弱监督标注 + CRF 序列标注 | 识别人名、机构、地点、作品、机器、概念和奖项 |
| 实体链接 | 候选召回 + 特征打分 + 上下文判断 | 将文本 mention 映射到结构化实体 ID |
| 共指解析 | 规则共指 + anchor 传播 | 补全代词、简称和上下文省略导致的实体指代 |
| 关系抽取 | 实体对候选 + 远程监督 + 关系分类模型 | 从句子中判断实体间可能存在的语义关系 |
| 事实抽取 | 规则触发词 + 结构化对齐 + 离线校验 | 聚合可入图的事实并记录置信度 |
| 事件抽取 | 触发词检测 + 论元绑定 | 提取文本中与人物协作、影响等相关的事件候选 |
| 图谱融合 | 多来源融合 + 去重 + 多格式导出 | 生成 JSON、CSV、HTML 和 Neo4j 导入文件 |

## 二、项目结构

```text
knowledge_graph/
├─ configs/                         # 配置文件
│  ├─ seed_entities.json             # 结构化抓取种子
│  ├─ structured_fetch_config.json   # Wikidata / Wikipedia 抓取配置
│  ├─ unstructured_sources.yaml      # HTML / PDF 数据源清单
│  ├─ relation_patterns.yaml         # 关系触发词规则
│  └─ relation_training_config.json  # 关系模型训练配置
├─ knowledge/
│  └─ ontology.json                  # 本体定义
├─ scripts/
│  └─ turing_kg.py                   # 统一命令行入口
├─ src/
│  ├─ structured_seed/               # 结构化种子库构建
│  ├─ unstructured_preprocess/       # 非结构化文本预处理
│  ├─ mention_crf/                   # Mention 识别
│  ├─ entity_linking/                # 实体链接
│  ├─ coreference/                   # 共指解析
│  ├─ relation_extraction/           # 关系抽取
│  ├─ event_extraction/              # 事件抽取
│  ├─ fact_extraction/               # 事实抽取
│  ├─ visualization_export/          # 图谱导出
│  └─ kg_core/                       # 公共工具与数据契约
├─ data/
│  ├─ raw/                           # 原始 HTML/PDF 数据
│  └─ processed/                     # 中间结果与最终产物
├─ results/                          # 课程展示截图
└─ requirements.txt                  # Python 依赖
```



## 三、数据来源与数据形式

### 3.1 课程材料

课程授课材料主要作为方法参考，用于指导本项目中的本体设计、实体识别、实体链接、关系抽取、知识融合和可视化等流程。这些材料不作为图谱事实数据直接入库。

### 3.2 结构化数据

结构化数据以 Alan Turing 的 Wikidata QID `Q7251` 为起点，获取其基础信息、别名、声明以及一跳/两跳关联实体。Wikipedia summary 用于补充实体描述。

| 数据内容 | 文件 | 数据形式 | 当前规模 |
| --- | --- | --- | ---: |
| 实体 | `data/processed/structured/csv/entities.csv` | CSV | 38 个实体 |
| 别名 | `data/processed/structured/csv/aliases.csv` | CSV | 125 条别名 |
| 结构化声明 | `data/processed/structured/csv/claims.csv` | CSV | 30 条事实声明 |
| 事件候选 | `data/processed/structured/csv/event_candidates.csv` | CSV | 24 条事件候选 |

结构化数据示例：

```csv
entity_id,canonical_name,label_en,label_zh,description_en,entity_type,confidence
Q7251,艾倫·圖靈,Alan Turing,艾倫·圖靈,English computer scientist (1912-1954),Person,0.95
```

### 3.3 非结构化数据

非结构化语料由 `configs/unstructured_sources.yaml` 统一登记，当前包含 11 个来源。采集时按可信度和用途划分为两级：一级来源作为主要事实证据，优先选择权威机构、大学、学术组织和原始论文材料；二级来源作为补充材料，用于补足叙述细节、增加文本覆盖面，并与一级来源进行交叉验证。

| 来源级别 | 采集原则 | 示例来源 | 原始形式 | 用途 |
| --- | --- | --- | --- | --- |
| 一级来源 | 优先选取权威机构、大学、学术组织、原始论文或高可信传记材料 | Bletchley Park、King's College Cambridge、Princeton University、University of Manchester、The Royal Society、Computing Machinery and Intelligence | HTML / PDF | 作为主要事实来源和高置信证据 |
| 二级来源 | 选择百科、展览介绍和综合性资料，补充叙述和背景信息 | Wikipedia、Britannica、Science and Industry Museum | HTML | 补充事实上下文，并辅助交叉验证 |

预处理后统一为以下数据形式：

| 数据内容 | 文件 | 数据形式 | 当前规模 |
| --- | --- | --- | ---: |
| 文档级数据 | `data/processed/unstructured/documents.jsonl` | JSONL | 11 篇文档 |
| 句子级数据 | `data/processed/unstructured/sentences.jsonl` | JSONL | 2383 条句子 |

统一句子数据保留 `doc_id`、`source_id`、`sentence_id`、`text`、时间表达和来源字段，后续所有抽取结果都可以回溯到对应的证据句。

## 四、核心模块与方法

本章按照代码中的实际流水线展开：本体约束先定义图谱可接受的实体、关系和事件；结构化模块从 Wikidata 获取种子事实；非结构化模块将 HTML / PDF 统一成句子级证据；随后依次完成 mention 识别、实体链接、共指解析、关系抽取、事实校验、事件抽取与最终图谱融合。统一命令入口位于 `scripts/turing_kg.py`，各子命令对应 `structured`、`unstructured`、`mentions`、`linking`、`coreference`、`relations`、`facts`、`events` 和 `visualization` 九个阶段。

### 4.1 本体设计与数据契约

本体文件位于 `knowledge/ontology.json`，是全流程的类型约束和融合依据。本体没有把 mention 当成最终图谱节点，而是将 mention 作为抽取过程中的中间证据；真正入图的对象分为实体节点、事件节点和字面值节点（具体实现中为时间节点）。

实体类以 `Entity` 为根类，当前定义了 `Person`、`Organization`、`Place`、`Work`、`Concept`、`Machine`、`Award` 和 `Event` 等类型。其中 `Event` 继续细分为 `BirthEvent`、`EducationEvent`、`EmploymentEvent`、`PublicationEvent`、`DeathEvent`、`ProposalEvent`、`DesignEvent`、`HonorEvent`、`CollaborationEvent`、`InfluenceEvent`。这种设计使人物经历、学术成果、机器设计和荣誉奖项可以分别以关系边或事件节点表达。

关系层共定义 20 类语义关系，其中核心关系包括：

| 关系 | 主体类型 | 客体类型 | 说明 |
| --- | --- | --- | --- |
| `BORN_IN` | `Person` | `Place` | 出生地 |
| `DIED_IN` | `Person` | `Place` | 死亡地 |
| `STUDIED_AT` | `Person` | `Organization` | 教育经历 |
| `WORKED_AT` | `Person` | `Organization` | 任职或工作机构 |
| `AUTHORED` | `Person` | `Work` | 作者与作品 |
| `PROPOSED` | `Person` | `Concept` | 提出理论或概念 |
| `DESIGNED` | `Person` | `Machine` | 设计机器或计算设备 |
| `AWARDED` | `Person` | `Award` | 获奖或荣誉 |
| `LOCATED_IN` | `Organization` | `Place` | 机构所在地 |

本体还定义了必填属性、实体 ID 唯一性、事件 ID 唯一性、置信度范围、关系 domain / range 检查、事件角色完整性检查和事件时间顺序检查。后续的关系候选生成、事实聚合、事件校验和可视化导出均会读取这套约束，因此本体不仅是展示层分类表，也是工程层的数据契约。

### 4.2 结构化种子库构建

结构化种子库由 `src/structured_seed/` 实现，配置入口为 `configs/seed_entities.json` 和 `configs/structured_fetch_config.json`。当前项目以 Alan Turing 的 Wikidata QID `Q7251` 作为唯一结构化种子，期望类型为 `Person`。

结构化抓取流程分为五步：

1. 读取种子 QID，并通过 Wikidata SPARQL 查询实体基础信息，包括英文标签、中文标签、描述、Wikidata URL 和 Wikipedia 标题。
2. 查询实体别名，按实体 ID 写入 `aliases` 表，为后续 mention 识别和实体链接提供词典资源。
3. 按配置中的 9 类关系抓取声明，包括 `BORN_IN`、`DIED_IN`、`STUDIED_AT`、`WORKED_AT`、`AUTHORED`、`PROPOSED`、`DESIGNED`、`AWARDED`、`LOCATED_IN`。
4. 将声明中的 object entity 回填到实体表；一跳实体落库后再次抓取 claims，从而把结构化邻域扩展到两跳。
5. 对具有事件语义的结构化声明生成事件候选，再导出 SQLite 与 CSV 结果，供实体链接、远程监督和图谱融合使用。

实体类型由 `src/structured_seed/entity_typing.py` 统一推断。推断顺序是：先利用 seed 中的人工期望类型，再根据 Wikidata instance / subclass 信息，最后结合标签、描述和关键词规则补足类型。这样可以避免只依赖远端类型字段造成的空值或过细分类问题。

结构化层在整个系统中承担三类作用：第一，提供初始事实骨架；第二，提供实体与别名词典；第三，为关系抽取的远程监督提供 `(subject_id, predicate, object_id)` 对齐依据。

### 4.3 多源非结构化语料预处理

非结构化语料由 `configs/unstructured_sources.yaml` 统一登记，当前包含 11 个来源，其中一级来源 8 个、二级来源 3 个。

预处理模块位于 `src/unstructured_preprocess/`，其目标是把 HTML 和 PDF 统一转换为可溯源的句子级样本。文档级处理由 `documents.py` 执行：HTML 通过 BeautifulSoup 解析正文，去除脚本、样式、导航、页脚、目录和引用等噪声块；PDF 通过 `pypdf` 读取文本层；随后统一压缩空白、推断语言，并写入 `documents.jsonl`。句子级处理由 `sentences.py` 执行：系统按文档块和标点边界切句，同时记录句子在原始 clean_text 中的字符偏移，并抽取基础时间表达。

句子切分并不是简单按句号分割。`sentence_splitter.py` 会处理标题块、目录块、换行续接、闭合引号、括号和过短噪声句，尽量保留学术网页和 PDF 转写文本中的有效语义单元。`time_mentions.py` 则识别英文日期、年月、年份区间、约略年份和年代表达，为事件抽取与事实限定词提供时间线索。

当前非结构化处理结果如下：

| 指标 | 数值 |
| --- | ---: |
| 文档数 | 11 |
| 句子数 | 2,383 |
| 预处理错误数 | 0 |
| 最大单文档句子数 | 586 |
| 最小单文档句子数 | 28 |

最终句子样本写入 `data/processed/unstructured/sentences.jsonl`。每条记录包含 `sentence_id`、`doc_id`、`source_id`、`sentence_index_in_doc`、`text`、`char_start`、`char_end`、`normalized_time`、`time_mentions` 等字段，后续 mention、关系、事件和事实输出都通过这些字段回溯到原始证据句。

### 4.4 Mention 识别

Mention 识别模块位于 `src/mention_crf/`。本项目没有使用端到端大模型直接抽取图谱，而是采用“弱监督标注 + CRF 序列标注 + 词典补召回”的方式，从句子中定位实体提及。

标注体系覆盖 7 类领域实体：`PER`、`ORG`、`LOC`、`WORK`、`CONCEPT`、`MACHINE`、`AWARD`。处理流程如下：

1. `data.py` 先将句子转换为固定 token 序列，并保留 token 到字符位置的映射，确保后续 BIO 标签能和原文 span 对齐。
2. `weak_label.py` 结合结构化实体别名词典和弱监督接口生成 BIO 标注，并进行自动校验。校验规则会拒绝 token 数量不一致、非法 BIO、实体 token 占比过高、实体 span 跨越异常标点等样本。
3. `dictionary.py` 从结构化实体表和别名表构造最大正向匹配词典，用于训练特征和预测阶段的补召回。
4. `features.py` 为每个 token 构造 CRF 特征，包括 token 原文、小写形式、词形、大小写、数字、连字符、前后缀、上下文窗口、词性、括号深度、引号位置、标题串长度、时间提示和词典命中特征。
5. `train.py` 使用 `sklearn-crfsuite` 训练 CRF，并输出整体指标、分类型指标、错误分析、BIO 合法性检查和置信度分桶指标。
6. `predict.py` 对全量句子预测 mention，并在模型输出后执行 BIO 合法化、span 解码、机器别名边界扩展、作品/概念片段合并、词典 fallback 和去重。

当前训练数据与开发集结果如下：

| 指标 | 数值 |
| --- | ---: |
| 弱监督样本 | 423 条 |
| 训练集 | 339 条 |
| 开发集 | 84 条 |
| 训练集实体 span | 497 个 |
| 开发集实体 span | 108 个 |
| 开发集 Precision | 0.824 |
| 开发集 Recall | 0.694 |
| 开发集 F1 | 0.754 |

在人工 gold 测试集上，模型识别 111 个实体 span，其中 87 个与 gold 完全匹配，Precision 为 0.784，Recall 为 0.654，F1 为 0.713。分类型看，人物、机构和地点相对稳定，`WORK`、`CONCEPT`、`MACHINE` 受术语边界和上下文歧义影响更明显。

全量预测后，`data/processed/mentions/mentions.jsonl` 中当前共有 2,468 条 mention，类型分布如下：

| Mention 类型 | 数量 |
| --- | ---: |
| `PERSON` | 1,105 |
| `ORGANIZATION` | 443 |
| `PLACE` | 302 |
| `MACHINE` | 264 |
| `CONCEPT` | 246 |
| `WORK` | 82 |
| `AWARD` | 26 |

### 4.5 实体链接与共指解析

实体链接模块位于 `src/entity_linking/`，目标是把 mention 映射到结构化实体 ID。当前候选空间主要来自 `entities.csv` 和 `aliases.csv`，并结合 claims、文档标题和句子上下文进行打分。

候选召回采用多路策略：

| 候选来源 | 说明 |
| --- | --- |
| `exact_alias` | 规范前完全别名命中 |
| `normalized_alias` | 大小写、空白和符号规范化后的别名命中 |
| `surname_alias` | 人名姓氏简称召回 |
| `abbreviation_alias` | 机构缩写召回 |
| `short_name_alias` | 机构短名召回 |
| `place_variant_alias` | 地名变体召回 |
| `tfidf_recall` | 基于实体标签、描述、别名的 TF-IDF 相似召回 |

候选打分由 `features.py` 中的线性特征组合完成，可表示为：

```text
Score(m, e) = 0.30 * f_alias
            + 0.20 * f_context
            + 0.20 * f_type
            + 0.10 * f_doc
            + 0.10 * f_mention
            + 0.10 * f_prior
```

其中 `m` 表示 mention，`e` 表示候选实体。系统按得分排序候选实体，并结合 mention 长度、是否直接别名命中、top1 与 top2 的分差和类型一致性，决定输出 `LINKED`、`REVIEW` 或 `NIL`。

为降低局部歧义，`disambiguation.py` 在文档级做轻量上下文支持：如果某个实体在同一文档内被高置信锚定，则同文档内相同或相近 mention 的候选会获得弱加分；如果 mention 与文档标题或已锚定实体上下文一致，也会提高最终链接置信度。

当前实体链接结果如下：

| 指标 | 数量 |
| --- | ---: |
| 输入 mention | 2,468 |
| `LINKED` | 1,276 |
| `REVIEW` | 220 |
| `NIL` | 972 |
| 平均候选数 | 4.9935 |

`NIL` 主要来自三类情况：证据不足 539 条、候选歧义 353 条、类型冲突 80 条。高频断链 mention 包括 `Science Museum Group`、`Entscheidungsproblem`、`Ferranti Mark 1` 等，这些记录已经写入 linking gap 报告，可作为后续补充实体库和别名表的优先清单。

共指解析模块位于 `src/coreference/`。它不重新训练模型，而是在实体链接输出上处理代词、简称和上下文省略。系统会在最大 3 句窗口内查找类型兼容的先行实体，并优先使用最近先行词、同实体锚点和明确人物上下文。

当前共指解析结果如下：

| 指标 | 数量 |
| --- | ---: |
| 处理 mention / skipped mention | 4,462 |
| `LINKED_BY_COREF` | 1,100 |
| `COREF_UNRESOLVED` | 894 |
| 原始 `SKIPPED_PRONOUN` | 1,824 |
| 原始 `SKIPPED_GENERIC` | 170 |

成功传播的先行实体以人物为主，共 1,012 条；其次是机器 32 条、地点 26 条、概念 22 条、机构 5 条、作品 3 条。共指层的输出 `resolved_mentions.jsonl` 是关系候选生成阶段的主要输入。

### 4.6 关系抽取

关系抽取模块位于 `src/relation_extraction/`，整体采用“实体对候选生成 + 远程监督弱标签 + PCNN-MIL Attention 关系模型”的路线。

候选生成由 `prepare.py` 完成。系统先读取 `resolved_mentions.jsonl`，在句内或局部上下文中构造实体对；再根据本体 domain / range、token 距离、低信息 mention、触发词和结构化 claims 进行过滤。对于文本中未直接链接但与结构化 claims 强相关的实体，系统还会构造 alias bridge，例如根据 claim 中的对象实体别名把 `King's College`、`Manchester University` 等表述桥接到结构化实体。

当前关系候选结果如下：

| 指标 | 数量 |
| --- | ---: |
| 句子数 | 2,383 |
| resolved mention 数 | 4,462 |
| alias bridge mention 数 | 184 |
| 关系候选总数 | 600 |
| bridge augmented 候选 | 326 |
| linked-linked 候选 | 172 |
| coref augmented 候选 | 102 |

候选关系覆盖 9 类目标关系，分布如下：

| 关系类型 | 候选数 |
| --- | ---: |
| `STUDIED_AT` | 176 |
| `WORKED_AT` | 156 |
| `DESIGNED` | 102 |
| `PROPOSED` | 46 |
| `DIED_IN` | 25 |
| `AWARDED` | 21 |
| `LOCATED_IN` | 12 |
| `BORN_IN` | 6 |
| `AUTHORED` | 5 |

远程监督由 `weak_label.py` 执行。系统把候选实体对与 Wikidata claims 对齐，形成 strict positive、hard negative、manual review 和 unknown 四类弱标签。当前 600 条候选中，`ds_strict` 263 条，`hard_negative` 248 条，`manual_review` 51 条，`unknown` 38 条。最终 predicate 分布中，`NA` 为 271 条，其余主要集中在 `STUDIED_AT` 109 条、`WORKED_AT` 88 条、`DESIGNED` 56 条。

关系模型由 `model.py` 实现，配置位于 `configs/relation_training_config.json`。模型结构为 PCNN + MIL Attention：

1. 句子编码阶段将 token embedding、subject 相对位置 embedding、object 相对位置 embedding 拼接。
2. 卷积层使用 kernel size 3 和 5，各 128 个通道，捕捉不同窗口大小的局部关系模式。
3. PCNN 按 subject / object 位置把句子切分为左、中、右三个 piece，并对每段做 max pooling。
4. 同一 `(doc_id, subject_id, object_id)` 的多条证据句组成一个 bag，最多保留 16 句。
5. 对每个关系类别分别学习 attention query，在 bag 内对证据句加权聚合。
6. 输出层做多标签分类，推理时根据本体约束和候选关系掩码过滤不合法关系。

模型训练使用 GloVe 100d 词向量，当前词表规模 984，预训练覆盖率 0.981。训练集 69 个 bag、开发集 14 个 bag、测试集 28 个 bag。最佳模型出现在第 4 轮；测试集 micro Precision 为 1.000，Recall 为 0.350，F1 为 0.519。该结果说明当前模型更偏向保守输出：误报较少，但受训练样本规模和正例分布限制，召回仍有明显提升空间。

关系预测结果写入 `data/processed/relations/extracted_claims.jsonl`，当前生成 41 条抽取声明，覆盖全部 9 类目标关系。

### 4.7 事实抽取、校验与聚合

事实抽取模块位于 `src/fact_extraction/`。它并不直接把关系模型输出全部入图，而是把每条关系声明转换为 fact candidate，再通过规则触发词、Wikidata 对齐、离线校验和聚合去重决定最终 facts。

候选生成由 `candidate_generator.py` 完成。对于来自 `extracted_claims.jsonl` 的关系声明，系统保留主体、客体、predicate、证据句、span、来源文档、抽取器名称和模型概率；同时调用 `pattern_rules.py` 对证据句进行触发词匹配。如果证据句中出现与 predicate 对应的触发表达，则添加 `pattern_match` 信号，当前命中分数为 0.35。

远程监督打分由 `distant_supervision.py` 完成。系统将 fact candidate 与结构化 claims 建立索引对齐，若 `(subject_id, predicate, object_id)` 与 Wikidata 声明一致，则添加 `wikidata_alignment` 信号。随后 `llm_verifier.py` 执行校验：在未配置在线 LLM 时使用 offline 模式，优先依据 Wikidata 对齐和规则触发词判断 `SUPPORTED` 或 `UNCERTAIN`。

聚合由 `aggregator.py` 完成。系统按 `(subject_id, predicate, object_id, qualifiers)` 合并重复事实，累计证据信号，根据信号分数和证据数量计算最终 confidence，并调用 `conflict_detector.py` 检查同主体同谓词下是否存在互斥对象。

当前事实抽取结果如下：

| 阶段 | 数量 |
| --- | ---: |
| 输入关系声明 | 41 |
| fact candidates | 41 |
| pattern 命中 | 21 |
| Wikidata 对齐 | 36 |
| offline `SUPPORTED` | 21 |
| offline `UNCERTAIN` | 20 |
| usable candidates | 35 |
| final facts | 28 |
| conflict facts | 0 |

最终事实按关系类型分布如下：

| 关系类型 | 数量 |
| --- | ---: |
| `STUDIED_AT` | 8 |
| `WORKED_AT` | 7 |
| `AWARDED` | 3 |
| `AUTHORED` | 2 |
| `DESIGNED` | 2 |
| `LOCATED_IN` | 2 |
| `PROPOSED` | 2 |
| `BORN_IN` | 1 |
| `DIED_IN` | 1 |

这一步是文本 evidence 入图前的质量闸门：关系模型负责召回候选，事实层负责把候选压缩为可审计、可溯源、可融合的图谱边。

### 4.8 事件抽取

事件抽取模块位于 `src/event_extraction/`。当前文本事件抽取聚焦 `CollaborationEvent` 和 `InfluenceEvent` 两类事件，用于补充普通二元关系难以表达的“合作”和“影响”语义。

触发词规则由 `trigger_detector.py` 定义，当前包括：

| 事件类型 | 触发模式 | 角色映射 |
| --- | --- | --- |
| `CollaborationEvent` | `worked with` | 左侧人物为 `person_a`，右侧人物为 `person_b` |
| `CollaborationEvent` | `collaborated with` | 左侧人物为 `person_a`，右侧人物为 `person_b` |
| `InfluenceEvent` | `was influenced by` | 左侧人物为 `target_person`，右侧人物为 `source_person` |
| `InfluenceEvent` | `influenced` | 左侧人物为 `source_person`，右侧人物为 `target_person` |

系统先在句子文本中匹配触发词，再从同句 resolved mention 中筛选人物 mention，并选择触发词左右最近的论元。事件 ID 由句子 ID、事件类型、触发词和角色实体组合哈希生成，保证重复运行时 ID 稳定。`candidate_generator.py` 还会把事件与同句关系候选做交叉引用，并附加句内时间表达。

事件校验由 `verifier.py` 完成。校验依据是本体中的 required roles 和角色类型约束：如果事件缺少必要角色、角色没有实体 ID、角色类型不匹配，或触发词存在需人工复核的上下文，事件状态会被置为 `REVIEW`，不会直接转为事实。`event_to_fact.py` 只会把 `VERIFIED` 事件投影为 `WORKED_WITH` 或 `INFLUENCED` fact candidate。

当前文本事件抽取结果如下：

| 指标 | 数量 |
| --- | ---: |
| 处理句子 | 2,383 |
| 事件候选 | 4 |
| 事件论元 | 8 |
| `CollaborationEvent` | 1 |
| `InfluenceEvent` | 3 |
| `REVIEW` 事件 | 4 |
| 投影为事实的事件 | 0 |

4 条事件均因关键论元未成功链接进入 `REVIEW`：1 条缺少 `person_a`，3 条缺少 `target_person`。这说明当前事件模块采用保守入图策略，宁可保留可审查候选，也不把未完整链接的事件强行写入最终 facts。

需要区分的是：文本事件模块当前没有产生 final facts，但结构化 Wikidata claims 已经派生出 24 条事件候选，并在可视化融合阶段作为 event 层进入图谱。

### 4.9 图谱融合、导出与可视化

图谱融合与导出模块位于 `src/visualization_export/`。它将三类来源统一成图谱视图：

| 来源层 | 输入 | 入图方式 |
| --- | --- | --- |
| `facts` | `data/processed/structured/csv/claims.csv` | Wikidata 结构化声明边 |
| `event` | `data/processed/structured/csv/event_candidates.csv` | 事件节点 + `EXT_EVENT_ARG` 论元边 |
| `evidence` | `data/processed/facts/facts_final.jsonl` | 文本抽取并校验后的证据边 |

节点构建由 `exporter.py` 完成。结构化实体直接转为实体节点；文本事实中没有实体 ID 的 object 会转为 literal 节点；事件候选会转为事件节点，并按角色连接人物、机构、地点、作品、机器、奖项或时间字面值。边 ID 由 source、target、关系标签、来源层和证据内容稳定哈希生成，保证重复导出时可去重。

融合逻辑由 `fusion.py` 负责。系统会先规范化边身份，再按来源层设置不同的语义去重规则：`facts` 层按结构化三元组去重，`event` 层按事件与角色去重，`evidence` 层保留证据哈希，允许同一实体对在不同证据句下保留多条可审计证据边。当前没有额外人工 correction，因此没有新增边、替换链接或拒绝边。

当前图谱规模如下：

| 指标 | 数量 |
| --- | ---: |
| 节点总数 | 73 |
| 边总数 | 121 |
| `facts` 层边 | 30 |
| `event` 层边 | 63 |
| `evidence` 层边 | 28 |

节点类型分布如下：

| 节点类型 | 数量 |
| --- | ---: |
| `Literal` | 14 |
| `Place` | 9 |
| `Organization` | 8 |
| `Concept` | 5 |
| `Award` | 4 |
| `Machine` | 4 |
| `Person` | 3 |
| `Work` | 2 |
| `HonorEvent` | 4 |
| `EducationEvent` | 4 |
| `EmploymentEvent` | 4 |
| `ProposalEvent` | 4 |
| `DesignEvent` | 3 |
| `PublicationEvent` | 2 |
| `BirthEvent` | 2 |
| `DeathEvent` | 1 |

边类型分布如下：

| 边类型 | 数量 |
| --- | ---: |
| `EXT_EVENT_ARG` | 63 |
| `STUDIED_AT` | 12 |
| `WORKED_AT` | 11 |
| `LOCATED_IN` | 8 |
| `PROPOSED` | 7 |
| `AWARDED` | 7 |
| `AUTHORED` | 4 |
| `DESIGNED` | 4 |
| `BORN_IN` | 3 |
| `DIED_IN` | 2 |

最终导出文件包括：

| 文件 | 用途 |
| --- | --- |
| `data/processed/visualization/graph.json` | 程序可读取的完整图谱 JSON |
| `data/processed/visualization/graph.html` | 离线交互式图谱页面 |
| `data/processed/visualization/nodes.csv` | Gephi 节点表 |
| `data/processed/visualization/edges.csv` | Gephi 边表 |
| `data/processed/visualization/neo4j/nodes.csv` | Neo4j 节点导入表 |
| `data/processed/visualization/neo4j/relationships.csv` | Neo4j 关系导入表 |
| `data/processed/visualization/neo4j/load_csv.cypher` | Neo4j 导入脚本 |
| `data/processed/visualization/neo4j/browser_queries.cypher` | Neo4j Browser 查询示例 |


## 五、运行方式
### 5.1 环境要求与安装

Python 版本：3.10+

本项目推荐使用 conda 环境运行。

```powershell
conda create -n turing-kg python=3.10 -y
conda activate turing-kg
pip install -r requirements.txt
```


关键依赖：

| 依赖 | 用途 |
| --- | --- |
| `PyYAML` | 解析数据源和训练配置 |
| `beautifulsoup4` | HTML 正文解析与清洗 |
| `pypdf` | PDF 文本抽取 |
| `sklearn-crfsuite` | CRF Mention 识别 |
| `joblib` | 模型保存与加载 |
| `numpy`、`nltk` | 文本特征和数据处理 |
| `torch` | 关系抽取模型训练与预测 |
| `Neo4j` | 可选图数据库导入与可视化工具，不是 Python 必装依赖 |



### 5.2 运行命令
查看统一入口：

```powershell
python knowledge_graph\scripts\turing_kg.py --help
```

推荐运行顺序：

```powershell
python knowledge_graph\scripts\turing_kg.py structured init-db
python knowledge_graph\scripts\turing_kg.py structured build
python knowledge_graph\scripts\turing_kg.py structured validate
python knowledge_graph\scripts\turing_kg.py structured export-csv

python knowledge_graph\scripts\turing_kg.py unstructured preprocess

python knowledge_graph\scripts\turing_kg.py mentions prepare
python knowledge_graph\scripts\turing_kg.py mentions weak-label
python knowledge_graph\scripts\turing_kg.py mentions train
python knowledge_graph\scripts\turing_kg.py mentions predict

python knowledge_graph\scripts\turing_kg.py linking link
python knowledge_graph\scripts\turing_kg.py linking mine-gaps
python knowledge_graph\scripts\turing_kg.py coreference resolve

python knowledge_graph\scripts\turing_kg.py relations prepare
python knowledge_graph\scripts\turing_kg.py relations weak-label
python knowledge_graph\scripts\turing_kg.py relations train
python knowledge_graph\scripts\turing_kg.py relations predict
python knowledge_graph\scripts\turing_kg.py relations evaluate

python knowledge_graph\scripts\turing_kg.py events extract
python knowledge_graph\scripts\turing_kg.py events to-facts
python knowledge_graph\scripts\turing_kg.py facts run
python knowledge_graph\scripts\turing_kg.py visualization export
```

可视化结果：

```text
knowledge_graph\data\processed\visualization\graph.html
```

除离线 HTML 页面外，也可以使用 Neo4j 进行图谱可视化。图谱导出后，`data/processed/visualization/neo4j/` 中会生成 Neo4j 导入所需的节点表、关系表和 Cypher 脚本：

| 文件 | 用途 |
| --- | --- |
| `data/processed/visualization/neo4j/nodes.csv` | Neo4j 节点导入文件 |
| `data/processed/visualization/neo4j/relationships.csv` | Neo4j 关系导入文件 |
| `data/processed/visualization/neo4j/load_csv.cypher` | Neo4j 导入脚本 |
| `data/processed/visualization/neo4j/browser_queries.cypher` | Neo4j Browser 查询示例 |

导入 Neo4j 后，可以在 Neo4j Browser 中通过 Cypher 查询查看 Alan Turing 的邻居节点、关系路径和局部子图。

## 六、当前成果与后续工作

当前已经完成从数据源登记、文本预处理、实体识别、实体链接、关系抽取、事件抽取到最终图谱导出的完整流程。最终图谱包含 73 个节点、121 条边，其中 facts 层 30 条、event 层 63 条、evidence 层 28 条。

需要说明的是，本项目当前定位为课程知识工程方法的完整实践，重点是把课堂涉及的内容串联起来，并保留中间层数据以便检查和复现。因此，项目没有优先选择当前效果最好的大模型端到端抽取方案，而是采用 CRF、规则、远程监督和可解释模型等更贴合课程方法的实现方式。

由于当前 Mention 识别采用 CRF 模型，整体效果仍有提升空间；而知识图谱构建流程具有明显的链式依赖，前一环节的识别误差会继续影响实体链接、关系候选生成、事实抽取和事件抽取。为了保证最终入图结果的可信度，后续模块采取了较保守的过滤、校验和审查策略，所以最终进入图谱的高置信数据规模相对较小。这也是当前图谱节点和边数量不大的主要原因。同样也是最后非结构化数据事件抽取失败的主要原因。

后续可以从以下方向继续完善：

1. 扩充种子实体，从单中心 Alan Turing 扩展到 Turing Machine、Turing Test、Bletchley Park、Turing Award 等多中心图谱。
2. 增加人工标注数据，提升 mention、linking、relation 和 event 模块的可评估性。
3. 补充高频 NIL/REVIEW mention 的别名和实体，如 Ferranti Mark 1、Manchester Mark 1、Bletchley Park、Enigma 等。
4. 扩大远程监督训练语料，提高关系模型召回率。
5. 围绕 Alan Turing 的人物关系、教育经历、工作经历、理论贡献和机器设计整理课堂展示视图。
