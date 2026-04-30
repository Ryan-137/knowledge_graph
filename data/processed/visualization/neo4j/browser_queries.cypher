// Neo4j Browser 可视化查询：每次复制一个查询到 Browser 执行，切换到 Graph 视图。

// 1. 全量图：显示所有节点和所有关系
MATCH (n:KgNode)
OPTIONAL MATCH p = (n)-[r]->(m:KgNode)
RETURN n, p;

// 2. Alan Turing 两跳核心网络
MATCH p = (turing:KgNode {id: 'Q7251'})-[*1..2]-(neighbor:KgNode)
RETURN p
LIMIT 180;

// 3. facts 层：Wikidata 结构化事实骨架
MATCH p = (source:KgNode)-[r]->(target:KgNode)
WHERE r.source_layer = 'facts'
RETURN p
LIMIT 120;

// 4. evidence 层：文本抽取证据关系
MATCH p = (source:KgNode)-[r]->(target:KgNode)
WHERE r.source_layer = 'evidence'
RETURN p
LIMIT 120;

// 5. wikipedia_direct 层：从 Wikipedia infobox 直连抽取的结构化事实
MATCH p = (source:KgNode)-[r]->(target:KgNode)
WHERE r.source_layer = 'wikipedia_direct'
RETURN p
LIMIT 80;

// 6. 事件节点网络：事件作为独立节点连接人物、作品、地点和时间
MATCH p = (event:Event)-[r]->(target:KgNode)
RETURN p
LIMIT 160;

// 7. 高连接节点总览
MATCH (n:KgNode)
WITH n, size((n)--()) AS degree
ORDER BY degree DESC
LIMIT 30
MATCH p = (n)--(m)
RETURN p
LIMIT 220;
