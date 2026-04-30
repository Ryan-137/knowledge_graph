// 将 nodes.csv 和 relationships.csv 放到 Neo4j import 目录后，在 Neo4j Browser 或 cypher-shell 中执行本脚本。
CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KgNode) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:KgNode {id: row.`:ID`})
SET n.name = row.name,
    n.neo4j_labels = row.`:LABEL`,
    n.entity_type = row.entity_type,
    n.description = row.description,
    n.source_layer = row.source_layer,
    n.source_name = row.source_name,
    n.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Award"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Award`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "BirthEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`BirthEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Concept"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Concept`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "DeathEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`DeathEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "DesignEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`DesignEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "EducationEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`EducationEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "EmploymentEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`EmploymentEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Entity"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Entity`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Event"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Event`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "HonorEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`HonorEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Literal"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Literal`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Machine"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Machine`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Organization"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Organization`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Person"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Person`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Place"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Place`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "ProposalEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`ProposalEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "PublicationEvent"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`PublicationEvent`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Time"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Time`;

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.`:LABEL` CONTAINS "Work"
MATCH (n:KgNode {id: row.`:ID`})
SET n:`Work`;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "AUTHORED"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`AUTHORED` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "AWARDED"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`AWARDED` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "BORN_IN"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`BORN_IN` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "DESIGNED"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`DESIGNED` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "DIED_IN"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`DIED_IN` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "EXT_EVENT_ARG"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`EXT_EVENT_ARG` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "LOCATED_IN"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`LOCATED_IN` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "PROPOSED"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`PROPOSED` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "STUDIED_AT"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`STUDIED_AT` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row WHERE row.`:TYPE` = "WORKED_AT"
MATCH (source:KgNode {id: row.`:START_ID`})
MATCH (target:KgNode {id: row.`:END_ID`})
MERGE (source)-[r:`WORKED_AT` {edge_id: row.edge_id}]->(target)
SET r.source_layer = row.source_layer,
    r.source_name = row.source_name,
    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,
    r.evidence_text = row.evidence_text,
    r.semantic_key = row.semantic_key,
    r.fusion_status = row.fusion_status,
    r.fusion_reason = row.fusion_reason,
    r.role = row.role,
    r.display_label = row.display_label;

MATCH (n:KgNode) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC;
MATCH ()-[r]->() RETURN type(r) AS relation, r.source_layer AS layer, count(*) AS count ORDER BY count DESC;
