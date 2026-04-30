from __future__ import annotations


def build_entity_base_query(qids: list[str], limit: int, offset: int) -> str:
    entity_values = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
    SELECT ?entity ?entityLabelEn ?entityLabelZh ?descriptionEn ?descriptionZh ?instanceOf ?instanceOfLabel ?birthDate ?deathDate ?wikipediaTitleEn WHERE {{
      VALUES ?entity {{ {entity_values} }}
      OPTIONAL {{
        ?entity rdfs:label ?entityLabelEn .
        FILTER(LANG(?entityLabelEn) = "en")
      }}
      OPTIONAL {{
        ?entity rdfs:label ?entityLabelZh .
        FILTER(LANG(?entityLabelZh) = "zh")
      }}
      OPTIONAL {{
        ?entity schema:description ?descriptionEn .
        FILTER(LANG(?descriptionEn) = "en")
      }}
      OPTIONAL {{
        ?entity schema:description ?descriptionZh .
        FILTER(LANG(?descriptionZh) = "zh")
      }}
      OPTIONAL {{
        ?entity wdt:P31 ?instanceOf .
        OPTIONAL {{
          ?instanceOf rdfs:label ?instanceOfLabel .
          FILTER(LANG(?instanceOfLabel) = "en")
        }}
      }}
      OPTIONAL {{ ?entity wdt:P569 ?birthDate . }}
      OPTIONAL {{ ?entity wdt:P570 ?deathDate . }}
      OPTIONAL {{
        ?article schema:about ?entity ;
                 schema:isPartOf <https://en.wikipedia.org/> ;
                 schema:name ?wikipediaTitleEn .
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_entity_alias_query(qids: list[str], limit: int, offset: int) -> str:
    entity_values = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
    SELECT ?entity ?alias ?aliasLang WHERE {{
      VALUES ?entity {{ {entity_values} }}
      ?entity skos:altLabel ?alias .
      BIND(LANG(?alias) AS ?aliasLang)
      FILTER(?aliasLang IN ("en", "zh"))
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_born_in_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel WHERE {{
      VALUES ?subject {{ {values} }}
      ?subject p:P19 ?statement .
      ?statement ps:P19 ?object .
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_died_in_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel WHERE {{
      VALUES ?subject {{ {values} }}
      ?subject p:P20 ?statement .
      ?statement ps:P20 ?object .
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_studied_at_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel ?startTime ?endTime WHERE {{
      VALUES ?subject {{ {values} }}
      ?subject p:P69 ?statement .
      ?statement ps:P69 ?object .
      OPTIONAL {{ ?statement pq:P580 ?startTime . }}
      OPTIONAL {{ ?statement pq:P582 ?endTime . }}
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_worked_at_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel ?startTime ?endTime WHERE {{
      VALUES ?subject {{ {values} }}
      ?subject p:P108 ?statement .
      ?statement ps:P108 ?object .
      OPTIONAL {{ ?statement pq:P580 ?startTime . }}
      OPTIONAL {{ ?statement pq:P582 ?endTime . }}
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_authored_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel ?publicationDate WHERE {{
      VALUES ?subject {{ {values} }}
      ?object p:P50 ?statement .
      ?statement ps:P50 ?subject .
      OPTIONAL {{ ?object wdt:P577 ?publicationDate . }}
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_proposed_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel WHERE {{
      VALUES ?subject {{ {values} }}
      ?object p:P61 ?statement .
      ?statement ps:P61 ?subject .
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_designed_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel ?startTime ?endTime WHERE {{
      VALUES ?subject {{ {values} }}
      ?object p:P287 ?statement .
      ?statement ps:P287 ?subject .
      OPTIONAL {{ ?statement pq:P580 ?startTime . }}
      OPTIONAL {{ ?statement pq:P582 ?endTime . }}
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_awarded_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel ?pointInTime WHERE {{
      VALUES ?subject {{ {values} }}
      ?subject p:P166 ?statement .
      ?statement ps:P166 ?object .
      OPTIONAL {{ ?statement pq:P585 ?pointInTime . }}
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """


def build_located_in_query(subject_ids: list[str], limit: int, offset: int) -> str:
    values = " ".join(f"wd:{qid}" for qid in subject_ids)
    return f"""
    SELECT ?subject ?statement ?object ?objectLabel ?pointInTime WHERE {{
      VALUES ?subject {{ {values} }}
      ?subject p:P159 ?statement .
      ?statement ps:P159 ?object .
      OPTIONAL {{ ?statement pq:P585 ?pointInTime . }}
      OPTIONAL {{
        ?object rdfs:label ?objectLabel .
        FILTER(LANG(?objectLabel) = "en")
      }}
    }}
    LIMIT {int(limit)}
    OFFSET {int(offset)}
    """
