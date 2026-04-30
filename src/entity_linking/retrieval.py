from __future__ import annotations

from dataclasses import dataclass

from .models import EntityProfile


@dataclass
class RetrievalHit:
    entity_id: str
    score: float


class TfidfEntityRetriever:
    def __init__(self, entity_texts: dict[str, str]) -> None:
        self.entity_ids = list(entity_texts)
        self.entity_texts = [entity_texts[entity_id] for entity_id in self.entity_ids]
        self.vectorizer = None
        self.entity_matrix = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.entity_matrix = self.vectorizer.fit_transform(self.entity_texts)
        except Exception:
            self.vectorizer = None
            self.entity_matrix = None

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0 or not query.strip():
            return []
        if self.vectorizer is None or self.entity_matrix is None:
            lowered_query = query.casefold()
            hits: list[RetrievalHit] = []
            for entity_id, text in zip(self.entity_ids, self.entity_texts, strict=True):
                if lowered_query in text.casefold():
                    hits.append(RetrievalHit(entity_id=entity_id, score=0.5))
            return hits[:top_k]

        query_vector = self.vectorizer.transform([query])
        similarity = (self.entity_matrix @ query_vector.T).toarray().ravel()
        ranked = sorted(
            (
                RetrievalHit(entity_id=entity_id, score=float(score))
                for entity_id, score in zip(self.entity_ids, similarity, strict=True)
                if score > 0.0
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k]


def build_entity_retrieval_texts(entity_profiles: dict[str, EntityProfile], retrieval_text_by_entity: dict[str, str]) -> dict[str, str]:
    return {
        entity_id: retrieval_text_by_entity.get(entity_id) or " ".join(
            [profile.canonical_name, profile.description, *profile.aliases]
        )
        for entity_id, profile in entity_profiles.items()
    }
