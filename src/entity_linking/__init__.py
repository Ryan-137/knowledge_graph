"""候选召回、可解释排序、NIL 与文档级协同消歧。"""

from .config import LinkingConfig
from .evaluation import evaluate_linking_from_paths
from .gap_mining import mine_linking_gaps_from_path
from .pipeline import EntityLinker, link_mentions_from_paths

__all__ = [
    "EntityLinker",
    "LinkingConfig",
    "evaluate_linking_from_paths",
    "link_mentions_from_paths",
    "mine_linking_gaps_from_path",
]
