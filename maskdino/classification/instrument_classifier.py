# ------------------------------------------------------------------------
# Site-specific instrument sub-type classification via k-NN cosine-similarity
# matching against a calibration database of decoder-query embeddings.
# Purely a runtime add-on: does not affect detection/segmentation.
# ------------------------------------------------------------------------
from dataclasses import dataclass

import numpy as np

UNKNOWN_LABEL = "unknown_instrument"


@dataclass
class ClassificationResult:
    label: str
    similarity: float


class InstrumentClassifier:
    """Loads a calibration database built by tools/build_instrument_database.py
    and classifies single embedding vectors against it via k-NN majority vote
    on cosine similarity, with a reject threshold for unfamiliar instruments.
    """

    def __init__(self, db_path, cfg_path=None, k: int = 5, similarity_threshold: float = 0.5):
        data = np.load(db_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)  # (N, hidden_dim), L2-normalized
        self.labels = data["labels"]  # (N,) str
        self.db_cfg_path = str(data["cfg_path"]) if "cfg_path" in data else None
        self.k = k
        self.similarity_threshold = similarity_threshold

        if cfg_path is not None and self.db_cfg_path is not None and str(cfg_path) != self.db_cfg_path:
            raise ValueError(
                f"Instrument database at {db_path} was built with config "
                f"'{self.db_cfg_path}', but the running model uses '{cfg_path}'. "
                "Embeddings from different model weights are not comparable — "
                "rebuild the database against the currently deployed model."
            )
        if len(self.embeddings) == 0:
            raise ValueError(f"Instrument database at {db_path} is empty.")

    def classify(self, embedding: np.ndarray) -> ClassificationResult:
        """embedding: (hidden_dim,) vector, expected L2-normalized (as produced by
        EmbeddingPredictor)."""
        similarities = self.embeddings @ embedding  # cosine similarity, both sides unit-norm
        k = min(self.k, len(similarities))
        top_idx = np.argpartition(-similarities, k - 1)[:k]
        top_idx = top_idx[np.argsort(-similarities[top_idx])]

        top_labels = self.labels[top_idx]
        top_sims = similarities[top_idx]

        # majority vote among the k nearest calibration examples; on ties, prefer
        # whichever candidate has the single closest match
        uniq, counts = np.unique(top_labels, return_counts=True)
        winners = uniq[counts == counts.max()]
        if len(winners) > 1:
            best_label = top_labels[0]
        else:
            best_label = winners[0]
        best_sim = float(top_sims[top_labels == best_label].mean())

        if best_sim < self.similarity_threshold:
            return ClassificationResult(label=UNKNOWN_LABEL, similarity=best_sim)
        return ClassificationResult(label=str(best_label), similarity=best_sim)

    def classify_batch(self, embeddings: np.ndarray):
        return [self.classify(e) for e in embeddings]
