import logging
from typing import Any

from river import cluster

from sentistream.shared.config import settings


logger = logging.getLogger(settings.app.name)


class StreamClusterer:
    __slots__ = ("_feature_keys", "model", "records_processed")

    def __init__(
        self,
        clustering_threshold: float = 1.5,
        fading_factor: float = 0.05,
        cleanup_interval: float = 2.0,
        intersection_factor: float = 0.3,
        minimum_weight: float = 1.0,
        n_dimensions: int = 5,
    ):
        """Initializes the DBStream algorithm for continuous, online clustering."""
        self.model = cluster.DBSTREAM(
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight,
        )
        self.records_processed = 0

        # Pre-compute feature keys to avoid string formatting overhead in the critical loop
        self._feature_keys = tuple(f"dim_{i}" for i in range(n_dimensions))

    def get_cluster(self, reduced_coords: list[float]) -> int:
        """
        Ingests a new reduced embedding, updates the DBStream graph,
        and returns the assigned cluster ID.
        """
        # dict(zip()) with pre-computed tuple is significantly faster
        # than repeated dict-comprehensions with f-string evaluations.
        x = dict(zip(self._feature_keys, reduced_coords, strict=False))

        # Update the DBSTREAM topological graph with the new point
        self.model.learn_one(x)
        self.records_processed += 1

        # Retrieve the assigned cluster ID
        cluster_id = self.model.predict_one(x)
        return cluster_id if cluster_id is not None else -1

    def get_active_clusters_info(self) -> list[dict[str, Any]]:
        """
        Utility to extract current micro-cluster centers and weights.
        Useful for generating visualizations directly from the River engine.
        """
        if not hasattr(self.model, "micro_clusters"):
            return []

        return [
            {
                "cluster_id": cluster_id,
                "weight": mc.weight,
                "center": mc.center,
            }
            for cluster_id, mc in self.model.micro_clusters.items()
        ]
