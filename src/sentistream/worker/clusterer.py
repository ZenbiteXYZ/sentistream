import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from river import cluster

from sentistream.shared.config import settings


logger = logging.getLogger(settings.app.name)


class StreamClusterer:
    __slots__ = (
        "_dbstream_params",
        "_feature_keys",
        "_last_replay_count",
        "_max_recent_points",
        "_recent_points",
        "model",
        "records_processed",
    )

    def __init__(
        self,
        clustering_threshold: float = 1.06,
        fading_factor: float = 0.05,
        cleanup_interval: float = 2.0,
        intersection_factor: float = 0.3,
        minimum_weight: float = 1.0,
        n_dimensions: int = 5,
        max_recent_points: int = 2000,
    ):
        """Initializes the DBStream algorithm for continuous, online clustering.
        All parameters can be overridden via environment variables:
        SENTISTREAM_DBSTREAM_CLUSTERING_THRESHOLD, SENTISTREAM_DBSTREAM_FADING_FACTOR,
        SENTISTREAM_DBSTREAM_CLEANUP_INTERVAL, SENTISTREAM_DBSTREAM_INTERSECTION_FACTOR,
        SENTISTREAM_DBSTREAM_MINIMUM_WEIGHT, SENTISTREAM_CLUSTERER_N_DIMENSIONS, SENTISTREAM_CLUSTERER_MAX_RECENT_POINTS
        """
        import os

        def get_env_float(var, default):
            val = os.environ.get(var)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    return default
            return default

        def get_env_int(var, default):
            val = os.environ.get(var)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    return default
            return default

        clustering_threshold = get_env_float(
            "SENTISTREAM_DBSTREAM_CLUSTERING_THRESHOLD", clustering_threshold
        )
        fading_factor = get_env_float(
            "SENTISTREAM_DBSTREAM_FADING_FACTOR", fading_factor
        )
        cleanup_interval = get_env_float(
            "SENTISTREAM_DBSTREAM_CLEANUP_INTERVAL", cleanup_interval
        )
        intersection_factor = get_env_float(
            "SENTISTREAM_DBSTREAM_INTERSECTION_FACTOR", intersection_factor
        )
        minimum_weight = get_env_float(
            "SENTISTREAM_DBSTREAM_MINIMUM_WEIGHT", minimum_weight
        )
        n_dimensions = get_env_int("SENTISTREAM_CLUSTERER_N_DIMENSIONS", n_dimensions)
        max_recent_points = get_env_int(
            "SENTISTREAM_CLUSTERER_MAX_RECENT_POINTS", max_recent_points
        )

        self._dbstream_params = {
            "clustering_threshold": clustering_threshold,
            "fading_factor": fading_factor,
            "cleanup_interval": cleanup_interval,
            "intersection_factor": intersection_factor,
            "minimum_weight": minimum_weight,
            "n_dimensions": n_dimensions,
            "max_recent_points": max_recent_points,
        }
        self.model = cluster.DBSTREAM(
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight,
        )
        self.records_processed = 0
        self._max_recent_points = max_recent_points
        self._recent_points = deque(maxlen=max_recent_points)
        self._last_replay_count = 0

        # Pre-compute feature keys to avoid string formatting overhead in the critical loop
        self._feature_keys = tuple(f"dim_{i}" for i in range(n_dimensions))

        logger.info(
            "StreamClusterer initialized with params: %s", self._dbstream_params
        )

    # Removed duplicate pickle-based save_state and load_state methods

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
        self._recent_points.append(reduced_coords)

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

    def get_recent_points_count(self) -> int:
        return len(self._recent_points)

    def get_last_replay_count(self) -> int:
        return self._last_replay_count

    def save_state(self, path: str) -> None:
        """Persists a JSON snapshot of recent points for a warm restart."""
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records_processed": self.records_processed,
            "params": self._dbstream_params,
            "recent_points": list(self._recent_points),
        }
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_state(self, path: str) -> bool:
        """Loads JSON snapshot and replays points to restore state."""
        state_path = Path(path)
        if not state_path.exists():
            return False

        with state_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        params = payload.get("params", {})
        self.__init__(
            clustering_threshold=params.get("clustering_threshold", 1.5),
            fading_factor=params.get("fading_factor", 0.05),
            cleanup_interval=params.get("cleanup_interval", 2.0),
            intersection_factor=params.get("intersection_factor", 0.3),
            minimum_weight=params.get("minimum_weight", 1.0),
            n_dimensions=params.get("n_dimensions", 5),
            max_recent_points=params.get("max_recent_points", 2000),
        )

        points = payload.get("recent_points", [])
        for point in points:
            self.get_cluster(point)

        self._last_replay_count = len(points)
        self.records_processed = payload.get("records_processed", 0)
        return True
