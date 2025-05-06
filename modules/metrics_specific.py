# forest_app/modules/metrics_specific.py

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetricsSpecificEngine:
    """
    Manages secondary metrics (currently: momentum gauge) and
    emits signals when key thresholds are crossed.
    """

    def __init__(self, alpha: float = 0.3, thresholds: Optional[Dict[str, float]] = None):
        # EWMA smoothing factor for momentum updates
        self.alpha = alpha
        # Current overall momentum (0–1 scale)
        self.momentum_overall: float = 0.5
        # Last inputs from calculate_metric_updates
        self._last_inputs: Dict[str, Any] = {}
        # Thresholds for signaling
        self.thresholds = thresholds or {
            "low_capacity": 0.3,
            "high_shadow": 0.7,
            "low_momentum": 0.3
        }

    def update_from_dict(self, data: Dict[str, Any]):
        """
        Rehydrates engine state from snapshot.component_state['metrics_engine'].
        """
        self.momentum_overall = data.get("momentum_overall", self.momentum_overall)
        self.alpha = data.get("alpha", self.alpha)
        # Optionally allow thresholds to be reconfigured
        if "thresholds" in data and isinstance(data["thresholds"], dict):
            self.thresholds.update(data["thresholds"])
        logger.debug("MetricsSpecificEngine state loaded: momentum=%s, alpha=%s",
                     self.momentum_overall, self.alpha)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the engine's persistent state for the snapshot.
        """
        return {
            "momentum_overall": self.momentum_overall,
            "alpha": self.alpha,
            "thresholds": self.thresholds
        }

    def calculate_metric_updates(self, metric_input: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates deltas for internal metrics based on new data.

        Expects metric_input to contain:
          - 'task_outcome': {'completed': bool, ...}
          - 'capacity': float (0–1)
          - 'shadow_score': float (0–1)

        Returns a dict of deltas, e.g. {'momentum_delta': 0.05}
        """
        self._last_inputs = metric_input.copy()

        # 1) Update momentum via EWMA on task success (1.0) or failure (0.0)
        completed = metric_input.get("task_outcome", {}).get("completed", False)
        success_val = 1.0 if completed else 0.0

        old_mu = self.momentum_overall
        new_mu = (1 - self.alpha) * old_mu + self.alpha * success_val
        momentum_delta = new_mu - old_mu

        logger.debug(
            "Momentum EWMA update: old=%.2f, success=%.1f, new=%.2f, Δ=%.3f",
            old_mu, success_val, new_mu, momentum_delta
        )

        return {"momentum_delta": momentum_delta}

    def apply_updates(self, deltas: Dict[str, float]):
        """
        Applies the computed deltas to the engine's state.
        """
        delta_mu = deltas.get("momentum_delta", 0.0)
        if delta_mu:
            self.momentum_overall = max(0.0, min(1.0, self.momentum_overall + delta_mu))
            logger.info("Applied momentum_delta=%.3f → momentum_overall=%.2f",
                        delta_mu, self.momentum_overall)

    def check_thresholds(self) -> Dict[str, bool]:
        """
        Emits boolean signals for narrative or interface triggers based on:
          - capacity (last_inputs['capacity'])
          - shadow_score (last_inputs['shadow_score'])
          - current momentum_overall
        """
        cap = self._last_inputs.get("capacity", 0.0)
        shadow = self._last_inputs.get("shadow_score", 0.0)

        signals = {
            "low_capacity": cap < self.thresholds["low_capacity"],
            "high_shadow": shadow > self.thresholds["high_shadow"],
            "low_momentum": self.momentum_overall < self.thresholds["low_momentum"]
        }
        logger.debug("Threshold signals: %s", signals)
        return signals
