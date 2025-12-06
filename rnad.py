"""
Compatibility shim for RNAD checkpoints.

The provided RNAD pickles reference the module name `rnad` and expect classes
like `RNaDSolver`, `RNaDConfig`, `StateRepresentation`, and `AdamConfig`.
We do not ship the original training code, so this stub allows unpickling
and provides a minimal fallback `action_probabilities` that returns a
uniform distribution over the legal actions of the given OpenSpiel state.

This ensures RNAD agents can run without crashing, but does not reproduce
the original RNAD policy behavior.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RNaDConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class StateRepresentation:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class AdamConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class RNaDSolver:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Real weights/config are injected during unpickling via __setstate__
        self._initialized = True
        self._fallback_warned = False
        self.__dict__.update(kwargs)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Avoid repeated warnings if many games are run
        self._fallback_warned = False

    def action_probabilities(self, state) -> Dict[int, float]:
        """
        Fallback policy: uniform over legal actions.
        """
        try:
            legal = list(state.legal_actions())
        except Exception:
            legal = []
        if not legal:
            return {}
        if not self._fallback_warned:
            logger.warning(
                "RNAD fallback policy in use (uniform over legal actions); "
                "original rnad module not available."
            )
            self._fallback_warned = True
        p = 1.0 / len(legal)
        return {int(a): p for a in legal}

