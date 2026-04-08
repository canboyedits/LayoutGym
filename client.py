from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from openenv.core import EnvClient

try:
    from .models import DesignGymAction, DesignGymObservation, DesignGymState
except Exception:
    from models import DesignGymAction, DesignGymObservation, DesignGymState


# Local StepResult (since your OpenEnv version doesn't expose it)
@dataclass
class StepResult:
    observation: DesignGymObservation
    reward: Optional[float] = None
    done: bool = False


class DesignGymEnv(EnvClient[DesignGymAction, DesignGymObservation, DesignGymState]):

    def _step_payload(self, action: DesignGymAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_payload = payload.get("observation", payload)
        observation = DesignGymObservation(**obs_payload)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DesignGymState:
        return DesignGymState(**payload)