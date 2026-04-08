from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DesignGymAction(BaseModel):
    action_type: str = "noop"
    element_id: Optional[str] = None
    element_ids: List[str] = Field(default_factory=list)
    group_id: Optional[str] = None
    region_id: Optional[str] = None
    template_id: Optional[str] = None
    pattern: Optional[str] = None
    axis: Optional[str] = None
    mode: Optional[str] = None
    anchor: str = "center"
    grid: int = 0
    dx: float = 0.0
    dy: float = 0.0
    dw: float = 0.0
    dh: float = 0.0
    strength: float = 0.0

    def canonical(self) -> str:
        return json.dumps(
            self.model_dump(exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )


class DesignGymObservation(BaseModel):
    message: str = ""
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 0
    done: bool = False
    reward: float = 0.0
    current_score: float = 0.0
    best_score_so_far: float = 0.0
    last_action_error: Optional[str] = None
    legal_actions: List[str] = Field(default_factory=list)
    layout_summary: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    metric_deltas: Dict[str, float] = Field(default_factory=dict)
    worst_metrics: List[str] = Field(default_factory=list)
    focus_elements: List[str] = Field(default_factory=list)
    element_blame: Dict[str, float] = Field(default_factory=dict)
    constraint_warnings: List[str] = Field(default_factory=list)
    suggested_edits: List[str] = Field(default_factory=list)


class DesignGymState(BaseModel):
    episode_id: str = ""
    seed: int = 0
    step_count: int = 0
    task_id: str = ""
    instance_id: str = ""
    max_steps: int = 0
    done: bool = False
    total_reward: float = 0.0
    last_reward: float = 0.0
    current_score: float = 0.0
    current_utility: float = 0.0
    best_score_so_far: float = 0.0
    last_action_error: Optional[str] = None
    invalid_actions: int = 0
    no_progress_steps: int = 0
    canvas: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    previous_metrics: Dict[str, float] = Field(default_factory=dict)
    metric_deltas: Dict[str, float] = Field(default_factory=dict)
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)