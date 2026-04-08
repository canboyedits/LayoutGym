from __future__ import annotations

import copy
import math
import random
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from openenv.core.env_server import Environment
except Exception:  # pragma: no cover
    class Environment:
        pass

try:
    from ..models import DesignGymAction, DesignGymObservation, DesignGymState
except Exception:  # pragma: no cover
    from models import DesignGymAction, DesignGymObservation, DesignGymState


EPS = 1e-9


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_exp(value: float) -> float:
    return math.exp(max(-50.0, min(50.0, value)))


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _intersect(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _anchors(box: Sequence[float]) -> Dict[str, float]:
    x, y, w, h = box
    return {
        "left": x,
        "center": x + w / 2.0,
        "right": x + w,
        "top": y,
        "middle": y + h / 2.0,
        "bottom": y + h,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _rank(values: Sequence[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    ma, mb = _mean(ra), _mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den_a = math.sqrt(sum((x - ma) ** 2 for x in ra))
    den_b = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if den_a <= EPS or den_b <= EPS:
        return 0.0
    return num / (den_a * den_b)


def _deepcopy_elements(elements: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [copy.deepcopy(e) for e in elements]


def _element_map(elements: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(e["id"]): e for e in elements}


def _el(
    element_id: str,
    role: str,
    typ: str,
    importance: float,
    group: str,
    content_len: int,
    min_size: List[float],
    max_size: List[float],
    aspect_ratio: Optional[float],
    precedence: int,
) -> Dict[str, object]:
    return {
        "id": element_id,
        "role": role,
        "type": typ,
        "importance": importance,
        "group": group,
        "content_len": content_len,
        "min_size": min_size,
        "max_size": max_size,
        "aspect_ratio": aspect_ratio,
        "precedence": precedence,
        "movable": True,
        "resizable": True,
    }


TASKS: Dict[str, Dict[str, object]] = {
    "poster_basic_v1": {
        "instance_id": "poster_basic_001",
        "max_steps": 7,
        "occupancy_target": 0.58,
        "occupancy_tolerance": 0.20,
        "init_noise": 0.020,
        "text_density_target": 0.62,
        "intent_regions": {
            "title": "top_band",
            "subtitle": "top_band",
            "hero_image": "hero_center",
            "cta": "safe_lower_right",
            "logo": "top_right",
            "badge": "right_column",
        },
        "weights": {
            "overlap": 0.14,
            "alignment": 0.12,
            "spacing": 0.09,
            "balance": 0.08,
            "hierarchy": 0.14,
            "grouping": 0.07,
            "reading_order": 0.08,
            "aspect_ratio": 0.08,
            "occupancy": 0.08,
            "text_fit": 0.04,
            "negative_space": 0.04,
            "intent_fit": 0.04,
        },
        "templates": {
            "hero": {
                "title": [0.08, 0.07, 0.68, 0.13],
                "subtitle": [0.08, 0.21, 0.56, 0.08],
                "hero_image": [0.08, 0.33, 0.64, 0.40],
                "cta": [0.08, 0.79, 0.28, 0.10],
                "logo": [0.78, 0.08, 0.14, 0.14],
                "badge": [0.74, 0.72, 0.18, 0.12],
            },
            "split": {
                "title": [0.06, 0.08, 0.42, 0.12],
                "subtitle": [0.06, 0.22, 0.36, 0.08],
                "hero_image": [0.52, 0.08, 0.40, 0.58],
                "cta": [0.06, 0.74, 0.26, 0.10],
                "logo": [0.06, 0.88, 0.12, 0.08],
                "badge": [0.66, 0.70, 0.18, 0.12],
            },
            "draft": {
                "title": [0.10, 0.08, 0.50, 0.12],
                "subtitle": [0.12, 0.24, 0.44, 0.08],
                "hero_image": [0.22, 0.34, 0.58, 0.36],
                "cta": [0.10, 0.76, 0.30, 0.10],
                "logo": [0.72, 0.10, 0.14, 0.14],
                "badge": [0.68, 0.72, 0.20, 0.14],
            },
        },
        "canvas": {"width": 1.0, "height": 1.0, "safe_margin": [0.04, 0.04, 0.04, 0.04], "forbidden_regions": []},
        "reading_order": [["title", "subtitle"], ["subtitle", "cta"]],
        "elements": [
            _el("title", "title", "text", 1.0, "headline", 46, [0.22, 0.08], [0.82, 0.18], None, 1),
            _el("subtitle", "subtitle", "text", 0.78, "headline", 74, [0.20, 0.06], [0.72, 0.12], None, 2),
            _el("hero_image", "image", "image", 0.92, "hero", 0, [0.30, 0.24], [0.82, 0.58], 1.6, 3),
            _el("cta", "cta", "text", 0.86, "footer", 18, [0.18, 0.08], [0.40, 0.14], None, 4),
            _el("logo", "logo", "image", 0.55, "brand", 0, [0.10, 0.08], [0.18, 0.18], 1.0, 5),
            _el("badge", "badge", "shape", 0.62, "support", 10, [0.12, 0.08], [0.24, 0.18], None, 6),
        ],
    },
    "editorial_cover_v1": {
        "instance_id": "editorial_cover_001",
        "max_steps": 9,
        "occupancy_target": 0.62,
        "occupancy_tolerance": 0.18,
        "init_noise": 0.018,
        "text_density_target": 0.70,
        "intent_regions": {
            "masthead": "top_band",
            "hero_image": "hero_center",
            "headline_1": "lower_left",
            "headline_2": "lower_left",
            "headline_3": "lower_left",
            "teaser": "right_column",
            "barcode": "footer_strip",
            "logo": "footer_left",
        },
        "weights": {
            "overlap": 0.12,
            "alignment": 0.11,
            "spacing": 0.09,
            "balance": 0.07,
            "hierarchy": 0.12,
            "grouping": 0.08,
            "reading_order": 0.13,
            "aspect_ratio": 0.06,
            "occupancy": 0.06,
            "text_fit": 0.05,
            "negative_space": 0.05,
            "intent_fit": 0.06,
        },
        "templates": {
            "editorial": {
                "masthead": [0.08, 0.05, 0.72, 0.10],
                "hero_image": [0.10, 0.18, 0.78, 0.44],
                "headline_1": [0.12, 0.66, 0.56, 0.10],
                "headline_2": [0.12, 0.77, 0.52, 0.08],
                "headline_3": [0.12, 0.86, 0.46, 0.06],
                "teaser": [0.72, 0.67, 0.16, 0.12],
                "barcode": [0.80, 0.88, 0.10, 0.08],
                "logo": [0.08, 0.88, 0.12, 0.08],
            },
            "grid": {
                "masthead": [0.08, 0.06, 0.70, 0.09],
                "hero_image": [0.08, 0.20, 0.44, 0.50],
                "headline_1": [0.56, 0.22, 0.30, 0.12],
                "headline_2": [0.56, 0.37, 0.28, 0.10],
                "headline_3": [0.56, 0.50, 0.26, 0.08],
                "teaser": [0.56, 0.64, 0.24, 0.12],
                "barcode": [0.78, 0.88, 0.12, 0.08],
                "logo": [0.08, 0.88, 0.12, 0.08],
            },
            "draft": {
                "masthead": [0.10, 0.06, 0.62, 0.10],
                "hero_image": [0.14, 0.22, 0.68, 0.38],
                "headline_1": [0.12, 0.63, 0.54, 0.10],
                "headline_2": [0.16, 0.76, 0.46, 0.08],
                "headline_3": [0.18, 0.86, 0.40, 0.06],
                "teaser": [0.72, 0.66, 0.16, 0.12],
                "barcode": [0.78, 0.88, 0.12, 0.08],
                "logo": [0.10, 0.88, 0.12, 0.08],
            },
        },
        "canvas": {"width": 1.0, "height": 1.0, "safe_margin": [0.04, 0.04, 0.04, 0.04], "forbidden_regions": []},
        "reading_order": [["masthead", "headline_1"], ["headline_1", "headline_2"], ["headline_2", "headline_3"]],
        "elements": [
            _el("masthead", "title", "text", 1.0, "header", 24, [0.40, 0.07], [0.82, 0.14], None, 1),
            _el("hero_image", "image", "image", 0.94, "hero", 0, [0.32, 0.28], [0.82, 0.58], None, 2),
            _el("headline_1", "title", "text", 0.88, "stories", 38, [0.28, 0.08], [0.64, 0.14], None, 3),
            _el("headline_2", "subtitle", "text", 0.78, "stories", 34, [0.26, 0.06], [0.56, 0.12], None, 4),
            _el("headline_3", "subtitle", "text", 0.68, "stories", 28, [0.20, 0.05], [0.48, 0.10], None, 5),
            _el("teaser", "badge", "text", 0.55, "support", 18, [0.12, 0.08], [0.28, 0.16], None, 6),
            _el("barcode", "caption", "shape", 0.25, "footer", 0, [0.08, 0.06], [0.16, 0.12], 1.5, 7),
            _el("logo", "logo", "image", 0.48, "brand", 0, [0.10, 0.06], [0.18, 0.12], 1.5, 8),
        ],
    },
    "dense_flyer_v1": {
        "instance_id": "dense_flyer_001",
        "max_steps": 10,
        "occupancy_target": 0.70,
        "occupancy_tolerance": 0.16,
        "init_noise": 0.016,
        "text_density_target": 0.76,
        "intent_regions": {
            "title": "top_band",
            "image_left": "left_column",
            "image_right": "right_column",
            "price_badge": "upper_right",
            "cta": "safe_lower_right",
            "details": "middle_band",
            "caption_1": "lower_left",
            "caption_2": "lower_right",
            "sponsor_strip": "footer_strip",
        },
        "weights": {
            "overlap": 0.12,
            "alignment": 0.11,
            "spacing": 0.11,
            "balance": 0.05,
            "hierarchy": 0.09,
            "grouping": 0.10,
            "reading_order": 0.09,
            "aspect_ratio": 0.05,
            "occupancy": 0.10,
            "text_fit": 0.06,
            "negative_space": 0.05,
            "intent_fit": 0.07,
        },
        "templates": {
            "grid": {
                "title": [0.06, 0.06, 0.60, 0.10],
                "image_left": [0.06, 0.20, 0.28, 0.24],
                "image_right": [0.38, 0.20, 0.28, 0.24],
                "price_badge": [0.72, 0.20, 0.18, 0.12],
                "cta": [0.72, 0.36, 0.18, 0.10],
                "details": [0.06, 0.50, 0.60, 0.16],
                "caption_1": [0.06, 0.70, 0.26, 0.10],
                "caption_2": [0.36, 0.70, 0.26, 0.10],
                "sponsor_strip": [0.06, 0.86, 0.84, 0.08],
            },
            "hero": {
                "title": [0.08, 0.06, 0.64, 0.11],
                "image_left": [0.08, 0.22, 0.36, 0.30],
                "image_right": [0.48, 0.22, 0.28, 0.22],
                "price_badge": [0.78, 0.22, 0.14, 0.12],
                "cta": [0.78, 0.38, 0.14, 0.10],
                "details": [0.08, 0.56, 0.56, 0.18],
                "caption_1": [0.08, 0.78, 0.24, 0.10],
                "caption_2": [0.36, 0.78, 0.24, 0.10],
                "sponsor_strip": [0.08, 0.90, 0.82, 0.06],
            },
            "draft": {
                "title": [0.08, 0.08, 0.56, 0.10],
                "image_left": [0.10, 0.24, 0.30, 0.22],
                "image_right": [0.42, 0.26, 0.30, 0.22],
                "price_badge": [0.74, 0.24, 0.16, 0.12],
                "cta": [0.70, 0.40, 0.20, 0.10],
                "details": [0.12, 0.52, 0.62, 0.18],
                "caption_1": [0.10, 0.74, 0.22, 0.10],
                "caption_2": [0.38, 0.74, 0.22, 0.10],
                "sponsor_strip": [0.10, 0.88, 0.78, 0.07],
            },
        },
        "canvas": {"width": 1.0, "height": 1.0, "safe_margin": [0.04, 0.04, 0.04, 0.04], "forbidden_regions": []},
        "reading_order": [["title", "details"], ["details", "cta"], ["cta", "sponsor_strip"]],
        "elements": [
            _el("title", "title", "text", 1.0, "headline", 42, [0.24, 0.08], [0.74, 0.14], None, 1),
            _el("image_left", "image", "image", 0.84, "visuals", 0, [0.20, 0.18], [0.42, 0.34], 1.2, 2),
            _el("image_right", "image", "image", 0.76, "visuals", 0, [0.20, 0.18], [0.40, 0.34], 1.2, 3),
            _el("price_badge", "badge", "shape", 0.82, "conversion", 10, [0.12, 0.08], [0.22, 0.16], None, 4),
            _el("cta", "cta", "text", 0.90, "conversion", 16, [0.14, 0.08], [0.26, 0.14], None, 5),
            _el("details", "body", "text", 0.72, "details", 160, [0.34, 0.12], [0.72, 0.24], None, 6),
            _el("caption_1", "caption", "text", 0.44, "support", 24, [0.18, 0.06], [0.30, 0.12], None, 7),
            _el("caption_2", "caption", "text", 0.40, "support", 22, [0.18, 0.06], [0.30, 0.12], None, 8),
            _el("sponsor_strip", "caption", "shape", 0.30, "footer", 0, [0.46, 0.05], [0.90, 0.12], None, 9),
        ],
    },
}


class DesignGymEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = DesignGymState()
        self._task_spec: Dict[str, object] = {}

    def _ensure_task_spec(self) -> None:
        task_id = getattr(self._state, "task_id", "") or "poster_basic_v1"
        if "templates" not in self._task_spec:
            self._task_spec = copy.deepcopy(TASKS.get(task_id, TASKS["poster_basic_v1"]))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> DesignGymObservation:
        selected_task = task_id or kwargs.get("task_id") or "poster_basic_v1"
        if selected_task not in TASKS:
            selected_task = "poster_basic_v1"

        self._task_spec = copy.deepcopy(TASKS[selected_task])
        local_seed = int(seed if seed is not None else kwargs.get("seed", 0) or 0)
        rng = random.Random(local_seed)

        initial_template = str(kwargs.get("template_id") or "draft")
        if initial_template not in self._task_spec["templates"]:
            initial_template = "draft"

        elements = self._build_initial_elements(self._task_spec, initial_template)
        elements = self._apply_seeded_imperfections(elements, rng)

        self._state = DesignGymState(
            episode_id=episode_id or str(uuid.uuid4()),
            seed=local_seed,
            step_count=0,
            task_id=selected_task,
            instance_id=str(self._task_spec["instance_id"]),
            max_steps=int(self._task_spec["max_steps"]),
            done=False,
            total_reward=0.0,
            last_reward=0.0,
            current_score=0.0,
            current_utility=0.0,
            best_score_so_far=0.0,
            last_action_error=None,
            invalid_actions=0,
            no_progress_steps=0,
            canvas=copy.deepcopy(self._task_spec["canvas"]),
            constraints={
                "reading_order": copy.deepcopy(self._task_spec["reading_order"]),
                "occupancy_target": self._task_spec["occupancy_target"],
                "required_elements": [e["id"] for e in self._task_spec["elements"]],
                "templates": list(self._task_spec["templates"].keys()),
                "intent_regions": copy.deepcopy(self._task_spec["intent_regions"]),
            },
            metrics={},
            previous_metrics={},
            metric_deltas={},
            elements=elements,
            action_history=[],
        )

        score_info = self._score_layout(self._state.elements)
        self._state.metrics = score_info["metrics"]
        self._state.previous_metrics = dict(score_info["metrics"])
        self._state.metric_deltas = {k: 0.0 for k in score_info["metrics"]}
        self._state.current_utility = float(score_info["utility"])
        self._state.current_score = float(score_info["score"])
        self._state.best_score_so_far = float(score_info["utility"])

        return self._observation(message=f"Ready: {selected_task}")

    @property
    def state(self) -> DesignGymState:
        return self._state

    def step(self, action: DesignGymAction, timeout_s: Optional[int] = None, **kwargs) -> DesignGymObservation:
        self._ensure_task_spec()

        if self._state.done:
            self._state.last_reward = 0.0
            self._state.last_action_error = "episode_already_done"
            return self._observation(message="Episode already finished.")

        canonical_action = action.canonical() if hasattr(action, "canonical") else str(action.action_type)
        proposed_elements = _deepcopy_elements(self._state.elements)

        if action.action_type == "finalize":
            self._state.done = True
            self._state.last_reward = 0.0
            self._state.last_action_error = None
            self._state.action_history.append(canonical_action)
            return self._observation(message="Layout finalized.")

        ok, error = self._apply_action(proposed_elements, action)
        self._state.step_count += 1
        self._state.action_history.append(canonical_action)

        if not ok:
            self._state.invalid_actions += 1
            self._state.no_progress_steps += 1
            self._state.last_reward = 0.0
            self._state.last_action_error = error
            if self._state.step_count >= self._state.max_steps:
                self._state.done = True
            return self._observation(message="Action rejected.")

        hard_valid, hard_error = self._check_hard_constraints(proposed_elements)
        if not hard_valid:
            self._state.invalid_actions += 1
            self._state.no_progress_steps += 1
            self._state.last_reward = 0.0
            self._state.last_action_error = hard_error
            if self._state.step_count >= self._state.max_steps:
                self._state.done = True
            return self._observation(message="Constraint violation; reverted.")

        prev_score = float(self._state.current_score)
        prev_utility = float(self._state.current_utility)
        prev_best = float(self._state.best_score_so_far)
        prev_metrics = dict(self._state.metrics)

        score_info = self._score_layout(proposed_elements)
        curr_score = float(score_info["score"])
        curr_utility = float(score_info["utility"])
        curr_metrics = dict(score_info["metrics"])

        neighborhood = self._neighborhood_utilities(
            base_elements=self._state.elements,
            focus_metrics=self._worst_metrics(prev_metrics, k=2),
        )
        pref_rank = self._percentile_rank(curr_utility, neighborhood)

        step_gain = max(0.0, curr_score - prev_score)
        best_gain = max(0.0, curr_utility - prev_best)
        frontier_keys = self._worst_metrics(prev_metrics, k=2)
        frontier_gain = _mean([max(0.0, curr_metrics.get(k, 0.0) - prev_metrics.get(k, 0.0)) for k in frontier_keys])

        oscillation_penalty = self._oscillation_penalty(action)
        waste_penalty = 0.03 if step_gain <= 1e-6 and best_gain <= 1e-6 and frontier_gain <= 1e-6 else 0.0

        reward = _clamp(
            0.45 * step_gain
            + 0.20 * best_gain
            + 0.20 * frontier_gain
            + 0.15 * pref_rank
            - oscillation_penalty
            - waste_penalty,
            0.0,
            1.0,
        )

        if step_gain <= 1e-6 and best_gain <= 1e-6:
            self._state.no_progress_steps += 1
        else:
            self._state.no_progress_steps = 0

        self._state.previous_metrics = prev_metrics
        self._state.metric_deltas = {
            key: round(curr_metrics.get(key, 0.0) - prev_metrics.get(key, 0.0), 6)
            for key in curr_metrics
        }
        self._state.best_score_so_far = max(prev_best, curr_utility)
        self._state.elements = proposed_elements
        self._state.metrics = curr_metrics
        self._state.current_utility = curr_utility

        efficiency = max(0.70, 1.0 - 0.05 * self._state.invalid_actions - 0.02 * self._state.no_progress_steps)
        self._state.current_score = _clamp(curr_utility * efficiency, 0.0, 1.0)

        self._state.total_reward = _clamp(self._state.total_reward + reward, 0.0, 1.0)
        self._state.last_reward = reward
        self._state.last_action_error = None

        if self._state.step_count >= self._state.max_steps:
            self._state.done = True
            return self._observation(message="Max steps reached.")

        return self._observation(message="Action applied.")

    def _build_initial_elements(self, task_spec: Dict[str, object], template_name: str) -> List[Dict[str, object]]:
        template = task_spec["templates"][template_name]
        elements: List[Dict[str, object]] = []
        for z, base in enumerate(task_spec["elements"], start=1):
            bbox = list(template[base["id"]])
            elements.append(
                {
                    "id": base["id"],
                    "role": base["role"],
                    "type": base["type"],
                    "importance": float(base["importance"]),
                    "group": base["group"],
                    "content_len": int(base.get("content_len", 0)),
                    "bbox": bbox,
                    "z": z,
                    "min_size": list(base["min_size"]),
                    "max_size": list(base["max_size"]),
                    "aspect_ratio": base["aspect_ratio"],
                    "precedence": int(base["precedence"]),
                    "movable": bool(base["movable"]),
                    "resizable": bool(base["resizable"]),
                    "placed": True,
                }
            )
        return elements

    def _apply_seeded_imperfections(self, elements: List[Dict[str, object]], rng: random.Random) -> List[Dict[str, object]]:
        intensity = float(self._task_spec.get("init_noise", 0.018))
        trial = _deepcopy_elements(elements)
        by_id = _element_map(trial)

        for e in trial:
            if not e.get("movable", False):
                continue
            e["bbox"][0] += rng.uniform(-intensity, intensity)
            e["bbox"][1] += rng.uniform(-intensity, intensity)

        task_id = str(self._task_spec["instance_id"])
        if "poster_basic" in task_id:
            by_id["subtitle"]["bbox"][0] += 0.045
            by_id["cta"]["bbox"][0] -= 0.035
            by_id["hero_image"]["bbox"][2] -= 0.06
            by_id["badge"]["bbox"][1] -= 0.03
        elif "editorial" in task_id:
            by_id["headline_2"]["bbox"][0] += 0.05
            by_id["headline_3"]["bbox"][0] += 0.07
            by_id["teaser"]["bbox"][1] += 0.03
            by_id["masthead"]["bbox"][2] -= 0.06
        else:
            by_id["caption_1"]["bbox"][1] -= 0.04
            by_id["caption_2"]["bbox"][1] -= 0.01
            by_id["cta"]["bbox"][0] -= 0.04
            by_id["details"]["bbox"][0] += 0.04
            by_id["details"]["bbox"][2] -= 0.10

        self._repair_layout_in_place(trial)
        return trial

    def _repair_layout_in_place(self, elements: List[Dict[str, object]]) -> None:
        left_m, top_m, right_m, bottom_m = [float(v) for v in self._task_spec["canvas"]["safe_margin"]]
        for e in elements:
            x, y, w, h = [float(v) for v in e["bbox"]]
            min_w, min_h = [float(v) for v in e["min_size"]]
            max_w, max_h = [float(v) for v in e["max_size"]]

            w = _clamp(w, min_w, max_w)
            h = _clamp(h, min_h, max_h)

            ar = e.get("aspect_ratio")
            if ar:
                target_w = _clamp(h * float(ar), min_w, max_w)
                target_h = _clamp(w / float(ar), min_h, max_h)
                if abs(target_w - w) <= abs(target_h - h) * float(ar):
                    w = target_w
                else:
                    h = target_h

            x = _clamp(x, left_m, 1.0 - right_m - w)
            y = _clamp(y, top_m, 1.0 - bottom_m - h)
            e["bbox"] = [x, y, w, h]

    def _observation(self, message: str) -> DesignGymObservation:
        summary_lines = []
        for e in sorted(self._state.elements, key=lambda item: item["z"]):
            x, y, w, h = e["bbox"]
            summary_lines.append(f"{e['id']}@({x:.2f},{y:.2f},{w:.2f},{h:.2f})")

        blame = self._element_blame(self._state.elements)
        focus = [k for k, _ in sorted(blame.items(), key=lambda item: item[1], reverse=True)[:3]]
        warnings = self._constraint_warnings(self._state.elements)
        worst = self._worst_metrics(self._state.metrics, k=3)

        return DesignGymObservation(
            message=message,
            task_id=self._state.task_id,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            done=self._state.done,
            reward=_clamp(self._state.last_reward, 0.0, 1.0),
            current_score=_clamp(self._state.current_score, 0.0, 1.0),
            best_score_so_far=_clamp(self._state.best_score_so_far, 0.0, 1.0),
            last_action_error=self._state.last_action_error,
            legal_actions=[
                "apply_template(template_id)",
                "move(element_id, dx, dy)",
                "resize(element_id, dw, dh, anchor)",
                "align(element_ids, axis, mode)",
                "distribute(element_ids, axis)",
                "swap_z(element_ids[0], element_ids[1])",
                "snap(element_id, grid)",
                "promote(element_id, strength)",
                "reflow_group(group_id, pattern)",
                "anchor_to_region(element_id, region_id, mode)",
                "finalize()",
            ],
            layout_summary="; ".join(summary_lines),
            metrics={k: round(float(v), 4) for k, v in self._state.metrics.items()},
            metric_deltas={k: round(float(v), 4) for k, v in self._state.metric_deltas.items()},
            worst_metrics=worst,
            focus_elements=focus,
            element_blame={k: round(float(v), 4) for k, v in blame.items()},
            constraint_warnings=warnings,
            suggested_edits=self._suggested_edits(worst, focus),
        )

    def _apply_action(self, elements: List[Dict[str, object]], action: DesignGymAction) -> Tuple[bool, Optional[str]]:
        by_id = _element_map(elements)

        if action.action_type == "apply_template":
            template_id = action.template_id or "draft"
            templates = self._task_spec["templates"]
            if template_id not in templates:
                return False, "unknown_template"
            for e in elements:
                e["bbox"] = list(templates[template_id][e["id"]])
            return True, None

        if action.action_type == "move":
            if not action.element_id or action.element_id not in by_id:
                return False, "unknown_element"
            e = by_id[action.element_id]
            if not e["movable"]:
                return False, "element_not_movable"
            x, y, w, h = e["bbox"]
            e["bbox"] = [x + action.dx, y + action.dy, w, h]
            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "resize":
            if not action.element_id or action.element_id not in by_id:
                return False, "unknown_element"
            e = by_id[action.element_id]
            if not e["resizable"]:
                return False, "element_not_resizable"

            x, y, w, h = e["bbox"]
            new_w = w + action.dw
            new_h = h + action.dh

            if action.anchor == "center":
                x -= action.dw / 2.0
                y -= action.dh / 2.0
            elif action.anchor == "east":
                y -= action.dh / 2.0
            elif action.anchor == "south":
                x -= action.dw / 2.0
            elif action.anchor == "ne":
                y -= action.dh
            elif action.anchor == "nw":
                x -= action.dw
                y -= action.dh
            elif action.anchor == "sw":
                x -= action.dw
            elif action.anchor == "north":
                x -= action.dw / 2.0
                y -= action.dh
            elif action.anchor == "west":
                x -= action.dw
                y -= action.dh / 2.0

            e["bbox"] = [x, y, new_w, new_h]
            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "align":
            ids = [i for i in action.element_ids if i in by_id]
            if len(ids) < 2:
                return False, "align_needs_two_or_more_elements"

            boxes = [by_id[i]["bbox"] for i in ids]

            if action.axis == "x":
                if action.mode == "left":
                    target = min(b[0] for b in boxes)
                    for i in ids:
                        by_id[i]["bbox"][0] = target
                elif action.mode == "center":
                    target = _mean([b[0] + b[2] / 2.0 for b in boxes])
                    for i in ids:
                        by_id[i]["bbox"][0] = target - by_id[i]["bbox"][2] / 2.0
                elif action.mode == "right":
                    target = max(b[0] + b[2] for b in boxes)
                    for i in ids:
                        by_id[i]["bbox"][0] = target - by_id[i]["bbox"][2]
                else:
                    return False, "invalid_align_mode"
            elif action.axis == "y":
                if action.mode == "top":
                    target = min(b[1] for b in boxes)
                    for i in ids:
                        by_id[i]["bbox"][1] = target
                elif action.mode == "middle":
                    target = _mean([b[1] + b[3] / 2.0 for b in boxes])
                    for i in ids:
                        by_id[i]["bbox"][1] = target - by_id[i]["bbox"][3] / 2.0
                elif action.mode == "bottom":
                    target = max(b[1] + b[3] for b in boxes)
                    for i in ids:
                        by_id[i]["bbox"][1] = target - by_id[i]["bbox"][3]
                else:
                    return False, "invalid_align_mode"
            else:
                return False, "invalid_axis"

            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "distribute":
            ids = [i for i in action.element_ids if i in by_id]
            if len(ids) < 3:
                return False, "distribute_needs_three_or_more_elements"

            if action.axis == "x":
                ids.sort(key=lambda i: by_id[i]["bbox"][0])
                left = by_id[ids[0]]["bbox"][0]
                right = by_id[ids[-1]]["bbox"][0] + by_id[ids[-1]]["bbox"][2]
                total_w = sum(by_id[i]["bbox"][2] for i in ids)
                gap = (right - left - total_w) / (len(ids) - 1)
                if gap < -EPS:
                    return False, "negative_distribution_gap"
                cursor = left
                for i in ids:
                    by_id[i]["bbox"][0] = cursor
                    cursor += by_id[i]["bbox"][2] + gap
            elif action.axis == "y":
                ids.sort(key=lambda i: by_id[i]["bbox"][1])
                top = by_id[ids[0]]["bbox"][1]
                bottom = by_id[ids[-1]]["bbox"][1] + by_id[ids[-1]]["bbox"][3]
                total_h = sum(by_id[i]["bbox"][3] for i in ids)
                gap = (bottom - top - total_h) / (len(ids) - 1)
                if gap < -EPS:
                    return False, "negative_distribution_gap"
                cursor = top
                for i in ids:
                    by_id[i]["bbox"][1] = cursor
                    cursor += by_id[i]["bbox"][3] + gap
            else:
                return False, "invalid_axis"

            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "swap_z":
            ids = [i for i in action.element_ids if i in by_id]
            if len(ids) != 2:
                return False, "swap_z_needs_exactly_two_elements"
            by_id[ids[0]]["z"], by_id[ids[1]]["z"] = by_id[ids[1]]["z"], by_id[ids[0]]["z"]
            return True, None

        if action.action_type == "snap":
            if not action.element_id or action.element_id not in by_id:
                return False, "unknown_element"
            grid = int(action.grid)
            if grid <= 0:
                return False, "grid_must_be_positive"
            e = by_id[action.element_id]
            x, y, w, h = e["bbox"]
            e["bbox"] = [round(x * grid) / grid, round(y * grid) / grid, round(w * grid) / grid, round(h * grid) / grid]
            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "promote":
            if not action.element_id or action.element_id not in by_id:
                return False, "unknown_element"
            e = by_id[action.element_id]
            strength = action.strength if abs(action.strength) > EPS else 0.06
            x, y, w, h = [float(v) for v in e["bbox"]]
            grow = abs(strength)

            if e["type"] == "text":
                e["bbox"] = [x - 0.5 * grow, y - 0.25 * grow, w + grow, h + 0.5 * grow]
            else:
                e["bbox"] = [x - 0.4 * grow, y - 0.4 * grow, w + 0.8 * grow, h + 0.8 * grow]

            e["z"] = max(int(item["z"]) for item in elements)
            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "reflow_group":
            if not action.group_id:
                return False, "missing_group_id"
            members = [e for e in elements if str(e["group"]) == action.group_id]
            if len(members) < 2:
                return False, "group_not_found_or_too_small"

            pattern = action.pattern or "stack"
            xs = [e["bbox"][0] for e in members]
            ys = [e["bbox"][1] for e in members]
            rights = [e["bbox"][0] + e["bbox"][2] for e in members]
            bottoms = [e["bbox"][1] + e["bbox"][3] for e in members]
            left, top = min(xs), min(ys)
            width, height = max(rights) - left, max(bottoms) - top
            ordered = sorted(members, key=lambda e: (e["precedence"], e["id"]))

            if pattern == "stack":
                gap = max(0.012, (height - sum(e["bbox"][3] for e in ordered)) / max(1, len(ordered) - 1))
                cursor = top
                for e in ordered:
                    e["bbox"][0] = left
                    e["bbox"][1] = cursor
                    cursor += e["bbox"][3] + gap
            elif pattern == "row":
                gap = max(0.012, (width - sum(e["bbox"][2] for e in ordered)) / max(1, len(ordered) - 1))
                cursor = left
                for e in ordered:
                    e["bbox"][0] = cursor
                    e["bbox"][1] = top
                    cursor += e["bbox"][2] + gap
            elif pattern == "grid2":
                col_w = max(e["bbox"][2] for e in ordered)
                row_h = max(e["bbox"][3] for e in ordered)
                for idx, e in enumerate(ordered):
                    row = idx // 2
                    col = idx % 2
                    e["bbox"][0] = left + col * (col_w + 0.018)
                    e["bbox"][1] = top + row * (row_h + 0.018)
            elif pattern == "sidebar":
                col_x = max(0.52, left)
                cursor = top
                for e in ordered:
                    e["bbox"][0] = col_x
                    e["bbox"][1] = cursor
                    cursor += e["bbox"][3] + 0.016
            else:
                return False, "unknown_reflow_pattern"

            self._repair_layout_in_place(elements)
            return True, None

        if action.action_type == "anchor_to_region":
            if not action.element_id or action.element_id not in by_id:
                return False, "unknown_element"
            if not action.region_id:
                return False, "missing_region_id"

            region = self._region_boxes().get(action.region_id)
            if region is None:
                return False, "unknown_region"

            e = by_id[action.element_id]
            ex, ey, ew, eh = [float(v) for v in e["bbox"]]
            rx, ry, rw, rh = region
            mode = action.mode or "center"

            if mode == "fill":
                ew = min(ew, rw)
                eh = min(eh, rh)
                ex = rx + (rw - ew) / 2.0
                ey = ry + (rh - eh) / 2.0
            elif mode == "start":
                ex = rx
                ey = ry
            elif mode == "end":
                ex = rx + rw - ew
                ey = ry + rh - eh
            else:
                ex = rx + (rw - ew) / 2.0
                ey = ry + (rh - eh) / 2.0

            e["bbox"] = [ex, ey, ew, eh]
            self._repair_layout_in_place(elements)
            return True, None

        return False, "unknown_action_type"

    def _check_hard_constraints(self, elements: List[Dict[str, object]]) -> Tuple[bool, Optional[str]]:
        left_m, top_m, right_m, bottom_m = [float(v) for v in self._task_spec["canvas"]["safe_margin"]]

        for e in elements:
            x, y, w, h = [float(v) for v in e["bbox"]]
            min_w, min_h = [float(v) for v in e["min_size"]]
            max_w, max_h = [float(v) for v in e["max_size"]]

            if w < min_w - EPS or h < min_h - EPS:
                return False, f"min_size_violation:{e['id']}"
            if w > max_w + EPS or h > max_h + EPS:
                return False, f"max_size_violation:{e['id']}"
            if x < left_m - EPS or y < top_m - EPS:
                return False, f"outside_safe_region:{e['id']}"
            if x + w > 1.0 - right_m + EPS or y + h > 1.0 - bottom_m + EPS:
                return False, f"outside_safe_region:{e['id']}"

            ar = e.get("aspect_ratio")
            if ar:
                ratio = w / max(h, EPS)
                if abs(math.log(ratio / float(ar))) > 0.18:
                    return False, f"aspect_ratio_violation:{e['id']}"

        for region in self._task_spec["canvas"].get("forbidden_regions", []):
            for e in elements:
                if _intersect(e["bbox"], region) > EPS:
                    return False, f"forbidden_region_overlap:{e['id']}"

        return True, None

    def _score_layout(self, elements: List[Dict[str, object]]) -> Dict[str, object]:
        hard_valid, _ = self._check_hard_constraints(elements)

        metrics = {
            "overlap": self._metric_overlap(elements),
            "alignment": self._metric_alignment(elements),
            "spacing": self._metric_spacing(elements),
            "balance": self._metric_balance(elements),
            "hierarchy": self._metric_hierarchy(elements),
            "grouping": self._metric_grouping(elements),
            "reading_order": self._metric_reading_order(elements),
            "aspect_ratio": self._metric_aspect_ratio(elements),
            "occupancy": self._metric_occupancy(elements),
            "text_fit": self._metric_text_fit(elements),
            "negative_space": self._metric_negative_space(elements),
            "intent_fit": self._metric_intent_fit(elements),
        }

        utility = 0.0
        for key, weight in self._task_spec["weights"].items():
            utility += float(weight) * float(metrics[key])

        utility = _clamp(utility, 0.0, 1.0)
        score = utility if hard_valid else 0.0
        return {"utility": utility, "score": score, "metrics": metrics}

    def _metric_overlap(self, elements: List[Dict[str, object]]) -> float:
        total_overlap = 0.0
        total_area = 0.0
        for i, a in enumerate(elements):
            total_area += _area(a["bbox"])
            for b in elements[i + 1:]:
                total_overlap += _intersect(a["bbox"], b["bbox"])
        return _clamp(_safe_exp(-(total_overlap / (total_area + EPS))), 0.0, 1.0)

    def _metric_alignment(self, elements: List[Dict[str, object]]) -> float:
        if len(elements) < 2:
            return 0.5

        canvas_guides_x = [0.04, 0.50, 0.96]
        canvas_guides_y = [0.04, 0.50, 0.96]
        distances: List[float] = []

        for i, e in enumerate(elements):
            anchors = _anchors(e["bbox"])
            other_x = canvas_guides_x[:]
            other_y = canvas_guides_y[:]

            for j, o in enumerate(elements):
                if i == j:
                    continue
                oa = _anchors(o["bbox"])
                other_x.extend([oa["left"], oa["center"], oa["right"]])
                other_y.extend([oa["top"], oa["middle"], oa["bottom"]])

            for name in ("left", "center", "right"):
                distances.append(min(abs(anchors[name] - g) for g in other_x))
            for name in ("top", "middle", "bottom"):
                distances.append(min(abs(anchors[name] - g) for g in other_y))

        return _clamp(_mean([_safe_exp(-d / 0.055) for d in distances]), 0.0, 1.0)

    def _metric_spacing(self, elements: List[Dict[str, object]]) -> float:
        gaps: List[float] = []
        xs = sorted(elements, key=lambda e: e["bbox"][0])
        ys = sorted(elements, key=lambda e: e["bbox"][1])

        for items, axis in ((xs, "x"), (ys, "y")):
            for a, b in zip(items, items[1:]):
                if axis == "x":
                    gap = b["bbox"][0] - (a["bbox"][0] + a["bbox"][2])
                    overlap_other = min(a["bbox"][1] + a["bbox"][3], b["bbox"][1] + b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1])
                else:
                    gap = b["bbox"][1] - (a["bbox"][1] + a["bbox"][3])
                    overlap_other = min(a["bbox"][0] + a["bbox"][2], b["bbox"][0] + b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0])

                if gap > 0 and overlap_other > 0:
                    gaps.append(gap)

        if len(gaps) < 2:
            return 0.5

        cv = _std(gaps) / (_mean(gaps) + EPS)
        return _clamp(_safe_exp(-(cv / 0.70)), 0.0, 1.0)

    def _metric_balance(self, elements: List[Dict[str, object]]) -> float:
        masses = []
        centers = []
        for e in elements:
            a = _area(e["bbox"])
            masses.append(a * float(e["importance"]))
            centers.append(_center(e["bbox"]))

        total_mass = sum(masses)
        if total_mass <= EPS:
            return 0.0

        cx = sum(m * c[0] for m, c in zip(masses, centers)) / total_mass
        cy = sum(m * c[1] for m, c in zip(masses, centers)) / total_mass
        dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        return _clamp(_safe_exp(-(dist / 0.22)), 0.0, 1.0)

    def _metric_hierarchy(self, elements: List[Dict[str, object]]) -> float:
        importance = []
        salience = []
        for e in elements:
            x, y, w, h = e["bbox"]
            a = _area(e["bbox"])
            focus_x = 1.0 - abs((x + w / 2.0) - 0.5) / 0.5
            zeta = 0.55 * math.log(a + 1e-4) - 0.22 * y + 0.18 * focus_x + 0.10 * e["z"] / max(1, len(elements))
            importance.append(float(e["importance"]))
            salience.append(zeta)
        rho = _spearman(importance, salience)
        return _clamp((1.0 + rho) / 2.0, 0.0, 1.0)

    def _metric_grouping(self, elements: List[Dict[str, object]]) -> float:
        groups: Dict[str, List[Tuple[float, float]]] = {}
        for e in elements:
            groups.setdefault(str(e["group"]), []).append(_center(e["bbox"]))

        if len(groups) < 2:
            return 0.5

        within = []
        group_centers: List[Tuple[float, float]] = []
        for centers in groups.values():
            gx = _mean([c[0] for c in centers])
            gy = _mean([c[1] for c in centers])
            group_centers.append((gx, gy))
            within.append(_mean([math.dist(c, (gx, gy)) for c in centers]))

        between = []
        for i, c1 in enumerate(group_centers):
            for c2 in group_centers[i + 1:]:
                between.append(math.dist(c1, c2))

        within_term = _safe_exp(-(_mean(within) / 0.22))
        between_term = 1.0 - _safe_exp(-(_mean(between) / 0.28))
        return _clamp(within_term * between_term, 0.0, 1.0)

    def _metric_reading_order(self, elements: List[Dict[str, object]]) -> float:
        if not self._task_spec.get("reading_order"):
            return 0.5

        by_id = _element_map(elements)
        good = 0
        total = 0

        for first_id, second_id in self._task_spec["reading_order"]:
            if first_id not in by_id or second_id not in by_id:
                continue

            total += 1
            a = by_id[first_id]["bbox"]
            b = by_id[second_id]["bbox"]

            if abs(a[1] - b[1]) <= 0.05:
                ok = a[0] <= b[0]
            else:
                ok = a[1] <= b[1]

            good += 1 if ok else 0

        return _clamp(good / total if total else 0.5, 0.0, 1.0)

    def _metric_aspect_ratio(self, elements: List[Dict[str, object]]) -> float:
        locked = [e for e in elements if e.get("aspect_ratio")]
        if not locked:
            return 1.0
        penalties = []
        for e in locked:
            w, h = e["bbox"][2], e["bbox"][3]
            penalties.append(abs(math.log((w / max(h, EPS)) / float(e["aspect_ratio"]))))
        return _clamp(_safe_exp(-_mean(penalties) / 0.9), 0.0, 1.0)

    def _metric_occupancy(self, elements: List[Dict[str, object]]) -> float:
        occ = sum(_area(e["bbox"]) for e in elements)
        target = float(self._task_spec["occupancy_target"])
        tol = float(self._task_spec["occupancy_tolerance"])
        return _clamp(max(0.0, 1.0 - abs(occ - target) / max(tol, EPS)), 0.0, 1.0)

    def _metric_text_fit(self, elements: List[Dict[str, object]]) -> float:
        penalties = []
        target = float(self._task_spec.get("text_density_target", 0.68))
        for e in elements:
            if e["type"] != "text":
                continue
            w, h = float(e["bbox"][2]), float(e["bbox"][3])
            capacity = max(EPS, w * h * 900.0)
            demand = float(e.get("content_len", 0))
            ratio = demand / capacity
            penalties.append(abs(ratio - target))
        if not penalties:
            return 1.0
        return _clamp(_safe_exp(-_mean(penalties) / 0.45), 0.0, 1.0)

    def _metric_negative_space(self, elements: List[Dict[str, object]]) -> float:
        occ = sum(_area(e["bbox"]) for e in elements)
        whitespace = max(0.0, 1.0 - occ)

        xs = sorted([e["bbox"][0] for e in elements] + [e["bbox"][0] + e["bbox"][2] for e in elements])
        ys = sorted([e["bbox"][1] for e in elements] + [e["bbox"][1] + e["bbox"][3] for e in elements])

        x_gaps = [max(0.0, b - a) for a, b in zip(xs, xs[1:])]
        y_gaps = [max(0.0, b - a) for a, b in zip(ys, ys[1:])]

        rhythm = 1.0 - min(1.0, (_std(x_gaps) + _std(y_gaps)) / 0.18) if x_gaps and y_gaps else 0.5
        whitespace_term = 1.0 - abs(whitespace - (1.0 - float(self._task_spec["occupancy_target"]))) / 0.30
        return _clamp(0.6 * rhythm + 0.4 * max(0.0, whitespace_term), 0.0, 1.0)

    def _metric_intent_fit(self, elements: List[Dict[str, object]]) -> float:
        regions = self._region_boxes()
        intent_regions = self._task_spec.get("intent_regions", {})
        scores = []

        for e in elements:
            region_id = intent_regions.get(e["id"])
            if not region_id or region_id not in regions:
                continue

            rx, ry, rw, rh = regions[region_id]
            cx, cy = _center(e["bbox"])
            tx, ty = rx + rw / 2.0, ry + rh / 2.0
            dist = math.dist((cx, cy), (tx, ty))
            diag = math.sqrt(rw * rw + rh * rh) + EPS
            scores.append(_safe_exp(-(dist / diag) / 0.65))

        return _clamp(_mean(scores) if scores else 0.5, 0.0, 1.0)

    def _region_boxes(self) -> Dict[str, List[float]]:
        left_m, top_m, right_m, bottom_m = [float(v) for v in self._task_spec["canvas"]["safe_margin"]]
        usable_x = left_m
        usable_y = top_m
        usable_w = 1.0 - left_m - right_m
        usable_h = 1.0 - top_m - bottom_m

        return {
            "top_band": [usable_x, usable_y, usable_w, usable_h * 0.18],
            "hero_center": [usable_x + usable_w * 0.12, usable_y + usable_h * 0.18, usable_w * 0.58, usable_h * 0.46],
            "left_column": [usable_x, usable_y + usable_h * 0.18, usable_w * 0.40, usable_h * 0.58],
            "right_column": [usable_x + usable_w * 0.60, usable_y + usable_h * 0.18, usable_w * 0.28, usable_h * 0.58],
            "upper_right": [usable_x + usable_w * 0.68, usable_y + usable_h * 0.12, usable_w * 0.24, usable_h * 0.18],
            "middle_band": [usable_x + usable_w * 0.08, usable_y + usable_h * 0.45, usable_w * 0.60, usable_h * 0.20],
            "lower_left": [usable_x + usable_w * 0.08, usable_y + usable_h * 0.60, usable_w * 0.44, usable_h * 0.22],
            "lower_right": [usable_x + usable_w * 0.54, usable_y + usable_h * 0.60, usable_w * 0.30, usable_h * 0.22],
            "footer_strip": [usable_x, usable_y + usable_h * 0.86, usable_w, usable_h * 0.10],
            "footer_left": [usable_x, usable_y + usable_h * 0.84, usable_w * 0.24, usable_h * 0.12],
            "top_right": [usable_x + usable_w * 0.72, usable_y, usable_w * 0.20, usable_h * 0.18],
            "safe_lower_right": [usable_x + usable_w * 0.64, usable_y + usable_h * 0.66, usable_w * 0.24, usable_h * 0.20],
        }

    def _worst_metrics(self, metrics: Dict[str, float], k: int = 3) -> List[str]:
        return [name for name, _ in sorted(metrics.items(), key=lambda item: item[1])[:k]]

    def _element_blame(self, elements: List[Dict[str, object]]) -> Dict[str, float]:
        by_id = _element_map(elements)
        blame = {str(e["id"]): 0.0 for e in elements}

        for i, a in enumerate(elements):
            for b in elements[i + 1:]:
                overlap = _intersect(a["bbox"], b["bbox"])
                if overlap > EPS:
                    norm = overlap / max(EPS, min(_area(a["bbox"]), _area(b["bbox"])))
                    blame[str(a["id"])] += norm
                    blame[str(b["id"])] += norm

        for e in elements:
            ea = _anchors(e["bbox"])
            dx = []
            dy = []
            for o in elements:
                if e["id"] == o["id"]:
                    continue
                oa = _anchors(o["bbox"])
                dx.extend([abs(ea["left"] - oa["left"]), abs(ea["center"] - oa["center"]), abs(ea["right"] - oa["right"])])
                dy.extend([abs(ea["top"] - oa["top"]), abs(ea["middle"] - oa["middle"]), abs(ea["bottom"] - oa["bottom"])])
            align_bad = min(dx) + min(dy) if dx and dy else 0.0
            blame[str(e["id"])] += align_bad * 1.4

        for first_id, second_id in self._task_spec.get("reading_order", []):
            if first_id not in by_id or second_id not in by_id:
                continue
            a = by_id[first_id]["bbox"]
            b = by_id[second_id]["bbox"]
            if not (a[1] <= b[1] or (abs(a[1] - b[1]) <= 0.05 and a[0] <= b[0])):
                blame[str(first_id)] += 0.15
                blame[str(second_id)] += 0.15

        importance = [float(e["importance"]) for e in elements]
        salience = []
        for e in elements:
            x, y, w, h = e["bbox"]
            salience.append(0.55 * math.log(_area(e["bbox"]) + 1e-4) - 0.22 * y + 0.18 * (1.0 - abs((x + w / 2.0) - 0.5) / 0.5))

        imp_r = _rank(importance)
        sal_r = _rank(salience)
        for e, ri, rs in zip(elements, imp_r, sal_r):
            blame[str(e["id"])] += abs(ri - rs) / max(1.0, len(elements))

        max_blame = max(blame.values()) if blame else 1.0
        if max_blame <= EPS:
            return blame
        return {k: _clamp(v / max_blame, 0.0, 1.0) for k, v in blame.items()}

    def _constraint_warnings(self, elements: List[Dict[str, object]]) -> List[str]:
        warnings: List[str] = []
        for e in elements:
            x, y, w, h = [float(v) for v in e["bbox"]]
            min_w, min_h = [float(v) for v in e["min_size"]]
            max_w, max_h = [float(v) for v in e["max_size"]]
            if w - min_w < 0.02 or h - min_h < 0.02:
                warnings.append(f"{e['id']}:near_min_size")
            if max_w - w < 0.02 or max_h - h < 0.02:
                warnings.append(f"{e['id']}:near_max_size")
        return warnings[:6]

    def _suggested_edits(self, worst: List[str], focus: List[str]) -> List[str]:
        suggestions: List[str] = []
        for metric in worst:
            if metric == "alignment":
                suggestions.append("align related elements on x or y")
            elif metric == "spacing":
                suggestions.append("distribute a crowded group")
            elif metric == "hierarchy":
                suggestions.append("promote the focal element")
            elif metric == "intent_fit":
                suggestions.append("anchor an important element to its semantic region")
            elif metric == "reading_order":
                suggestions.append("reflow a story group or vertical stack")
            elif metric == "occupancy":
                suggestions.append("resize the hero or body block toward target fill")
            elif metric == "text_fit":
                suggestions.append("resize text blocks to improve copy fit")

        if focus:
            suggestions.append(f"inspect focus elements: {', '.join(focus[:2])}")

        out = []
        seen = set()
        for item in suggestions:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out[:5]

    def _neighborhood_utilities(self, base_elements: List[Dict[str, object]], focus_metrics: List[str]) -> List[float]:
        candidates: List[List[Dict[str, object]]] = []
        by_group: Dict[str, List[str]] = {}
        for e in base_elements:
            by_group.setdefault(str(e["group"]), []).append(str(e["id"]))

        if "alignment" in focus_metrics or "spacing" in focus_metrics:
            headline_ids = [e["id"] for e in base_elements if e["group"] in {"headline", "header", "stories"}]
            if len(headline_ids) >= 2:
                cand = _deepcopy_elements(base_elements)
                self._apply_action(cand, DesignGymAction(action_type="align", element_ids=headline_ids[:3], axis="x", mode="left"))
                candidates.append(cand)

        if "hierarchy" in focus_metrics or "occupancy" in focus_metrics:
            important = max(base_elements, key=lambda e: float(e["importance"]))
            cand = _deepcopy_elements(base_elements)
            self._apply_action(cand, DesignGymAction(action_type="promote", element_id=str(important["id"]), strength=0.05))
            candidates.append(cand)

        if "intent_fit" in focus_metrics or "reading_order" in focus_metrics:
            for element_id, region_id in self._task_spec.get("intent_regions", {}).items():
                if any(str(e["id"]) == element_id for e in base_elements):
                    cand = _deepcopy_elements(base_elements)
                    self._apply_action(cand, DesignGymAction(action_type="anchor_to_region", element_id=element_id, region_id=str(region_id), mode="center"))
                    candidates.append(cand)
                    break

        for group_id, ids in by_group.items():
            if len(ids) >= 3:
                cand = _deepcopy_elements(base_elements)
                self._apply_action(cand, DesignGymAction(action_type="reflow_group", group_id=group_id, pattern="stack"))
                candidates.append(cand)
                break

        utilities = []
        for cand in candidates[:4]:
            utilities.append(float(self._score_layout(cand)["utility"]))
        return utilities

    def _percentile_rank(self, utility: float, neighborhood: List[float]) -> float:
        if not neighborhood:
            return 0.5
        wins = sum(1 for value in neighborhood if utility >= value - 1e-9)
        return _clamp(wins / len(neighborhood), 0.0, 1.0)

    def _oscillation_penalty(self, action: DesignGymAction) -> float:
        history = self._state.action_history[-2:]
        if len(history) < 2:
            return 0.0
        prev = history[-1]
        current = action.action_type
        if current == "move" and '"action_type":"move"' in prev:
            return 0.01
        if current == "apply_template" and '"action_type":"apply_template"' in prev:
            return 0.02
        return 0.0