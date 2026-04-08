from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional, Sequence

from openai import OpenAI

try:
    from DesignGym import DesignGymAction, DesignGymEnv
except Exception:
    from models import DesignGymAction
    from client import DesignGymEnv


HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:scaleway")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
TASK_NAME = os.getenv("DESIGNGYM_TASK", "poster_basic_v1")
BENCHMARK = os.getenv("DESIGNGYM_BENCHMARK", "designgym")
MAX_STEPS = int(os.getenv("DESIGNGYM_MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "24"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.55"))


SYSTEM_PROMPT = """
You are choosing one action for a layout optimization environment.

Return exactly one minified JSON object:
{"choice": <integer>}

Rules:
- Choose exactly one candidate index.
- Do not explain.
- Do not output markdown.
- Prefer actions that improve the weakest metrics first.
- Avoid repeating low-gain actions.
- Do not choose finalize unless it is clearly appropriate late in the episode.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


_LAYOUT_ID_RE = re.compile(r"([A-Za-z0-9_]+)@\(")


def present_ids(obs) -> List[str]:
    return _LAYOUT_ID_RE.findall(obs.layout_summary or "")


def has_id(obs, element_id: str) -> bool:
    return f"{element_id}@(" in (obs.layout_summary or "")


def ids_in_obs(obs, ids: Sequence[str]) -> List[str]:
    return [x for x in ids if has_id(obs, x)]


def task_kind(task_id: str) -> str:
    if "editorial" in task_id:
        return "editorial"
    if "dense" in task_id:
        return "dense"
    return "poster"


def min_steps_for_task(task_id: str, max_steps: int) -> int:
    kind = task_kind(task_id)
    if kind == "poster":
        return min(max_steps, 5)
    if kind == "editorial":
        return min(max_steps, 6)
    return min(max_steps, 7)


def phase_for(step: int, obs) -> str:
    score = float(obs.current_score)
    min_steps = min_steps_for_task(obs.task_id, obs.max_steps)

    if step <= 2:
        return "build"
    if step < min_steps - 1:
        return "improve"
    if score < 0.78:
        return "repair"
    return "polish"


def should_allow_finalize(step: int, obs, recent_rewards: List[float]) -> bool:
    score = float(obs.current_score)
    min_steps = min_steps_for_task(obs.task_id, obs.max_steps)

    if step < min_steps:
        return False
    if step >= obs.max_steps:
        return True
    if score >= 0.84:
        return True

    tail = recent_rewards[-2:]
    if len(tail) == 2 and max(tail) <= 0.01 and score >= 0.74:
        return True

    return False


def make_resize(element_id: str, dw: float, dh: float, anchor: str = "center") -> DesignGymAction:
    return DesignGymAction(action_type="resize", element_id=element_id, dw=dw, dh=dh, anchor=anchor)


def make_move(element_id: str, dx: float, dy: float) -> DesignGymAction:
    return DesignGymAction(action_type="move", element_id=element_id, dx=dx, dy=dy)


def make_promote(element_id: str, strength: float = 0.04) -> DesignGymAction:
    return DesignGymAction(action_type="promote", element_id=element_id, strength=strength)


def make_align(ids: Sequence[str], axis: str, mode: str) -> DesignGymAction:
    return DesignGymAction(action_type="align", element_ids=list(ids), axis=axis, mode=mode)


def make_distribute(ids: Sequence[str], axis: str) -> DesignGymAction:
    return DesignGymAction(action_type="distribute", element_ids=list(ids), axis=axis)


def make_anchor(element_id: str, region_id: str, mode: str = "center") -> DesignGymAction:
    return DesignGymAction(
        action_type="anchor_to_region",
        element_id=element_id,
        region_id=region_id,
        mode=mode,
    )


def make_reflow(group_id: str, pattern: str) -> DesignGymAction:
    return DesignGymAction(action_type="reflow_group", group_id=group_id, pattern=pattern)


def heuristic_action(step: int, obs, recent_rewards: List[float], recent_actions: List[str]) -> DesignGymAction:
    worst = list(obs.worst_metrics or [])
    metrics = dict(obs.metrics or {})
    kind = task_kind(obs.task_id)
    phase = phase_for(step, obs)

    if "occupancy" in worst or metrics.get("occupancy", 1.0) < 0.62:
        if has_id(obs, "hero_image"):
            return make_resize("hero_image", 0.03, 0.02)
        if has_id(obs, "details"):
            return make_resize("details", 0.02, 0.02)

    if "hierarchy" in worst:
        for target in ["title", "headline_1", "masthead", "cta", "details"]:
            if has_id(obs, target):
                return make_promote(target, 0.04 if phase != "polish" else 0.03)

    if "alignment" in worst:
        ids = ids_in_obs(obs, ["title", "subtitle", "masthead", "headline_1", "headline_2", "headline_3"])
        if len(ids) >= 2:
            return make_align(ids[: min(3, len(ids))], "x", "left")
        ids = ids_in_obs(obs, ["caption_1", "caption_2"])
        if len(ids) >= 2:
            return make_align(ids, "y", "top")

    if "reading_order" in worst or "spacing" in worst:
        if kind == "poster":
            return make_reflow("headline", "stack")
        if kind == "editorial":
            return make_reflow("stories", "stack")
        return make_reflow("support", "row")

    if "intent_fit" in worst:
        if has_id(obs, "hero_image"):
            return make_anchor("hero_image", "hero_center")
        if has_id(obs, "cta"):
            return make_anchor("cta", "safe_lower_right")
        if has_id(obs, "masthead"):
            return make_anchor("masthead", "top_band")

    if "text_fit" in worst:
        for target in ["details", "subtitle", "headline_2", "headline_3"]:
            if has_id(obs, target):
                return make_resize(target, 0.02, 0.01)

    if phase in {"repair", "polish"}:
        if has_id(obs, "hero_image"):
            return make_resize("hero_image", 0.02, 0.01)
        if has_id(obs, "title") and has_id(obs, "subtitle"):
            return make_align(["title", "subtitle"], "x", "left")

    if should_allow_finalize(step, obs, recent_rewards):
        return DesignGymAction(action_type="finalize")

    for target in ["hero_image", "title", "subtitle", "details", "cta"]:
        if has_id(obs, target):
            return make_move(target, 0.01, -0.01)

    return DesignGymAction(action_type="finalize")


def score_candidate_locally(
    action: DesignGymAction,
    obs,
    step: int,
    recent_rewards: List[float],
    recent_actions: List[str],
) -> float:
    worst = set(obs.worst_metrics or [])
    metrics = dict(obs.metrics or {})
    phase = phase_for(step, obs)
    score = 0.0

    if action.action_type == "finalize":
        return 100.0 if should_allow_finalize(step, obs, recent_rewards) else -100.0

    action_str = action.canonical()

    if recent_actions and action_str == recent_actions[-1]:
        score -= 30.0
    if len(recent_actions) >= 2 and action_str == recent_actions[-2]:
        score -= 15.0
    if recent_rewards and recent_rewards[-1] <= 1e-6 and recent_actions and action_str == recent_actions[-1]:
        score -= 50.0

    if action.action_type == "resize":
        if action.element_id == "hero_image":
            score += 20.0
            if "occupancy" in worst:
                score += 18.0
            if "hierarchy" in worst:
                score += 10.0
            if phase == "build":
                score += 8.0
        if action.element_id == "details" and "text_fit" in worst:
            score += 14.0

    if action.action_type == "promote":
        if "hierarchy" in worst:
            score += 18.0
        if action.element_id in {"title", "headline_1", "masthead"}:
            score += 10.0

    if action.action_type == "align":
        if "alignment" in worst:
            score += 22.0
        if "reading_order" in worst:
            score += 8.0
        score += 4.0

    if action.action_type == "reflow_group":
        if "reading_order" in worst:
            score += 20.0
        if "spacing" in worst:
            score += 18.0
        if phase == "build":
            score += 5.0

    if action.action_type == "anchor_to_region":
        if "intent_fit" in worst:
            score += 20.0
        if action.element_id == "hero_image":
            score += 5.0

    if action.action_type == "distribute":
        if "spacing" in worst:
            score += 18.0
        if "alignment" in worst:
            score += 5.0

    if action.action_type == "move":
        score += 1.0
        if phase == "polish":
            score += 3.0

    if metrics.get("occupancy", 1.0) < 0.60 and action.action_type == "resize":
        score += 6.0

    return score


def candidate_actions(
    step: int,
    obs,
    recent_rewards: List[float],
    recent_actions: List[str],
) -> List[DesignGymAction]:
    worst = set(obs.worst_metrics or [])
    metrics = dict(obs.metrics or {})
    kind = task_kind(obs.task_id)
    phase = phase_for(step, obs)

    actions: List[DesignGymAction] = []

    if step == 1 and float(obs.current_score) < 0.60:
        if kind == "poster":
            actions.append(DesignGymAction(action_type="apply_template", template_id="hero"))
        elif kind == "editorial":
            actions.append(DesignGymAction(action_type="apply_template", template_id="editorial"))
        else:
            actions.append(DesignGymAction(action_type="apply_template", template_id="grid"))

    if has_id(obs, "hero_image"):
        actions.append(make_resize("hero_image", 0.03, 0.02))
        if phase in {"repair", "polish"}:
            actions.append(make_resize("hero_image", 0.02, 0.01))
        if "intent_fit" in worst:
            actions.append(make_anchor("hero_image", "hero_center"))

    if has_id(obs, "details") and ("text_fit" in worst or "occupancy" in worst):
        actions.append(make_resize("details", 0.02, 0.02))
        actions.append(make_resize("details", -0.02, 0.01))

    for target in ["title", "headline_1", "masthead", "cta"]:
        if has_id(obs, target):
            actions.append(make_promote(target, 0.04))
            break

    headline_ids = ids_in_obs(obs, ["title", "subtitle", "masthead", "headline_1", "headline_2", "headline_3"])
    if len(headline_ids) >= 2:
        actions.append(make_align(headline_ids[: min(3, len(headline_ids))], "x", "left"))
    if len(headline_ids) >= 3:
        actions.append(make_distribute(headline_ids[:3], "y"))

    if kind == "poster":
        actions.append(make_reflow("headline", "stack"))
    elif kind == "editorial":
        actions.append(make_reflow("stories", "stack"))
    else:
        actions.append(make_reflow("support", "row"))
        caption_ids = ids_in_obs(obs, ["caption_1", "caption_2"])
        if len(caption_ids) >= 2:
            actions.append(make_align(caption_ids, "y", "top"))

    if has_id(obs, "cta"):
        actions.append(make_anchor("cta", "safe_lower_right"))
    if has_id(obs, "masthead"):
        actions.append(make_anchor("masthead", "top_band"))
    if has_id(obs, "logo") and kind == "poster":
        actions.append(make_anchor("logo", "top_right"))

    if phase in {"repair", "polish"}:
        for target in ["hero_image", "title", "subtitle", "details", "cta"]:
            if has_id(obs, target):
                actions.append(make_move(target, 0.01, -0.01))
                break

    heur = heuristic_action(step, obs, recent_rewards, recent_actions)
    actions.append(heur)

    if should_allow_finalize(step, obs, recent_rewards):
        actions.append(DesignGymAction(action_type="finalize"))

    dedup: List[DesignGymAction] = []
    seen = set()
    for action in actions:
        action_str = action.canonical()
        if action_str not in seen:
            seen.add(action_str)
            dedup.append(action)

    filtered: List[DesignGymAction] = []
    for action in dedup:
        action_str = action.canonical()
        if recent_actions and action_str == recent_actions[-1] and recent_rewards and recent_rewards[-1] <= 1e-6:
            continue
        if len(recent_actions) >= 2 and action_str == recent_actions[-2]:
            continue
        filtered.append(action)

    if not filtered:
        filtered = dedup[:]

    ranked = sorted(
        filtered,
        key=lambda a: score_candidate_locally(a, obs, step, recent_rewards, recent_actions),
        reverse=True,
    )

    return ranked[:6]


def build_choice_prompt(
    step: int,
    obs,
    history: List[str],
    recent_rewards: List[float],
    candidates: List[DesignGymAction],
) -> str:
    recent = "\n".join(history[-4:]) if history else "None"
    allow_finalize = should_allow_finalize(step, obs, recent_rewards)
    phase = phase_for(step, obs)

    candidate_lines = []
    for idx, action in enumerate(candidates):
        marker = "allowed" if action.action_type != "finalize" or allow_finalize else "blocked"
        candidate_lines.append(f"{idx}: {action.canonical()} [{marker}]")

    return textwrap.dedent(
        f"""
        Task: {obs.task_id}
        Step: {step}
        Max steps: {obs.max_steps}
        Phase: {phase}
        Current score: {obs.current_score:.4f}
        Best score so far: {obs.best_score_so_far:.4f}
        Worst metrics: {json.dumps(obs.worst_metrics)}
        Metrics: {json.dumps(obs.metrics, sort_keys=True)}
        Metric deltas: {json.dumps(obs.metric_deltas, sort_keys=True)}
        Focus elements: {json.dumps(obs.focus_elements)}
        Suggested edits: {json.dumps(obs.suggested_edits)}
        Layout summary: {obs.layout_summary}
        Recent rewards: {json.dumps([round(x, 4) for x in recent_rewards[-4:]])}
        Finalize allowed: {str(allow_finalize).lower()}

        Previous actions:
        {recent}

        Candidate actions:
        {chr(10).join(candidate_lines)}

        Return exactly one JSON object:
        {{"choice": N}}
        """
    ).strip()


def get_model_action_sync(
    client: Optional[OpenAI],
    step: int,
    obs,
    history: List[str],
    recent_rewards: List[float],
    recent_actions: List[str],
) -> DesignGymAction:
    candidates = candidate_actions(step, obs, recent_rewards, recent_actions)
    best_local = candidates[0]

    if client is None:
        return best_local

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_choice_prompt(step, obs, history, recent_rewards, candidates)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()
        payload = json.loads(text)
        choice = int(payload["choice"])

        if choice < 0 or choice >= len(candidates):
            return best_local

        selected = candidates[choice]

        if selected.action_type == "finalize" and not should_allow_finalize(step, obs, recent_rewards):
            return best_local

        return selected

    except Exception:
        return best_local


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    env = await DesignGymEnv.from_docker_image(LOCAL_IMAGE_NAME) if LOCAL_IMAGE_NAME else DesignGymEnv(base_url=BASE_URL)

    rewards: List[float] = []
    history: List[str] = []
    recent_actions: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env:
            result = await env.reset(task_id=TASK_NAME, seed=0)
            obs = result.observation

            for step in range(1, min(MAX_STEPS, obs.max_steps) + 1):
                if result.done or obs.done:
                    break

                action = await asyncio.to_thread(
                    get_model_action_sync,
                    client,
                    step,
                    obs,
                    history,
                    rewards,
                    recent_actions,
                )
                action_str = action.canonical()

                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)

                rewards.append(reward)
                recent_actions.append(action_str)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=bool(result.done),
                    error=obs.last_action_error,
                )

                history.append(
                    f"step={step} action={action_str} reward={reward:.4f} "
                    f"score={obs.current_score:.4f} worst={','.join(obs.worst_metrics)}"
                )

                if result.done or obs.done:
                    break

            state = await env.state()
            score = max(0.0, min(1.0, float(state.current_score)))
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())