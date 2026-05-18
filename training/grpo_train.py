#!/usr/bin/env python3
"""
DesignGym GRPO Training Script
==============================
Trains a Qwen 0.5B model with GRPO on the DesignGym environment,
starting from an SFT-warm-started adapter.

Produces a 3-way comparison: base → SFT → GRPO.

Usage (smoke test):
    python training/grpo_train.py --smoke

Usage (full training):
    python training/grpo_train.py \
        --num_train_states 400 --max_steps 150 --eval_seeds 3

Run from the repo root:  cd /workspace/DesignGym && python training/grpo_train.py
"""

from __future__ import annotations

# ── stdlib (safe, no side-effects) ──────────────────────────────────────────
import argparse
import copy
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ── repo root on sys.path BEFORE any local imports ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── unsloth BEFORE transformers / peft (required for patching) ─────────────
try:
    from unsloth import FastLanguageModel
except Exception as exc:
    sys.exit(f"[FATAL] Unsloth is required.\n{exc}")

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, login
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer
# llm_blender (pulled by trl 0.24 judges) uses a removed transformers constant
import transformers.utils.hub as _thub
if not hasattr(_thub, "TRANSFORMERS_CACHE"):
    from huggingface_hub.constants import HF_HUB_CACHE
    _thub.TRANSFORMERS_CACHE = HF_HUB_CACHE

from trl import GRPOConfig, GRPOTrainer

# ── local (DesignGym) ──────────────────────────────────────────────────────
try:
    from models import DesignGymAction
    from server.DesignGym_environment import DesignGymEnvironment
except ImportError:
    from DesignGym.models import DesignGymAction
    from DesignGym.server.DesignGym_environment import DesignGymEnvironment

try:
    from training.generate_sft_data import (
        SYSTEM_PROMPT, compact_action, choose_expert_action,
        prompt_from_obs, preferred_action_type_for_example,
        candidate_actions, ids_in_layout,
    )
except ImportError:
    from generate_sft_data import (
        SYSTEM_PROMPT, compact_action, choose_expert_action,
        prompt_from_obs, preferred_action_type_for_example,
        candidate_actions, ids_in_layout,
    )

# ── constants ──────────────────────────────────────────────────────────────
TASKS = ["poster_basic_v1", "editorial_cover_v1", "dense_flyer_v1"]
ALLOWED_ACTION_TYPES = {
    "apply_template", "anchor_to_region", "resize", "move",
    "align", "distribute", "promote", "reflow_group", "finalize",
}
EVAL_METRICS = [
    "final_score", "instruction_score", "total_reward",
    "valid_json_rate", "valid_action_type_rate", "env_rejection_rate",
    "early_finalize_count", "invalid_actions",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def extract_json_object(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text), text
    except Exception:
        pass
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None, text
    candidate = match.group(0)
    try:
        return json.loads(candidate), candidate
    except Exception:
        return None, text


def parse_action_from_text(text: str):
    parsed, extracted = extract_json_object(text)
    if parsed is None:
        return None, {"valid_json": False, "valid_action_type": False,
                       "pydantic_valid": False, "extracted": extracted}
    valid_at = parsed.get("action_type") in ALLOWED_ACTION_TYPES
    if not valid_at:
        return None, {"valid_json": True, "valid_action_type": False,
                       "pydantic_valid": False, "extracted": extracted}
    try:
        action = DesignGymAction(**parsed)
        return action, {"valid_json": True, "valid_action_type": True,
                         "pydantic_valid": True, "extracted": extracted}
    except Exception:
        return None, {"valid_json": True, "valid_action_type": True,
                       "pydantic_valid": False, "extracted": extracted}


def make_prompt(obs) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_from_obs(obs)},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 2. GRPO DATASET  (collect env states for on-policy training)
# ═══════════════════════════════════════════════════════════════════════════

# Element ids used for "awkward" perturbations.  Wide list so at least
# one will exist in every task.
_AWKWARD_ELEMENT_IDS = [
    "title", "subtitle", "headline_1", "headline_2", "masthead",
    "hero_image", "image_left", "image_right", "details", "cta",
    "price_badge", "logo", "caption_1", "caption_2",
]


def _pick_awkward_perturbation(obs: Any, rng: random.Random):
    """Tiny random move/resize to dirty the layout slightly."""
    present = set(ids_in_layout(obs))
    candidates = [eid for eid in _AWKWARD_ELEMENT_IDS if eid in present]
    if not candidates:
        return None
    eid = rng.choice(candidates)
    if rng.random() < 0.5:
        dx = rng.choice([-0.04, -0.03, 0.03, 0.04])
        dy = rng.choice([-0.04, -0.03, 0.03, 0.04])
        return DesignGymAction(action_type="move", element_id=eid, dx=dx, dy=dy)
    dw = rng.choice([-0.04, -0.03, 0.03, 0.04])
    dh = rng.choice([-0.03, 0.03])
    return DesignGymAction(action_type="resize", element_id=eid,
                            dw=dw, dh=dh, anchor="center")


def build_grpo_dataset(num_states: int,
                       seed: int = 0,
                       bad_state_ratio: float = 0.0) -> Dataset:
    """Build a GRPO dataset of env states.

    Args:
        num_states: target number of rows.
        seed: episode-seed offset.
        bad_state_ratio: fraction of rows captured AFTER a non-expert
            perturbation.  Of that fraction, ~2/3 come from sub-optimal
            valid candidates (still legal moves the env accepts but the
            expert wouldn't pick), and ~1/3 come from "awkward" tiny
            random moves/resizes.  Helps GRPO see contrast.
    """
    log(f"Building GRPO dataset ({num_states} states, bad_state_ratio={bad_state_ratio:.2f})…")
    rng = random.Random(seed * 7 + 13)
    rows: List[Dict[str, Any]] = []
    episode_idx = 0
    counts = {"expert": 0, "bad_random": 0, "bad_awkward": 0}

    while len(rows) < num_states:
        task_id = TASKS[episode_idx % len(TASKS)]
        episode_seed = seed + episode_idx
        env = DesignGymEnvironment()
        obs = env.reset(task_id=task_id, seed=episode_seed)
        prefix_actions: List[str] = []

        for local_step in range(int(obs.max_steps)):
            if len(rows) >= num_states or env.state.done:
                break

            roll = rng.random()
            kind = "expert"
            if roll < bad_state_ratio:
                # Of the "bad" budget, ~67% random-valid, ~33% awkward.
                if rng.random() < 0.67:
                    kind = "bad_random"
                else:
                    kind = "bad_awkward"

            if kind != "expert":
                perturb = None
                if kind == "bad_random":
                    try:
                        cands = candidate_actions(obs, list(prefix_actions))
                        # Drop "finalize" so we don't end the episode early.
                        cands = [a for a in cands if a.action_type != "finalize"] or cands
                        if cands:
                            perturb = rng.choice(cands)
                    except Exception:
                        perturb = None
                if perturb is None:  # awkward fallback or explicitly chosen
                    perturb = _pick_awkward_perturbation(obs, rng)

                if perturb is not None:
                    try:
                        obs2 = env.step(perturb)
                        if not getattr(obs2, "last_action_error", None) and not env.state.done:
                            prefix_actions.append(compact_action(perturb))
                            obs = obs2
                        else:
                            kind = "expert"
                    except Exception:
                        kind = "expert"
                else:
                    kind = "expert"

            rows.append({
                "prompt":         make_prompt(obs),
                "task_id":        task_id,
                "seed":           episode_seed,
                "prefix_actions": json.dumps(prefix_actions),
                "step_index":     local_step,
                "state_kind":     kind,
            })
            counts[kind] = counts.get(kind, 0) + 1

            pref = preferred_action_type_for_example(episode_idx, local_step, obs)
            expert = choose_expert_action(env, obs, preferred_action_type=pref)
            obs = env.step(expert)
            prefix_actions.append(compact_action(expert))

        episode_idx += 1

    random.Random(seed).shuffle(rows)
    ds = Dataset.from_list(rows)
    log(f"Dataset ready: {len(ds)} rows from {episode_idx} episodes  "
        f"(expert={counts['expert']}  bad_random={counts['bad_random']}  "
        f"bad_awkward={counts['bad_awkward']})")
    return ds


# ═══════════════════════════════════════════════════════════════════════════
# 3. REWARD FUNCTION  (called by GRPOTrainer during rollouts)
# ═══════════════════════════════════════════════════════════════════════════

# Reward debug capture (populated during training, dumped at end).
REWARD_DEBUG_ROWS: List[Dict[str, Any]] = []
MAX_REWARD_DEBUG_ROWS: int = 240
REWARD_VERSION: str = "v3_anchored"

# Worst-metric → action-type compatibility (process bonus map).
# v3: `promote` was over-rewarded in v2 (caused 81% mode-collapse). It's
# removed from `balance`, kept only as a weak alternative in `hierarchy` /
# `salience`.  Bonus cap is also halved (0.05 → 0.025) so this acts as a
# tie-breaker between roughly-equally-good actions, not a main signal.
PROCESS_BONUS_MAP: Dict[str, Dict[str, float]] = {
    # overlap / crowding family
    "overlap":            {"move": 1.0, "resize": 0.8, "reflow_group": 0.9, "distribute": 0.6},
    "crowding":           {"move": 1.0, "resize": 0.8, "reflow_group": 0.9, "distribute": 0.6},
    "occupancy":          {"resize": 1.0, "move": 0.6, "reflow_group": 0.7},
    "text_fit":           {"resize": 1.0, "reflow_group": 0.6},
    # alignment / spacing family
    "alignment":          {"align": 1.0, "distribute": 0.7, "reflow_group": 0.6, "snap": 0.7},
    "spacing":            {"distribute": 1.0, "align": 0.6, "reflow_group": 0.8},
    "negative_space":     {"distribute": 1.0, "reflow_group": 0.7, "resize": 0.5},
    "rhythm":             {"distribute": 1.0, "align": 0.6, "reflow_group": 0.6},
    "reading_order":      {"reflow_group": 1.0, "anchor_to_region": 0.7, "align": 0.6},
    # hierarchy / balance family — promote demoted, no longer a free win
    "hierarchy":          {"resize": 0.8, "apply_template": 0.7, "promote": 0.4},
    "balance":            {"move": 0.8, "resize": 0.7, "reflow_group": 0.6},
    "salience":           {"resize": 0.7, "apply_template": 0.6, "promote": 0.5},
    # intent / semantic placement family
    "intent":             {"anchor_to_region": 1.0, "apply_template": 0.7},
    "instruction":        {"anchor_to_region": 1.0, "apply_template": 0.7},
    "semantic_placement": {"anchor_to_region": 1.0, "apply_template": 0.7},
}

# Action types that meaningfully move `instruction_score` (they target the
# brief's required regions or global structure).  Used by the instruction
# floor bonus in v3.
INSTRUCTION_LEVER_ACTIONS = {"anchor_to_region", "apply_template"}


def _safe_float(obj, attr: str, default: float = 0.0) -> float:
    try:
        v = getattr(obj, attr, default)
        return float(v if v is not None else default)
    except Exception:
        return default


def _scaled_delta(d: float, scale: float) -> float:
    """Tanh-scaled signed delta in approx [-1, 1].

    With scale=0.025, a typical step gain of ~0.015 maps to tanh(0.6) ≈ 0.54
    of the channel range — gives signal without making tiny gains 'free'."""
    if scale <= 0:
        return 0.0
    return math.tanh(float(d) / float(scale))


def _process_bonus(action_type: str, worst_metrics, phase: str) -> float:
    """Small bonus when the action type matches what the weak metrics need.

    v3: capped at +0.025 (was +0.05 in v2) so this no longer dominates the
    score-gain channel.  Structure-phase nudge for `apply_template` is also
    softened (+0.025 instead of +0.05)."""
    if not worst_metrics:
        bonus = 0.0
    else:
        metrics = [str(m).lower() for m in worst_metrics]
        best = 0.0
        for m in metrics:
            for key, table in PROCESS_BONUS_MAP.items():
                if key in m:
                    best = max(best, float(table.get(action_type, 0.0)))
        bonus = 0.025 * best  # capped at +0.025
    # Structure-phase nudge: prefer global action types early.
    if phase == "structure":
        if action_type == "apply_template":
            bonus += 0.025
        elif action_type in {"move", "resize"}:
            bonus -= 0.015
    return bonus


def _instruction_floor_bonus(action_type: str,
                              before_inst: float,
                              inst_gain: float) -> float:
    """Reward `anchor_to_region` / `apply_template` when instruction_score
    is still low — these are the only actions that move it up reliably and
    were abandoned during the v2 mode-collapse."""
    if (before_inst < 0.70
            and inst_gain > 1e-3
            and action_type in INSTRUCTION_LEVER_ACTIONS):
        return 0.10
    return 0.0


def _action_repetition_penalty(action_type: str, prefix_json: str) -> float:
    """−0.04 if `action_type` was already used ≥ 3 times in the rollout
    prefix.  Direct counter to the v2 'spam promote' attractor."""
    if not prefix_json:
        return 0.0
    try:
        prior = json.loads(prefix_json)
    except Exception:
        return 0.0
    used = 0
    for aj in prior:
        try:
            payload = json.loads(aj) if isinstance(aj, str) else aj
            if str(payload.get("action_type", "")) == action_type:
                used += 1
        except Exception:
            continue
    return -0.04 if used >= 3 else 0.0


def reconstruct_env(task_id: str, seed: int, prefix_json: str) -> DesignGymEnvironment:
    env = DesignGymEnvironment()
    env.reset(task_id=task_id, seed=int(seed))
    for aj in json.loads(prefix_json or "[]"):
        payload = json.loads(aj) if isinstance(aj, str) else aj
        env.step(DesignGymAction(**payload))
        if env.state.done:
            break
    return env


def _push_debug(row: Dict[str, Any]) -> None:
    if len(REWARD_DEBUG_ROWS) < MAX_REWARD_DEBUG_ROWS:
        REWARD_DEBUG_ROWS.append(row)


def designgym_reward(completions, prompts=None, completion_ids=None,
                     task_id=None, seed=None, prefix_actions=None,
                     step_index=None, **kw):
    """v3_anchored reward (default since 2026-04-26).

    Designed to fix v2's mode-collapse onto `promote` while preserving
    SFT's anchor_to_region behavior. Components after passing format
    gates and not env-rejected:

        baseline                    = +0.03   (validity baseline only)
        score_gain  scaled (0.025)  = +/- 0.30
        instr_gain  scaled (0.020)  = +/- 0.45  (DOMINANT — what we eval on)
        phase_gain  scaled (0.025)  = +/- 0.10
        best_gain   scaled (0.015)  = +/- 0.05
        env_reward  clip            = +/- 0.05
        process_bonus               = ~+0.025  (capped, tie-breaker only)
        instruction_floor_bonus     = +0.10 if anchor_to_region/apply_template
                                       lifts inst from <0.70
        action_repetition_penalty   = -0.04 if same type used >=3 times in prefix
        useless_pen                 = -0.05 (when all gains are ~0)
        early_finalize_pen          = -0.30 (unless score and instruction high)

    Then clipped to [-1, 1].  Format failures bypass and short-circuit:

        invalid_json      -> -1.00
        invalid_action    -> -0.75
        pydantic_invalid  -> -0.55
        env_rejected      -> -0.40
    """
    rewards: List[float] = []
    n = len(completions)
    si = step_index if step_index is not None else [None] * n

    for completion, t, s, p, st_idx in zip(completions, task_id, seed,
                                            prefix_actions, si):
        text = completion_to_text(completion)
        action, meta = parse_action_from_text(text)

        if not meta["valid_json"]:
            r = -1.0
            _push_debug({"task_id": str(t), "seed": int(s), "step": st_idx,
                         "action_type": None, "reward": r, "reason": "invalid_json",
                         "raw_preview": (text or "")[:120]})
            rewards.append(r); continue
        if not meta["valid_action_type"]:
            r = -0.75
            _push_debug({"task_id": str(t), "seed": int(s), "step": st_idx,
                         "action_type": None, "reward": r, "reason": "invalid_action_type",
                         "raw_preview": (text or "")[:120]})
            rewards.append(r); continue
        if not meta["pydantic_valid"] or action is None:
            r = -0.55
            _push_debug({"task_id": str(t), "seed": int(s), "step": st_idx,
                         "action_type": None, "reward": r, "reason": "pydantic_invalid",
                         "raw_preview": (text or "")[:120]})
            rewards.append(r); continue

        try:
            env = reconstruct_env(str(t), int(s), str(p))
            before_score = _safe_float(env.state, "current_score")
            before_inst  = _safe_float(env.state, "instruction_score")
            before_phase = _safe_float(env.state, "phase_score")
            before_best  = _safe_float(env.state, "best_score_so_far")
            phase = str(getattr(env.state, "phase", "") or "")
            worst_metrics = list(getattr(env.state, "worst_metrics", []) or [])

            obs = env.step(action)

            if getattr(obs, "last_action_error", None):
                r = -0.40
                _push_debug({"task_id": str(t), "seed": int(s), "step": st_idx,
                             "action_type": action.action_type, "reward": r,
                             "reason": "env_rejected",
                             "before_score": before_score, "after_score": before_score,
                             "phase": phase, "weak_metrics": worst_metrics})
                rewards.append(r); continue

            after_score = _safe_float(env.state, "current_score")
            after_inst  = _safe_float(env.state, "instruction_score")
            after_phase = _safe_float(env.state, "phase_score")
            after_best  = _safe_float(env.state, "best_score_so_far")
            env_r       = _safe_float(env.state, "last_reward")

            score_gain = after_score - before_score
            inst_gain  = after_inst  - before_inst
            phase_gain = after_phase - before_phase
            best_gain  = after_best  - before_best

            # v3: score_gain scale bumped 0.015 -> 0.025 so tiny +0.005 gains
            # are no longer worth as much (kills the v2 promote arbitrage).
            sd_score = _scaled_delta(score_gain, 0.025)
            sd_inst  = _scaled_delta(inst_gain,  0.020)
            sd_phase = _scaled_delta(phase_gain, 0.025)
            sd_best  = _scaled_delta(best_gain,  0.015)
            env_r_c  = max(-1.0, min(1.0, env_r))

            process_bonus = _process_bonus(action.action_type, worst_metrics, phase)
            inst_floor    = _instruction_floor_bonus(action.action_type,
                                                     before_inst, inst_gain)
            rep_penalty   = _action_repetition_penalty(action.action_type, str(p))

            # v3 weights: instruction_gain dominates (matches what we eval on);
            # score_gain still strong; phase/best as supporting signals.
            r = (
                0.03
                + 0.30 * sd_score
                + 0.45 * sd_inst
                + 0.10 * sd_phase
                + 0.05 * sd_best
                + 0.05 * env_r_c
                + process_bonus
                + inst_floor
                + rep_penalty
            )

            # Penalize legal but useless actions (all gains ~ 0 and env_r ~ 0).
            if (abs(score_gain) < 1e-3 and abs(inst_gain) < 1e-3
                    and abs(phase_gain) < 1e-3 and abs(env_r) < 1e-3):
                r -= 0.05

            # Early finalize penalty unless layout is already strong.
            if action.action_type == "finalize":
                step_count = int(getattr(env.state, "step_count", 0) or 0)
                max_steps  = int(getattr(env.state, "max_steps", 1) or 1)
                if step_count < int(0.70 * max_steps):
                    if not (after_score >= 0.78 and after_inst >= 0.70):
                        r -= 0.30

            r = max(-1.0, min(1.0, float(r)))

            _push_debug({
                "task_id": str(t), "seed": int(s), "step": st_idx,
                "action_type": action.action_type, "reward": r,
                "reason": "ok",
                "before_score": before_score, "after_score": after_score,
                "score_gain": score_gain,
                "before_instruction": before_inst, "after_instruction": after_inst,
                "instruction_gain": inst_gain,
                "phase_gain": phase_gain, "best_gain": best_gain,
                "env_reward": env_r, "process_bonus": process_bonus,
                "instruction_floor_bonus": inst_floor,
                "action_repetition_penalty": rep_penalty,
                "phase": phase, "weak_metrics": worst_metrics,
            })
            rewards.append(r)
        except Exception as exc:  # safety net — never crash GRPO rollouts
            r = -0.55
            _push_debug({"task_id": str(t), "seed": int(s), "step": st_idx,
                         "action_type": getattr(action, "action_type", None),
                         "reward": r, "reason": f"exception:{type(exc).__name__}"})
            rewards.append(r)

    return rewards


def write_reward_debug(out_dir: Path, n_sample: int = 40) -> Optional[Dict[str, Any]]:
    """Dump reward debug stats + sample rows to smoke_reward_debug.json."""
    if not REWARD_DEBUG_ROWS:
        return None
    rs = [float(x.get("reward", 0.0)) for x in REWARD_DEBUG_ROWS]
    action_counts = Counter(str(r.get("action_type")) for r in REWARD_DEBUG_ROWS)
    reason_counts = Counter(str(r.get("reason")) for r in REWARD_DEBUG_ROWS)
    sample = REWARD_DEBUG_ROWS[: max(1, int(n_sample))]
    payload: Dict[str, Any] = {
        "reward_version": REWARD_VERSION,
        "n_samples_collected": len(REWARD_DEBUG_ROWS),
        "reward_mean":  float(sum(rs) / len(rs)),
        "reward_std":   float((sum((x - sum(rs)/len(rs))**2 for x in rs) / len(rs)) ** 0.5),
        "reward_min":   float(min(rs)),
        "reward_max":   float(max(rs)),
        "action_type_counts": dict(action_counts),
        "reason_counts": dict(reason_counts),
        "rows": sample,
    }
    out = Path(out_dir) / "smoke_reward_debug.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    log(f"Reward debug written to {out}")
    log(f"  mean={payload['reward_mean']:.4f}  std={payload['reward_std']:.4f}  "
        f"min={payload['reward_min']:.3f}  max={payload['reward_max']:.3f}")
    log(f"  action_type_counts={payload['action_type_counts']}")
    log(f"  reason_counts={payload['reason_counts']}")
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION  (full env rollouts, matching base_vs_sft format)
# ═══════════════════════════════════════════════════════════════════════════

def generate_action(model, tokenizer, obs, max_seq: int = 4096):
    messages = make_prompt(obs)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, temperature=0.0,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    action, meta = parse_action_from_text(text)
    return action, text, meta


def _score_candidate_action(env_snapshot, action) -> float:
    """One-step look-ahead reward used to rank best-of-N candidates.

    Returns a real-valued score; higher = better. Uses a deepcopy of the env
    so the real rollout is untouched. Heavily penalises None / invalid actions.
    """
    if action is None:
        return -100.0
    try:
        env_copy = copy.deepcopy(env_snapshot)
        env_copy.step(action)
        st = env_copy.state
        if getattr(st, "last_action_error", None):
            return -50.0 + 0.05 * float(st.current_score or 0)
        # Positive linear combo: env reward + small bonuses on absolute scores.
        return (
            float(st.last_reward or 0.0)
            + 0.30 * float(st.instruction_score or 0.0)
            + 0.20 * float(st.current_score or 0.0)
            + 0.10 * float(st.phase_score or 0.0)
        )
    except Exception:
        return -100.0


def generate_action_bestofN(
    model, tokenizer, env, obs, n: int, max_seq: int = 4096,
    temperature: float = 0.9, top_p: float = 0.95,
):
    """Sample N completions, parse each, pick the one with the best look-ahead.

    Falls back to a per-sample loop if `num_return_sequences` is not supported
    by the underlying generate (some Unsloth builds disable it for 4-bit).
    Returns (action, text, meta) for the winning candidate, plus diagnostics
    via meta["best_of_n"].
    """
    if n <= 1:
        action, text, meta = generate_action(model, tokenizer, obs, max_seq)
        meta = dict(meta)
        meta["best_of_n"] = {"n": 1, "picked": 0, "scores": []}
        return action, text, meta

    messages = make_prompt(obs)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq).to("cuda")
    prompt_len = inputs["input_ids"].shape[-1]

    texts: List[str] = []
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
            )
        for i in range(out.shape[0]):
            gen_ids = out[i][prompt_len:]
            texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    except Exception as exc:
        log(f"[BoN] num_return_sequences failed ({exc!r}); falling back to loop")
        for _ in range(n):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = out[0][prompt_len:]
            texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    candidates = []
    for txt in texts:
        a, m = parse_action_from_text(txt)
        candidates.append((txt, a, m))

    scores = [_score_candidate_action(env, a) for (_t, a, _m) in candidates]

    valid_idxs = [i for i, (_, a, m) in enumerate(candidates)
                  if a is not None and bool(m.get("valid_json"))
                  and bool(m.get("valid_action_type"))]
    if valid_idxs:
        best_i = max(valid_idxs, key=lambda i: scores[i])
    else:
        best_i = max(range(len(candidates)), key=lambda i: scores[i])

    text, action, meta = candidates[best_i]
    meta = dict(meta)
    meta["best_of_n"] = {
        "n": n,
        "picked": int(best_i),
        "scores": [float(s) for s in scores],
        "n_valid": int(len(valid_idxs)),
    }
    return action, text, meta


def run_rollout(model, tokenizer, task_id: str, seed: int, max_seq: int = 4096,
                best_of: int = 1, bon_temperature: float = 0.9, bon_top_p: float = 0.95):
    env = DesignGymEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    records, vj, va, rej, ef = [], 0, 0, 0, 0
    max_steps = int(obs.max_steps)

    for step_idx in range(max_steps):
        if env.state.done:
            break
        if best_of > 1:
            action, raw, meta = generate_action_bestofN(
                model, tokenizer, env, obs, best_of, max_seq,
                temperature=bon_temperature, top_p=bon_top_p,
            )
        else:
            action, raw, meta = generate_action(model, tokenizer, obs, max_seq)
        vj += int(meta["valid_json"])
        va += int(meta["valid_action_type"] and meta["pydantic_valid"])
        if action is None:
            action = DesignGymAction(action_type="finalize")
        if action.action_type == "finalize" and step_idx < int(0.70 * max_steps):
            ef += 1
        obs = env.step(action)
        env_rej = bool(obs.last_action_error)
        rej += int(env_rej)
        records.append({
            "task_id": task_id, "seed": seed, "step": step_idx + 1,
            "phase": obs.phase, "action_type": action.action_type,
            "raw_generation": raw, "valid_json": bool(meta["valid_json"]),
            "valid_action_type": bool(meta["valid_action_type"]),
            "pydantic_valid": bool(meta["pydantic_valid"]),
            "env_rejected": env_rej, "reward": float(env.state.last_reward),
            "current_score": float(env.state.current_score),
            "instruction_score": float(env.state.instruction_score),
            "phase_score": float(env.state.phase_score),
            "done": bool(env.state.done),
        })
        if env.state.done:
            break

    n = max(1, len(records))
    return {
        "task_id": task_id, "seed": seed, "steps": len(records),
        "final_score": float(env.state.current_score),
        "instruction_score": float(env.state.instruction_score),
        "total_reward": float(env.state.total_reward),
        "valid_json_rate": vj / n,
        "valid_action_type_rate": va / n,
        "env_rejection_rate": rej / n,
        "early_finalize_count": ef,
        "invalid_actions": int(env.state.invalid_actions),
        "no_progress_steps": int(env.state.no_progress_steps),
    }, records


def evaluate_policy(model, tokenizer, name: str, seeds, max_seq: int, out: Path,
                    best_of: int = 1, bon_temperature: float = 0.9,
                    bon_top_p: float = 0.95):
    seed_list = list(seeds)
    bon_tag = f" best_of={best_of}" if best_of > 1 else ""
    log(f"Evaluating policy '{name}' ({len(TASKS)} tasks × {len(seed_list)} seeds){bon_tag}…")
    model.eval()
    summaries, all_steps = [], []
    for task_id in TASKS:
        for s in tqdm(seed_list, desc=f"{name}:{task_id}", leave=False):
            summary, steps = run_rollout(
                model, tokenizer, task_id, int(s), max_seq,
                best_of=best_of, bon_temperature=bon_temperature, bon_top_p=bon_top_p,
            )
            summary["policy"] = name
            summary["best_of"] = int(best_of)
            for r in steps:
                r["policy"] = name
                r["best_of"] = int(best_of)
            summaries.append(summary)
            all_steps.extend(steps)
    df = pd.DataFrame(summaries)
    df.to_csv(out / f"{name}_rollout_summary.csv", index=False)
    pd.DataFrame(all_steps).to_csv(out / f"{name}_rollout_steps.csv", index=False)
    log(f"  {name} mean final_score={df['final_score'].mean():.4f}  "
        f"instruction={df['instruction_score'].mean():.4f}  "
        f"valid_json={df['valid_json_rate'].mean():.2%}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMPARISON TABLES & PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison(dfs: Dict[str, pd.DataFrame], out: Path):
    combined = pd.concat(list(dfs.values()), ignore_index=True)
    combined.to_csv(out / "all_rollout_summary.csv", index=False)

    table = combined.groupby("policy")[EVAL_METRICS].mean().reset_index()
    table.to_csv(out / "comparison_summary_table.csv", index=False)
    table.to_json(out / "comparison_summary_table.json", orient="records", indent=2)

    by_task = combined.groupby(["policy", "task_id"])[EVAL_METRICS].mean().reset_index()
    by_task.to_csv(out / "comparison_by_task.csv", index=False)

    log("=" * 60)
    log("COMPARISON TABLE")
    log("=" * 60)
    print(table.to_string(index=False))
    log("=" * 60)

    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    policy_order = list(dfs.keys())
    palette = ["#999999", "#4a90d9", "#3578c6", "#27ae60", "#1e8449",
               "#c0392b", "#8e44ad"]

    def _color_for(p: str, idx: int) -> str:
        if p.startswith("base"):
            return palette[0]
        if p.startswith("sft"):
            return palette[2] if "@" in p else palette[1]
        if p.startswith("grpo"):
            return palette[4] if "@" in p else palette[3]
        return palette[idx % len(palette)]

    colors = [_color_for(p, i) for i, p in enumerate(policy_order)]
    for metric in EVAL_METRICS:
        plt.figure(figsize=(max(7, 1.4 * len(policy_order)), 4))
        vals = [table.loc[table["policy"] == p, metric].values[0] for p in policy_order]
        plt.bar(policy_order, vals, color=colors)
        plt.title(f"Policy comparison: {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=20, ha="right")
        if metric not in ("env_rejection_rate", "early_finalize_count", "invalid_actions"):
            plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(plots_dir / f"comparison_{metric}.png", dpi=160)
        plt.close()

    return table


# ═══════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOGS
# ═══════════════════════════════════════════════════════════════════════════

def save_training_logs(log_history: list, out: Path):
    with open(out / "grpo_trainer_log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)
    pd.DataFrame(log_history).to_csv(out / "grpo_trainer_log_history.csv", index=False)

    train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]
    if train_logs:
        plt.figure(figsize=(8, 5))
        plt.plot([x["step"] for x in train_logs], [x["loss"] for x in train_logs])
        plt.title("GRPO Training Loss")
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(out / "grpo_loss_curve.png", dpi=160)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DesignGym GRPO Training")
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke test (v2 schedule: 240 states, 120 steps, "
                        "8 gens, lr 3e-5, eval_seeds 5)")

    g = p.add_argument_group("Models")
    g.add_argument("--base_model",  default="unsloth/Qwen2.5-0.5B-Instruct")
    g.add_argument("--sft_adapter", default="yashvyasop/designgym2-sft-qwen05-lora")
    g.add_argument("--output_repo", default="yashvyasop/designgym2-grpo-qwen05-lora")

    g = p.add_argument_group("Training")
    g.add_argument("--num_train_states", type=int, default=400)
    g.add_argument("--max_steps",        type=int, default=200)
    g.add_argument("--num_generations",  type=int, default=4)
    g.add_argument("--batch_size",       type=int, default=4,
                   help="per_device_train_batch_size")
    g.add_argument("--per_device_train_batch_size", type=int, default=None,
                   help="alias for --batch_size")
    g.add_argument("--grad_accum",       type=int, default=2)
    g.add_argument("--gradient_accumulation_steps", type=int, default=None,
                   help="alias for --grad_accum")
    g.add_argument("--learning_rate",    type=float, default=5e-6)
    g.add_argument("--max_seq_length",   type=int, default=4096)
    g.add_argument("--max_completion_length", type=int, default=128)
    g.add_argument("--temperature",      type=float, default=1.0)
    g.add_argument("--top_p",            type=float, default=0.95)
    g.add_argument("--beta",             type=float, default=0.04)

    g = p.add_argument_group("Reward / Dataset (v3)")
    g.add_argument("--reward_version",       default="v3_anchored",
                   help="Reward shape identifier (v3_anchored is implemented)")
    g.add_argument("--bad_state_ratio",      type=float, default=0.0,
                   help="Fraction of training states captured AFTER a "
                        "sub-optimal/awkward perturbation (0.0–0.5)")
    g.add_argument("--reward_debug_samples", type=int, default=40)

    g = p.add_argument_group("Evaluation")
    g.add_argument("--eval_seeds",  type=int, default=3)
    g.add_argument("--skip_base_eval", action="store_true",
                   help="Skip base model eval (reuse existing results)")
    g.add_argument("--eval_best_of", type=int, default=1,
                   help="If > 1, also evaluate SFT and GRPO with best-of-N "
                        "sampling (base stays @1). Adds policies "
                        "'sft_qwen05@N' and 'grpo_qwen05@N' to the comparison.")
    g.add_argument("--bon_temperature", type=float, default=0.9,
                   help="Sampling temperature used for best-of-N candidates.")
    g.add_argument("--bon_top_p",       type=float, default=0.95,
                   help="Top-p used for best-of-N candidates.")

    g = p.add_argument_group("Output")
    g.add_argument("--out_dir", default="/workspace/grpo_designgym2_qwen05")
    g.add_argument("--output_dir", default=None,
                   help="alias for --out_dir")
    g.add_argument("--smoke_report_json", default=None,
                   help="If set, write a final per-policy summary JSON here")
    g.add_argument("--hub_model_id", default=None,
                   help="Override --output_repo (HF jobs alias)")

    args = p.parse_args()

    # Aliases
    if args.per_device_train_batch_size is not None:
        args.batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        args.grad_accum = args.gradient_accumulation_steps
    if args.hub_model_id:
        args.output_repo = args.hub_model_id
    if args.output_dir:
        args.out_dir = args.output_dir

    if args.smoke:
        # v3 smoke schedule (target ≤ 60 min on t4-small).
        # Tuned to fix v2's mode-collapse:
        #   beta 0.01 → 0.04 (canonical KL anchor — keeps GRPO close to SFT)
        #   lr   3e-5 → 2e-5 (less aggressive drift)
        #   temp 1.15 → 1.10 (SFT already covers format; trim exploration)
        #   max_steps     120 → 100 (faster, still enough to show signal)
        #   eval_seeds    5   → 4   (saves ~6 min eval, still 12 rollouts/policy)
        #   bad_state_ratio 0.25 → 0.30 (more contrast for GRPO groups)
        args.num_train_states = max(args.num_train_states, 1)
        if args.num_train_states == 400:  # i.e. user did not override
            args.num_train_states = 240
        if args.max_steps == 200:
            args.max_steps = 100
        if args.num_generations == 4:
            args.num_generations = 8
        if args.batch_size == 4:
            args.batch_size = 2
        if args.grad_accum == 2:
            args.grad_accum = 4
        if args.learning_rate == 5e-6:
            args.learning_rate = 2e-5
        if args.temperature == 1.0:
            args.temperature = 1.10
        if args.beta == 0.04:
            args.beta = 0.04  # explicit: keep canonical
        if args.eval_seeds == 3:
            args.eval_seeds = 4
        if args.bad_state_ratio == 0.0:
            args.bad_state_ratio = 0.30
        if args.eval_best_of == 1:
            args.eval_best_of = 4
        if args.smoke_report_json is None:
            args.smoke_report_json = str(Path(args.out_dir) / "smoke_report.json")
        # Bump output suffix so v3 runs don't overwrite v2 artifacts.
        if "smoke-v3" not in args.output_repo:
            base = args.output_repo
            for sfx in ("-smoke-v2", "-smoke", ):
                if base.endswith(sfx):
                    base = base[: -len(sfx)]
                    break
            args.output_repo = base + "-smoke-v3"
        log("🔬 SMOKE TEST MODE — v3_anchored schedule")

    return args


def _build_grpo_config(args, out: Path) -> GRPOConfig:
    """Build GRPOConfig, filtering out fields the installed TRL doesn't support."""
    desired: Dict[str, Any] = {
        "output_dir":                  str(out / "trainer_output"),
        "learning_rate":               args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_generations":             args.num_generations,
        "max_prompt_length":           args.max_seq_length,
        "max_completion_length":       args.max_completion_length,
        "max_steps":                   args.max_steps,
        "temperature":                 args.temperature,
        "top_p":                       args.top_p,
        "beta":                        args.beta,
        "logging_steps":               5,
        "save_steps":                  50,
        "save_total_limit":            2,
        "bf16":                        torch.cuda.is_bf16_supported(),
        "fp16":                        not torch.cuda.is_bf16_supported(),
        "report_to":                   "none",
        "remove_unused_columns":       False,
    }
    try:
        sig = inspect.signature(GRPOConfig.__init__)
        accepted = set(sig.parameters.keys())
        accepted.discard("self")
    except (TypeError, ValueError):
        accepted = set(desired.keys())
    filtered, dropped = {}, []
    for k, v in desired.items():
        if k in accepted:
            filtered[k] = v
        else:
            dropped.append(k)
    if dropped:
        log(f"GRPOConfig: dropping unsupported fields: {dropped}")
    return GRPOConfig(**filtered)


def main():
    global REWARD_VERSION, MAX_REWARD_DEBUG_ROWS
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    REWARD_VERSION = args.reward_version
    MAX_REWARD_DEBUG_ROWS = max(40, int(args.reward_debug_samples) * 6)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU:  {torch.cuda.get_device_name(0)}")
    log(f"Output: {out}")
    log(f"Reward version: {REWARD_VERSION}  bad_state_ratio={args.bad_state_ratio:.2f}")
    log(f"Sampling: temp={args.temperature}  top_p={args.top_p}  beta={args.beta}  "
        f"max_completion_length={args.max_completion_length}")

    # ── 1. Build dataset ────────────────────────────────────────────────
    train_dataset = build_grpo_dataset(
        args.num_train_states, seed=123,
        bad_state_ratio=float(args.bad_state_ratio),
    )
    train_dataset.save_to_disk(str(out / "grpo_train_dataset"))

    # ── 2. Load base model ──────────────────────────────────────────────
    log(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Evaluate BASE model ──────────────────────────────────────────
    eval_dfs: Dict[str, pd.DataFrame] = {}
    if not args.skip_base_eval:
        log("Evaluating BASE model…")
        FastLanguageModel.for_inference(model)
        eval_dfs["base_qwen05"] = evaluate_policy(
            model, tokenizer, "base_qwen05",
            range(args.eval_seeds), args.max_seq_length, out)
    cleanup_gpu()

    # ── 4. Load SFT adapter & evaluate ──────────────────────────────────
    log(f"Loading SFT adapter: {args.sft_adapter}")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    log("Evaluating SFT model…")
    FastLanguageModel.for_inference(model)
    eval_dfs["sft_qwen05"] = evaluate_policy(
        model, tokenizer, "sft_qwen05",
        range(args.eval_seeds), args.max_seq_length, out)
    if args.eval_best_of > 1:
        bon = int(args.eval_best_of)
        log(f"Evaluating SFT model best-of-{bon}…")
        eval_dfs[f"sft_qwen05@{bon}"] = evaluate_policy(
            model, tokenizer, f"sft_qwen05@{bon}",
            range(args.eval_seeds), args.max_seq_length, out,
            best_of=bon, bon_temperature=args.bon_temperature,
            bon_top_p=args.bon_top_p)

    # ── 5. GRPO Training ───────────────────────────────────────────────
    log("Switching to training mode…")
    model.train()

    grpo_config = _build_grpo_config(args, out)

    # transformers 5.x removed `warnings_issued`; TRL 0.24 still expects it
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=designgym_reward,
        args=grpo_config,
        train_dataset=train_dataset,
    )

    log(f"Starting GRPO training ({args.max_steps} steps)…")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log(f"GRPO training done in {elapsed / 60:.1f} min")

    save_training_logs(trainer.state.log_history, out)

    # ── 6. Evaluate GRPO model ──────────────────────────────────────────
    log("Evaluating GRPO model…")
    FastLanguageModel.for_inference(model)
    eval_dfs["grpo_qwen05"] = evaluate_policy(
        model, tokenizer, "grpo_qwen05",
        range(args.eval_seeds), args.max_seq_length, out)
    if args.eval_best_of > 1:
        bon = int(args.eval_best_of)
        log(f"Evaluating GRPO model best-of-{bon}…")
        eval_dfs[f"grpo_qwen05@{bon}"] = evaluate_policy(
            model, tokenizer, f"grpo_qwen05@{bon}",
            range(args.eval_seeds), args.max_seq_length, out,
            best_of=bon, bon_temperature=args.bon_temperature,
            bon_top_p=args.bon_top_p)

    # ── 7. Comparison table & plots ─────────────────────────────────────
    table = build_comparison(eval_dfs, out)

    if args.eval_best_of > 1:
        bon = int(args.eval_best_of)
        log("=" * 60)
        log(f"BEST-OF-{bon} HEADLINE COMPARISON")
        log("=" * 60)
        for metric in ("final_score", "instruction_score"):
            def _m(name):
                df = eval_dfs.get(name)
                return None if df is None or metric not in df.columns else float(df[metric].mean())
            sft1, grpo1 = _m("sft_qwen05"), _m("grpo_qwen05")
            sftN, grpoN = _m(f"sft_qwen05@{bon}"), _m(f"grpo_qwen05@{bon}")
            log(f"  {metric}:")
            log(f"    SFT@1   = {sft1:.4f}    GRPO@1   = {grpo1:.4f}    "
                f"Δ = {(grpo1 - sft1):+.4f}")
            if sftN is not None and grpoN is not None:
                log(f"    SFT@{bon}  = {sftN:.4f}    GRPO@{bon}  = {grpoN:.4f}    "
                    f"Δ = {(grpoN - sftN):+.4f}")
        log("=" * 60)

    # ── 7b. Reward debug + smoke report ────────────────────────────────
    debug_payload = write_reward_debug(out, n_sample=args.reward_debug_samples)

    if args.smoke_report_json:
        try:
            report: Dict[str, Any] = {
                "reward_version":  REWARD_VERSION,
                "bad_state_ratio": float(args.bad_state_ratio),
                "schedule": {
                    "num_train_states":       int(args.num_train_states),
                    "max_steps":              int(args.max_steps),
                    "num_generations":        int(args.num_generations),
                    "batch_size":             int(args.batch_size),
                    "grad_accum":             int(args.grad_accum),
                    "learning_rate":          float(args.learning_rate),
                    "temperature":            float(args.temperature),
                    "top_p":                  float(args.top_p),
                    "beta":                   float(args.beta),
                    "max_completion_length":  int(args.max_completion_length),
                    "eval_seeds":             int(args.eval_seeds),
                    "eval_best_of":           int(args.eval_best_of),
                    "bon_temperature":        float(args.bon_temperature),
                    "bon_top_p":              float(args.bon_top_p),
                },
                "policies": {},
            }
            for pol, df in eval_dfs.items():
                report["policies"][pol] = {
                    m: float(df[m].mean()) for m in EVAL_METRICS if m in df.columns
                }
            try:
                bon = int(args.eval_best_of)
                deltas: Dict[str, Dict[str, float]] = {}
                def _mean(name, metric):
                    df = eval_dfs.get(name)
                    if df is None or metric not in df.columns:
                        return None
                    return float(df[metric].mean())
                for metric in ("final_score", "instruction_score"):
                    base = _mean("base_qwen05", metric)
                    sft1 = _mean("sft_qwen05", metric)
                    grpo1 = _mean("grpo_qwen05", metric)
                    sftN = _mean(f"sft_qwen05@{bon}", metric) if bon > 1 else None
                    grpoN = _mean(f"grpo_qwen05@{bon}", metric) if bon > 1 else None
                    deltas[metric] = {
                        "base":              base,
                        "sft@1":              sft1,
                        "grpo@1":             grpo1,
                        f"sft@{bon}":         sftN,
                        f"grpo@{bon}":        grpoN,
                        "delta_base_sft1":   None if (base is None or sft1 is None) else sft1 - base,
                        "delta_sft1_grpo1":  None if (sft1 is None or grpo1 is None) else grpo1 - sft1,
                        f"delta_sft{bon}_grpo{bon}":
                            None if (sftN is None or grpoN is None) else grpoN - sftN,
                    }
                report["headline_deltas"] = deltas
            except Exception as exc:
                log(f"[WARN] failed to compute headline deltas: {exc}")
            if debug_payload is not None:
                report["reward_debug"] = {
                    k: debug_payload[k] for k in
                    ("reward_mean", "reward_std", "reward_min", "reward_max",
                     "action_type_counts", "reason_counts", "n_samples_collected")
                    if k in debug_payload
                }
            Path(args.smoke_report_json).write_text(json.dumps(report, indent=2))
            log(f"Smoke report written to {args.smoke_report_json}")
        except Exception as exc:
            log(f"[WARN] failed to write smoke report: {exc}")

    # ── 8. Save adapter & upload ────────────────────────────────────────
    adapter_dir = out / "designgym2_grpo_qwen05_adapter"
    log(f"Saving GRPO adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.output_repo and hf_token:
        log(f"Uploading to Hub: {args.output_repo}")
        create_repo(args.output_repo, repo_type="model", private=False,
                     exist_ok=True, token=hf_token)
        model.push_to_hub(args.output_repo, token=hf_token)
        tokenizer.push_to_hub(args.output_repo, token=hf_token)

        api = HfApi(token=hf_token)
        for pattern in ("*.csv", "*.json", "*.png", "plots/*.png"):
            for fp in out.glob(pattern):
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=f"results/{fp.name}",
                    repo_id=args.output_repo, repo_type="model")

    log("=" * 60)
    log("DONE — all results saved to: " + str(out))
    log("=" * 60)


if __name__ == "__main__":
    main()
