from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

try:
    from openenv.core.env_server import create_fastapi_app
except Exception:
    from fastapi import FastAPI

    def create_fastapi_app(env_cls, action_cls, observation_cls):
        app = FastAPI(title="DesignGym")

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        return app

try:
    from ..models import DesignGymAction, DesignGymObservation
except Exception:
    from models import DesignGymAction, DesignGymObservation

try:
    from .DesignGym_environment import DesignGymEnvironment, TASKS
except Exception:
    from server.DesignGym_environment import DesignGymEnvironment, TASKS

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import inference as INF
    _INFERENCE_OK = True
except Exception:
    INF = None
    _INFERENCE_OK = False

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

_LLM_CLIENT_CACHE: Dict[str, Any] = {"client": None, "key": None}


def _get_llm_client():
    if not (_INFERENCE_OK and _OPENAI_OK):
        return None
    token = os.getenv("HF_TOKEN") or getattr(INF, "HF_TOKEN", None)
    base = os.getenv("API_BASE_URL", getattr(INF, "API_BASE_URL", "https://router.huggingface.co/v1"))
    if not token:
        return None
    cache_key = f"{token[:6]}::{base}"
    if _LLM_CLIENT_CACHE["key"] != cache_key:
        _LLM_CLIENT_CACHE["client"] = OpenAI(base_url=base, api_key=token)
        _LLM_CLIENT_CACHE["key"] = cache_key
    return _LLM_CLIENT_CACHE["client"]


app = create_fastapi_app(
    DesignGymEnvironment,
    DesignGymAction,
    DesignGymObservation,
)

DEMO_ENV = DesignGymEnvironment()
_LAST_OBS: Dict[str, Any] = {"obs": None}


def _current_obs():
    """Return the most recent observation, falling back to the env's internal builder."""
    if _LAST_OBS.get("obs") is not None:
        return _LAST_OBS["obs"]
    builder = getattr(DEMO_ENV, "_observation", None)
    if callable(builder):
        try:
            return builder(message="snapshot")
        except Exception:
            return None
    return None

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
    
WEB_DIR = ROOT_DIR / "web"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/web", include_in_schema=False)
def web_index_no_slash():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/web/", include_in_schema=False)
def web_index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/web/{path:path}", include_in_schema=False)
def web_static(path: str):
    file_path = WEB_DIR / path
    if not file_path.exists() or not file_path.is_file():
        return FileResponse(str(WEB_DIR / "index.html"))
    return FileResponse(str(file_path))



def _task_description(task_id: str) -> str:
    if task_id == "poster_basic_v1":
        return "Poster layout optimization with hero image, title hierarchy, CTA placement, and alignment."
    if task_id == "editorial_cover_v1":
        return "Editorial cover optimization with masthead preservation, headline stack, and reading order."
    if task_id == "dense_flyer_v1":
        return "Dense flyer optimization with support-group reflow, spacing, occupancy, and caption alignment."
    return "Design layout optimization task."


def _task_catalog():
    difficulty_map = {
        "poster_basic_v1": "easy",
        "editorial_cover_v1": "medium",
        "dense_flyer_v1": "hard",
    }

    catalog = []
    for task_id, spec in TASKS.items():
        catalog.append(
            {
                "task_id": task_id,
                "difficulty": difficulty_map.get(task_id, "medium"),
                "graded": True,
                "grader": {
                    "type": "programmatic",
                    "name": "deterministic_layout_utility",
                    "deterministic": True,
                    "source": "server/DesignGym_environment.py",
                },
                "description": _task_description(task_id),
                "max_steps": int(spec.get("max_steps", 0)),
                "instance_id": spec.get("instance_id"),
                "reward_range": [0.0, 1.0],
                "score_range": [0.0, 1.0],
            }
        )
    return catalog

@app.get("/info")
def info():
    tasks = _task_catalog()
    return JSONResponse(
        {
            "name": "DesignGym",
            "description": "OpenEnv-compatible reinforcement learning environment for design layout optimization.",
            "task_count": len(tasks),
            "default_task_id": "poster_basic_v1",
            "tasks": tasks,
            "reward_range": [0.0, 1.0],
            "score_range": [0.0, 1.0],
            "supports_seeded_reset": True,
            "supports_task_id_reset": True,
        }
    )

@app.get("/demo/ping")
def demo_ping():
    return {"ok": True, "message": "DesignGym 2.0 demo endpoints are live"}

@app.get("/tasks")
def tasks():
    return JSONResponse(
        {
            "tasks": _task_catalog()
        }
    )

@app.post("/demo/reset")
async def demo_reset(request: Request):
    payload = await request.json()
    obs = DEMO_ENV.reset(**payload)
    _LAST_OBS["obs"] = obs
    return {
        "observation": obs.model_dump(),
        "state": DEMO_ENV.state.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/demo/step")
async def demo_step(request: Request):
    payload = await request.json()

    action_payload = payload.get("action", payload)

    action = DesignGymAction(**action_payload)
    obs = DEMO_ENV.step(action)
    _LAST_OBS["obs"] = obs

    return {
        "observation": obs.model_dump(),
        "state": DEMO_ENV.state.model_dump(),
        "reward": float(DEMO_ENV.state.last_reward),
        "done": bool(DEMO_ENV.state.done),
    }


@app.get("/demo/state")
def demo_state():
    return {
        "state": DEMO_ENV.state.model_dump()
    }


def _summary_from_state(state, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [t.get("reward", 0.0) for t in trajectory]
    valid = [t for t in trajectory if t.get("error") is None]
    finalized = any(t.get("action_type") == "finalize" for t in trajectory)
    return {
        "final_score": float(getattr(state, "current_score", 0.0) or 0.0),
        "instruction_score": float(getattr(state, "instruction_score", 0.0) or 0.0),
        "phase_score": float(getattr(state, "phase_score", 0.0) or 0.0),
        "best_score_so_far": float(getattr(state, "best_score_so_far", 0.0) or 0.0),
        "steps_taken": len(trajectory),
        "total_reward": sum(rewards),
        "valid_action_rate": (len(valid) / len(trajectory)) if trajectory else 0.0,
        "finalized": finalized,
        "done": bool(getattr(state, "done", False)),
        "phase": getattr(state, "phase", None),
        "task_id": getattr(state, "task_id", None),
    }


def _record_step(step: int, action, obs, prev_score: float) -> Dict[str, Any]:
    return {
        "step": step,
        "action_type": getattr(action, "action_type", None),
        "action": getattr(action, "canonical", lambda: str(action))(),
        "reward": float(getattr(obs, "last_reward", 0.0) or 0.0),
        "score": float(getattr(obs, "current_score", 0.0) or 0.0),
        "delta_score": float(getattr(obs, "current_score", 0.0) or 0.0) - prev_score,
        "instruction_score": float(getattr(obs, "instruction_score", 0.0) or 0.0),
        "worst_metrics": list(getattr(obs, "worst_metrics", []) or []),
        "error": getattr(obs, "last_action_error", None),
        "done": bool(getattr(obs, "done", False)),
    }


def _choose_action(policy: str, step: int, obs, history: List[str], rewards: List[float], recent_actions: List[str]):
    """Pick one action using the requested policy. Falls back to heuristic if LLM unavailable."""
    if not _INFERENCE_OK:
        # Hard fallback: just finalize if we can't import inference.py
        return DesignGymAction(action_type="finalize"), "fallback_no_inference"

    if policy == "heuristic":
        return INF.heuristic_action(step, obs, rewards, recent_actions), "heuristic"

    client = _get_llm_client()
    if client is None:
        # SFT/LLM requested but no HF_TOKEN -> degrade gracefully to local-best candidate
        action = INF.get_model_action_sync(None, step, obs, history, rewards, recent_actions)
        return action, "llm_fallback_local_best"

    action = INF.get_model_action_sync(client, step, obs, history, rewards, recent_actions)
    label = "sft_llm" if policy == "sft" else f"llm_{policy}"
    return action, label


@app.post("/demo/policy_step")
async def demo_policy_step(request: Request):
    """Run a single step using the requested policy ({"policy": "heuristic"|"sft"})."""
    payload = await request.json()
    policy = (payload.get("policy") or "heuristic").lower()

    obs = _current_obs()
    if obs is None:
        return JSONResponse(
            status_code=409,
            content={"error": "no_active_episode", "hint": "Call /demo/reset first."},
        )

    state = DEMO_ENV.state
    step = int(getattr(state, "step_count", 0) or 0) + 1
    recent_actions: List[str] = list(getattr(state, "action_history", []) or [])
    rewards: List[float] = []  # not tracked on state; OK since heuristic mostly checks last action
    history = recent_actions[-4:]

    prev_score = float(getattr(obs, "current_score", 0.0) or 0.0)
    action, used = _choose_action(policy, step, obs, history, rewards, recent_actions)
    obs_after = DEMO_ENV.step(action)
    _LAST_OBS["obs"] = obs_after
    record = _record_step(step, action, obs_after, prev_score)
    record["policy"] = used

    return {
        "observation": obs_after.model_dump(),
        "state": DEMO_ENV.state.model_dump(),
        "step_record": record,
        "reward": record["reward"],
        "done": record["done"],
    }


@app.post("/demo/run_episode")
async def demo_run_episode(request: Request):
    """Run a full episode server-side and return the trajectory + summary in one call.

    Body: {"policy": "heuristic"|"sft", "task_id": str, "seed": int, "max_steps": int}
    """
    payload = await request.json()
    policy = (payload.get("policy") or "heuristic").lower()
    task_id = payload.get("task_id") or "poster_basic_v1"
    seed = int(payload.get("seed") or 0)
    max_steps_override = payload.get("max_steps")

    t0 = time.time()
    obs = DEMO_ENV.reset(task_id=task_id, seed=seed)
    _LAST_OBS["obs"] = obs

    declared_max = int(getattr(obs, "max_steps", 8) or 8)
    if max_steps_override:
        declared_max = min(declared_max, int(max_steps_override))

    trajectory: List[Dict[str, Any]] = []
    history: List[str] = []
    rewards: List[float] = []
    recent_actions: List[str] = []

    try:
        for step in range(1, declared_max + 1):
            if bool(getattr(obs, "done", False)):
                break

            prev_score = float(getattr(obs, "current_score", 0.0) or 0.0)
            action, used = _choose_action(policy, step, obs, history, rewards, recent_actions)
            obs = DEMO_ENV.step(action)
            _LAST_OBS["obs"] = obs
            record = _record_step(step, action, obs, prev_score)
            record["policy"] = used

            trajectory.append(record)
            history.append(record["action"])
            rewards.append(record["reward"])
            recent_actions.append(record["action"])

            if record["done"]:
                break

        summary = _summary_from_state(DEMO_ENV.state, trajectory)
        summary["policy_requested"] = policy
        summary["llm_available"] = _get_llm_client() is not None
        summary["wall_time_sec"] = round(time.time() - t0, 3)

        return {
            "summary": summary,
            "trajectory": trajectory,
            "final_observation": obs.model_dump(),
            "final_state": DEMO_ENV.state.model_dump(),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "trace": traceback.format_exc().splitlines()[-12:],
                "trajectory": trajectory,
            },
        )


@app.get("/demo/policies")
def demo_policies():
    """Tell the frontend which policies are usable right now."""
    llm_ok = _get_llm_client() is not None
    return {
        "policies": [
            {
                "id": "heuristic",
                "label": "Heuristic Planner",
                "available": _INFERENCE_OK,
                "description": "Rule-based planner from inference.py — fast, no API key needed.",
            },
            {
                "id": "sft",
                "label": "SFT-LLM Picker" if llm_ok else "SFT-LLM Picker (offline → falls back to local best)",
                "available": _INFERENCE_OK,
                "description": (
                    f"Calls Hugging Face Inference Providers ({getattr(INF, 'MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')}) "
                    "to pick from candidate actions. Requires HF_TOKEN."
                ),
                "llm_active": llm_ok,
            },
        ],
        "llm_active": llm_ok,
        "model_name": getattr(INF, "MODEL_NAME", None) if _INFERENCE_OK else None,
    }


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()