from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse 
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


app = create_fastapi_app(
    DesignGymEnvironment,
    DesignGymAction,
    DesignGymObservation,
)

DEMO_ENV = DesignGymEnvironment()

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

@app.get("/")
def home():
    return RedirectResponse(url="/web/")


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
    return {
        "observation": obs.model_dump(),
        "state": DEMO_ENV.state.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/demo/step")
async def demo_step(request: Request):
    payload = await request.json()

    # Accept either {"action": {...}} or raw {...}
    action_payload = payload.get("action", payload)

    action = DesignGymAction(**action_payload)
    obs = DEMO_ENV.step(action)

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


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()