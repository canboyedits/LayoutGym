from __future__ import annotations

import os

import uvicorn

try:
    from openenv.core.env_server import create_fastapi_app
except Exception:
    from fastapi import FastAPI

    def create_fastapi_app(env_cls, action_cls, observation_cls):
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        return app

try:
    from ..models import DesignGymAction, DesignGymObservation
except Exception:
    from models import DesignGymAction, DesignGymObservation

try:
    from .DesignGym_environment import DesignGymEnvironment
except Exception:
    from server.DesignGym_environment import DesignGymEnvironment


app = create_fastapi_app(
    DesignGymEnvironment,
    DesignGymAction,
    DesignGymObservation,
)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()