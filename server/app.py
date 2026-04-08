from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
    from .DesignGym_environment import DesignGymEnvironment
except Exception:
    from server.DesignGym_environment import DesignGymEnvironment


app = create_fastapi_app(
    DesignGymEnvironment,
    DesignGymAction,
    DesignGymObservation,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    image_block = ""
    if ASSETS_DIR.exists() and (ASSETS_DIR / "DesignGym.png").exists():
        image_block = """
        <div class="hero-card">
          <img src="/assets/DesignGym.png" alt="DesignGym architecture diagram" class="hero-image" />
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>DesignGym</title>
        <style>
          :root {{
            --bg: #0b1020;
            --panel: #121932;
            --panel-2: #1a2344;
            --text: #eef2ff;
            --muted: #b9c1e0;
            --accent: #7c9cff;
            --accent-2: #89f0c7;
            --border: rgba(255,255,255,0.10);
            --shadow: 0 18px 50px rgba(0,0,0,0.32);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
              radial-gradient(circle at top left, rgba(124,156,255,0.16), transparent 28%),
              radial-gradient(circle at top right, rgba(137,240,199,0.14), transparent 24%),
              linear-gradient(180deg, #0a0f1d 0%, #0b1020 100%);
            color: var(--text);
          }}
          .wrap {{
            max-width: 1120px;
            margin: 0 auto;
            padding: 40px 20px 56px;
          }}
          .header {{
            display: flex;
            gap: 18px;
            align-items: center;
            margin-bottom: 24px;
          }}
          .badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            background: rgba(255,255,255,0.04);
            font-size: 14px;
          }}
          h1 {{
            margin: 0;
            font-size: clamp(34px, 5vw, 56px);
            line-height: 1.05;
            letter-spacing: -0.03em;
          }}
          .subtitle {{
            max-width: 860px;
            margin: 14px 0 0;
            color: var(--muted);
            font-size: 18px;
            line-height: 1.65;
          }}
          .grid {{
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 24px;
            margin-top: 28px;
          }}
          .card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            overflow: hidden;
          }}
          .card-inner {{
            padding: 22px;
          }}
          .hero-card {{
            background: linear-gradient(180deg, rgba(124,156,255,0.08), rgba(255,255,255,0.03));
            border: 1px solid var(--border);
            border-radius: 22px;
            overflow: hidden;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
          }}
          .hero-image {{
            display: block;
            width: 100%;
            height: auto;
            background: #0e1430;
          }}
          h2 {{
            margin: 0 0 14px;
            font-size: 22px;
            letter-spacing: -0.02em;
          }}
          p {{
            margin: 0 0 14px;
            color: var(--muted);
            line-height: 1.65;
          }}
          ul {{
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            line-height: 1.85;
          }}
          li + li {{
            margin-top: 4px;
          }}
          .links {{
            display: grid;
            gap: 12px;
            margin-top: 14px;
          }}
          a {{
            color: var(--text);
            text-decoration: none;
          }}
          .link-card {{
            display: block;
            padding: 14px 16px;
            border-radius: 16px;
            background: var(--panel-2);
            border: 1px solid var(--border);
            transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease;
          }}
          .link-card:hover {{
            transform: translateY(-1px);
            border-color: rgba(124,156,255,0.5);
            background: #212d59;
          }}
          .link-title {{
            font-weight: 700;
            color: var(--text);
            display: block;
            margin-bottom: 4px;
          }}
          .link-sub {{
            color: var(--muted);
            font-size: 14px;
          }}
          code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.95em;
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 2px 7px;
            color: var(--accent-2);
          }}
          .footer {{
            margin-top: 28px;
            color: var(--muted);
            font-size: 14px;
          }}
          @media (max-width: 900px) {{
            .grid {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <main class="wrap">
          <div class="header">
            <div class="badge">🎨 OpenEnv • RL Layout Environment</div>
          </div>

          <h1>DesignGym</h1>
          <p class="subtitle">
            A reinforcement learning environment for design layout optimization across posters,
            editorial covers, and dense flyers. The agent improves alignment, hierarchy, spacing,
            reading order, occupancy, and semantic placement through structured actions.
          </p>
          <br>

          {image_block}

          <section class="grid">
            <div class="card">
              <div class="card-inner">
                <h2>What this Space serves</h2>
                <p>
                  This Hugging Face Space hosts the live OpenEnv-compatible DesignGym environment.
                  It is designed for evaluation, inference, and deployment testing.
                </p>
                <ul>
                  <li><code>POST /reset</code> starts a task episode</li>
                  <li><code>POST /step</code> applies an action</li>
                  <li><code>GET /state</code> returns the live state</li>
                  <li><code>GET /health</code> returns a health check</li>
                  <li><code>/docs</code> shows FastAPI/OpenAPI docs</li>
                </ul>
              </div>
            </div>

            <div class="card">
              <div class="card-inner">
                <h2>Quick links</h2>
                <div class="links">
                  <a class="link-card" href="/docs">
                    <span class="link-title">API Docs</span>
                    <span class="link-sub">Inspect endpoints and schemas</span>
                  </a>
                  <a class="link-card" href="/health">
                    <span class="link-title">Health Check</span>
                    <span class="link-sub">Simple runtime status endpoint</span>
                  </a>
                  <a class="link-card" href="/assets/DesignGym.png">
                    <span class="link-title">Architecture Image</span>
                    <span class="link-sub">Open the repo asset directly</span>
                  </a>
                </div>
              </div>
            </div>
          </section>

          <p class="footer">
            DesignGym is deployed as a Docker-based Hugging Face Space and packaged for OpenEnv validation.
          </p>
        </main>
      </body>
    </html>
    """


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()