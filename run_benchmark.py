#!/usr/bin/env python3
"""DesignGym full benchmark — runs every (model x task x seed) combination
against the local server and prints an honest comparison report.

Usage:
    conda activate RL
    python run_benchmark.py

Prerequisites: the server must NOT be running — this script boots its own.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error

# ── Config ──────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:8000"
TASKS = ["poster_basic_v1", "editorial_cover_v1", "dense_flyer_v1"]
SEEDS = [0, 1, 2]
BACKENDS = [
    {"name": "heuristic",     "adapter": None,   "policy": "heuristic"},
    {"name": "base",          "adapter": "base", "policy": "llm"},
    {"name": "sft_finetuned", "adapter": "sft",  "policy": "llm"},
    {"name": "grpo_finetuned","adapter": "grpo", "policy": "llm"},
]
TIMEOUT_PER_EPISODE = 300  # seconds


def api(method: str, path: str, body: Optional[dict] = None, timeout: int = TIMEOUT_PER_EPISODE) -> dict:
    url = f"{SERVER_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_server(max_wait: int = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            api("GET", "/demo/ping", timeout=5)
            return True
        except Exception:
            time.sleep(1)
    return False


def wait_for_model_ready(max_wait: int = 180) -> dict:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            info = api("GET", "/demo/backend_info", timeout=5)
            if info.get("ready") or info.get("backend") == "none":
                return info
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return {"backend": "timeout"}


def switch_adapter(adapter_key: str) -> dict:
    return api("POST", "/demo/switch_adapter", {"adapter": adapter_key}, timeout=180)


def run_episode(policy: str, task_id: str, seed: int) -> dict:
    return api("POST", "/demo/run_episode", {
        "policy": policy,
        "task_id": task_id,
        "seed": seed,
    })


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["DESIGNGYM_BACKEND"] = "local"
    env["DESIGNGYM_ADAPTER"] = "sft"
    env["TORCH_NUM_THREADS"] = "4"
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.app"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    return proc


def kill_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ── Reporting ───────────────────────────────────────────────────────────
def fmt(n, d=4):
    if n is None:
        return "  N/A "
    return f"{n:>{d+3}.{d}f}"


def print_separator(char="─", width=100):
    print(char * width)


def print_report(results: List[Dict[str, Any]]):
    print("\n")
    print_separator("═")
    print("  DESIGNGYM BENCHMARK REPORT")
    print(f"  {len(results)} episodes · {len(TASKS)} tasks · {len(SEEDS)} seeds · {len(BACKENDS)} backends")
    print_separator("═")

    # ── Per-run detail table ──
    print(f"\n{'Backend':<16} {'Task':<22} {'Seed':>4}  {'Score':>7} {'Instr':>7} {'Steps':>5} {'Reward':>7} {'Time':>6}  {'Policy Tags'}")
    print_separator("─")
    for r in sorted(results, key=lambda x: (x["backend"], x["task_id"], x["seed"])):
        tags = set()
        for t in r.get("trajectory", []):
            tags.add(t.get("policy", "?"))
        tag_str = ", ".join(sorted(tags))
        s = r["summary"]
        print(
            f"{r['backend']:<16} {r['task_id']:<22} {r['seed']:>4}  "
            f"{fmt(s.get('final_score'), 4)} {fmt(s.get('instruction_score'), 4)} "
            f"{s.get('steps_taken', 0):>5} {fmt(s.get('total_reward'), 3)} "
            f"{s.get('wall_time_sec', 0):>5.1f}s  {tag_str}"
        )

    # ── Aggregate per backend ──
    print("\n")
    print_separator("═")
    print("  AGGREGATE BY BACKEND")
    print_separator("═")

    by_backend: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_backend[r["backend"]].append(r)

    print(f"\n{'Backend':<16} {'Runs':>4} {'Avg Score':>10} {'Avg Instr':>10} {'Avg Steps':>10} {'Avg Reward':>11} {'Avg Time':>9}")
    print_separator("─")
    backend_agg = {}
    for name in ["heuristic", "base", "sft_finetuned", "grpo_finetuned"]:
        runs = by_backend.get(name, [])
        if not runs:
            print(f"{name:<16} {'0':>4}  {'—':>9} {'—':>9} {'—':>9} {'—':>10} {'—':>8}")
            continue
        scores = [r["summary"].get("final_score", 0) for r in runs]
        instrs = [r["summary"].get("instruction_score", 0) for r in runs]
        steps = [r["summary"].get("steps_taken", 0) for r in runs]
        rewards = [r["summary"].get("total_reward", 0) for r in runs]
        times = [r["summary"].get("wall_time_sec", 0) for r in runs]
        avg_s = sum(scores) / len(scores)
        avg_i = sum(instrs) / len(instrs)
        avg_st = sum(steps) / len(steps)
        avg_r = sum(rewards) / len(rewards)
        avg_t = sum(times) / len(times)
        backend_agg[name] = {"score": avg_s, "instr": avg_i, "steps": avg_st, "reward": avg_r, "time": avg_t}
        print(
            f"{name:<16} {len(runs):>4} {avg_s:>10.4f} {avg_i:>10.4f} "
            f"{avg_st:>10.1f} {avg_r:>11.3f} {avg_t:>8.1f}s"
        )

    # ── Per-task breakdown ──
    print("\n")
    print_separator("═")
    print("  BREAKDOWN BY TASK")
    print_separator("═")

    for task in TASKS:
        print(f"\n  {task}")
        print(f"  {'Backend':<16} {'Avg Score':>10} {'Avg Instr':>10} {'Avg Reward':>11}")
        print(f"  " + "─" * 52)
        for name in ["heuristic", "base", "sft_finetuned", "grpo_finetuned"]:
            runs = [r for r in by_backend.get(name, []) if r["task_id"] == task]
            if not runs:
                print(f"  {name:<16} {'—':>9} {'—':>9} {'—':>10}")
                continue
            avg_s = sum(r["summary"]["final_score"] for r in runs) / len(runs)
            avg_i = sum(r["summary"]["instruction_score"] for r in runs) / len(runs)
            avg_r = sum(r["summary"]["total_reward"] for r in runs) / len(runs)
            print(f"  {name:<16} {avg_s:>10.4f} {avg_i:>10.4f} {avg_r:>11.3f}")

    # ── Honest assessment ──
    print("\n")
    print_separator("═")
    print("  HONEST ASSESSMENT")
    print_separator("═")
    print()

    if "sft_finetuned" in backend_agg and "heuristic" in backend_agg:
        sft = backend_agg["sft_finetuned"]
        heur = backend_agg["heuristic"]
        diff = sft["score"] - heur["score"]
        if diff > 0.01:
            print(f"  SFT vs Heuristic:  +{diff:.4f} avg score  —  SFT fine-tune IMPROVES over heuristic")
        elif diff < -0.01:
            print(f"  SFT vs Heuristic:  {diff:.4f} avg score  —  SFT fine-tune WORSE than heuristic")
        else:
            print(f"  SFT vs Heuristic:  {diff:+.4f} avg score  —  roughly EQUAL")

    if "grpo_finetuned" in backend_agg and "heuristic" in backend_agg:
        grpo = backend_agg["grpo_finetuned"]
        heur = backend_agg["heuristic"]
        diff = grpo["score"] - heur["score"]
        if diff > 0.01:
            print(f"  GRPO vs Heuristic: +{diff:.4f} avg score  —  GRPO fine-tune IMPROVES over heuristic")
        elif diff < -0.01:
            print(f"  GRPO vs Heuristic: {diff:.4f} avg score  —  GRPO fine-tune WORSE than heuristic")
        else:
            print(f"  GRPO vs Heuristic: {diff:+.4f} avg score  —  roughly EQUAL")

    if "sft_finetuned" in backend_agg and "base" in backend_agg:
        sft = backend_agg["sft_finetuned"]
        base = backend_agg["base"]
        diff = sft["score"] - base["score"]
        if diff > 0.01:
            print(f"  SFT vs Base:       +{diff:.4f} avg score  —  LoRA adapter provides real lift over base model")
        elif diff < -0.01:
            print(f"  SFT vs Base:       {diff:.4f} avg score  —  LoRA adapter WORSE than base (check training)")
        else:
            print(f"  SFT vs Base:       {diff:+.4f} avg score  —  no meaningful difference (adapter may not be helping)")

    if "grpo_finetuned" in backend_agg and "base" in backend_agg:
        grpo = backend_agg["grpo_finetuned"]
        base = backend_agg["base"]
        diff = grpo["score"] - base["score"]
        if diff > 0.01:
            print(f"  GRPO vs Base:      +{diff:.4f} avg score  —  GRPO adapter provides real lift over base model")
        elif diff < -0.01:
            print(f"  GRPO vs Base:      {diff:.4f} avg score  —  GRPO adapter WORSE than base (check training)")
        else:
            print(f"  GRPO vs Base:      {diff:+.4f} avg score  —  no meaningful difference")

    if "sft_finetuned" in backend_agg and "grpo_finetuned" in backend_agg:
        sft = backend_agg["sft_finetuned"]
        grpo = backend_agg["grpo_finetuned"]
        diff = sft["score"] - grpo["score"]
        if abs(diff) < 0.005:
            print(f"  SFT vs GRPO:       {diff:+.4f} avg score  —  virtually identical")
        elif diff > 0:
            print(f"  SFT vs GRPO:       +{diff:.4f} avg score  —  SFT edges out GRPO")
        else:
            print(f"  SFT vs GRPO:       {diff:.4f} avg score  —  GRPO edges out SFT")

    # ── LLM parse success rate ──
    print()
    for name in ["base", "sft_finetuned", "grpo_finetuned"]:
        runs = by_backend.get(name, [])
        if not runs:
            continue
        total_steps = 0
        llm_ok_steps = 0
        for r in runs:
            for t in r.get("trajectory", []):
                total_steps += 1
                tag = t.get("policy", "")
                if tag.startswith("finetuned_") or tag == "local_base" or tag == "router_base":
                    llm_ok_steps += 1
        pct = (llm_ok_steps / total_steps * 100) if total_steps else 0
        print(f"  {name:<16} LLM steered {llm_ok_steps}/{total_steps} steps ({pct:.0f}%)"
              f"  — {'GOOD' if pct > 60 else 'LOW: model often falls back to heuristic'}")

    print()
    print_separator("═")
    print()


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DesignGym Full Benchmark")
    print(f"  {len(BACKENDS)} backends x {len(TASKS)} tasks x {len(SEEDS)} seeds = {len(BACKENDS)*len(TASKS)*len(SEEDS)} episodes")
    print("=" * 60)

    # Boot server
    print("\n[1/4] Starting server ...", flush=True)
    server = start_server()
    try:
        if not wait_for_server(max_wait=90):
            print("FATAL: server did not start within 90s")
            kill_server(server)
            sys.exit(1)
        print("  Server is up.", flush=True)

        # Wait for initial model warm-up
        print("[2/4] Waiting for model warm-up ...", flush=True)
        info = wait_for_model_ready(max_wait=180)
        print(f"  Backend: {info.get('backend')}  device={info.get('device')}  ready={info.get('ready')}", flush=True)

        # Run all combinations
        print(f"[3/4] Running {len(BACKENDS)*len(TASKS)*len(SEEDS)} episodes ...\n", flush=True)
        results: List[Dict[str, Any]] = []
        total = len(BACKENDS) * len(TASKS) * len(SEEDS)
        done = 0

        for backend_cfg in BACKENDS:
            bname = backend_cfg["name"]
            adapter = backend_cfg["adapter"]
            policy = backend_cfg["policy"]

            # Switch adapter if needed (or skip for heuristic)
            if adapter is not None:
                print(f"  Switching to adapter={adapter} ...", end=" ", flush=True)
                try:
                    resp = switch_adapter(adapter)
                    print(f"OK ({resp.get('info', {}).get('load_seconds', '?')}s)", flush=True)
                except Exception as e:
                    print(f"FAILED: {e}", flush=True)
                    # Still try to run — will fall back
                time.sleep(0.5)

            for task_id in TASKS:
                for seed in SEEDS:
                    done += 1
                    label = f"[{done}/{total}] {bname:<16} {task_id:<22} seed={seed}"
                    print(f"  {label} ...", end=" ", flush=True)
                    t0 = time.time()
                    try:
                        data = run_episode(policy, task_id, seed)
                        s = data.get("summary", {})
                        elapsed = time.time() - t0
                        score = s.get("final_score", 0)
                        print(f"score={score:.4f}  time={elapsed:.1f}s", flush=True)
                        results.append({
                            "backend": bname,
                            "task_id": task_id,
                            "seed": seed,
                            "policy": policy,
                            "summary": s,
                            "trajectory": data.get("trajectory", []),
                        })
                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"ERROR: {e}  ({elapsed:.1f}s)", flush=True)
                        results.append({
                            "backend": bname,
                            "task_id": task_id,
                            "seed": seed,
                            "policy": policy,
                            "summary": {"final_score": 0, "instruction_score": 0,
                                        "steps_taken": 0, "total_reward": 0,
                                        "wall_time_sec": elapsed},
                            "trajectory": [],
                            "error": str(e),
                        })

        # Print report
        print("\n[4/4] Generating report ...", flush=True)
        print_report(results)

        # Save raw JSON
        out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Raw results saved to {out_path}")

    finally:
        print("\nShutting down server ...", flush=True)
        kill_server(server)
        print("Done.")


if __name__ == "__main__":
    main()
