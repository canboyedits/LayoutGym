// DesignGym 2.0 — minimal demo client.
// Talks only to the backend /demo/* endpoints; all policy logic lives server-side.

const $ = (id) => document.getElementById(id);

const taskSel = $("task");
const runBtn = $("run");
const stepBtn = $("step");
const resetBtn = $("reset");
const statusBox = $("status");
const scrollDetailsBtn = $("scroll-details");
const canvas = $("canvas");

const scoreFinal = $("score-final");
const scoreInstr = $("score-instr");
const scoreSteps = $("score-steps");
const scoreReward = $("score-reward");
const scoreValid = $("score-valid");

const logBody = $("log-body");
const logCount = $("log-count");
const rawSummary = $("raw-summary");

let llmActive = false;
let lastTrajectory = [];

window.addEventListener("error", (e) => setStatus(`JS error: ${e.message}`, "error"));
window.addEventListener("unhandledrejection", (e) => setStatus(`Promise: ${e.reason}`, "error"));

function setStatus(text, kind) {
  statusBox.textContent = text;
  statusBox.className = "status " + (kind || "muted");
}

function selectedPolicy() {
  const r = document.querySelector('input[name="policy"]:checked');
  return r ? r.value : "heuristic";
}

function policyLabel(id) {
  return id === "sft" ? "SFT-LLM Picker" : "Heuristic Planner";
}

function rectColor(role, type) {
  const key = role || type || "default";
  const colors = {
    title: "#bfdbfe", subtitle: "#dbeafe", image: "#bbf7d0",
    cta: "#fecaca", logo: "#fde68a", badge: "#ddd6fe",
    body: "#e2e8f0", caption: "#fef3c7", shape: "#ddd6fe",
    masthead: "#fed7aa", headline: "#a7f3d0", default: "#e5e7eb"
  };
  return colors[key] || colors.default;
}

function drawState(state) {
  canvas.innerHTML = "";

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", 0); bg.setAttribute("y", 0);
  bg.setAttribute("width", 800); bg.setAttribute("height", 1000);
  bg.setAttribute("fill", "#f8fafc");
  canvas.appendChild(bg);

  const elements = state?.elements || [];
  if (!elements.length) {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", 40); t.setAttribute("y", 60);
    t.setAttribute("fill", "#0f172a"); t.setAttribute("font-size", "22");
    t.textContent = "Reset to load a layout.";
    canvas.appendChild(t);
    return;
  }

  for (const el of elements) {
    const b = el.bbox || el.box;
    if (!b || b.length < 4) continue;
    const x = Number(b[0]) * 800;
    const y = Number(b[1]) * 1000;
    const w = Number(b[2]) * 800;
    const h = Number(b[3]) * 1000;

    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x); r.setAttribute("y", y);
    r.setAttribute("width", w); r.setAttribute("height", h);
    r.setAttribute("rx", 8);
    r.setAttribute("fill", rectColor(el.role, el.type));
    r.setAttribute("stroke", "#0f172a"); r.setAttribute("stroke-width", 2);
    canvas.appendChild(r);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x + 8); label.setAttribute("y", y + 22);
    label.setAttribute("fill", "#0f172a"); label.setAttribute("font-size", "15");
    label.setAttribute("font-family", "ui-monospace, monospace");
    label.textContent = el.id || "element";
    canvas.appendChild(label);

    const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
    sub.setAttribute("x", x + 8); sub.setAttribute("y", y + 42);
    sub.setAttribute("fill", "#334155"); sub.setAttribute("font-size", "12");
    sub.setAttribute("font-family", "ui-monospace, monospace");
    sub.textContent = el.role || el.type || "";
    canvas.appendChild(sub);
  }
}

function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}
function fmtPct(n) {
  if (n === null || n === undefined) return "—";
  return `${(Number(n) * 100).toFixed(0)}%`;
}

function renderScores(summary, state) {
  const finalScore = summary?.final_score ?? state?.current_score;
  const instrScore = summary?.instruction_score ?? state?.instruction_score;
  scoreFinal.textContent = fmt(finalScore, 3);
  scoreInstr.textContent = fmt(instrScore, 3);
  scoreSteps.textContent = (summary?.steps_taken ?? state?.step_count ?? 0).toString();
  scoreReward.textContent = fmt(summary?.total_reward ?? 0, 2);
  scoreValid.textContent = summary?.valid_action_rate !== undefined
    ? fmtPct(summary.valid_action_rate)
    : "—";
}

function renderLog(trajectory) {
  logBody.innerHTML = "";
  for (const t of trajectory) {
    const tr = document.createElement("tr");
    if (t.error) tr.className = "error-row";
    const deltaCls = t.delta_score > 0 ? "delta-pos" : (t.delta_score < 0 ? "delta-neg" : "");
    const status = t.error ? `❌ ${t.error}` : (t.action_type === "finalize" ? "🏁 finalize" : "✓");
    tr.innerHTML = `
      <td>${t.step}</td>
      <td>${t.action}</td>
      <td>${fmt(t.reward, 3)}</td>
      <td class="${deltaCls}">${(t.delta_score >= 0 ? "+" : "") + fmt(t.delta_score, 3)}</td>
      <td>${fmt(t.score, 3)}</td>
      <td>${(t.worst_metrics || []).slice(0, 2).join(", ") || "—"}</td>
      <td>${status}</td>
    `;
    logBody.appendChild(tr);
  }
  logCount.textContent = trajectory.length
    ? `· ${trajectory.length} step${trajectory.length === 1 ? "" : "s"}`
    : "— no steps yet";
}

function setRawSummary(payload) {
  rawSummary.textContent = JSON.stringify(payload, null, 2);
}

async function fetchPolicies() {
  try {
    const res = await fetch("/demo/policies");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    llmActive = !!data.llm_active;
  } catch (err) {
    llmActive = false;
  }
}

async function resetEnv() {
  setStatus("Resetting environment…", "running");
  try {
    const res = await fetch("/demo/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskSel.value, seed: 0 }),
    });
    if (!res.ok) throw new Error(`reset failed (${res.status})`);
    const payload = await res.json();
    drawState(payload.state);
    lastTrajectory = [];
    renderLog([]);
    renderScores({ steps_taken: 0, total_reward: 0, valid_action_rate: 0 }, payload.state);
    setStatus(`Ready: ${taskSel.value}. Click Run to start.`, "muted");
    setRawSummary({ task_id: taskSel.value, brief: payload.observation?.brief });
  } catch (err) {
    setStatus(`Reset error: ${err.message}`, "error");
  }
}

async function stepOnce() {
  setStatus(`Stepping (policy=${selectedPolicy()})…`, "running");
  try {
    const res = await fetch("/demo/policy_step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy: selectedPolicy() }),
    });
    if (res.status === 409) {
      await resetEnv();
      return stepOnce();
    }
    if (!res.ok) throw new Error(`step failed (${res.status})`);
    const data = await res.json();
    drawState(data.state);

    lastTrajectory.push(data.step_record);
    renderLog(lastTrajectory);

    const liveSummary = {
      final_score: data.state.current_score,
      instruction_score: data.state.instruction_score,
      steps_taken: lastTrajectory.length,
      total_reward: lastTrajectory.reduce((a, r) => a + (r.reward || 0), 0),
      valid_action_rate:
        lastTrajectory.filter((r) => !r.error).length / lastTrajectory.length,
    };
    renderScores(liveSummary, data.state);
    setRawSummary({ step_record: data.step_record, summary: liveSummary });

    if (data.done) {
      setStatus("Episode complete.", "success");
    } else {
      setStatus(`Step ${data.step_record.step} done. Click Step or Run.`, "muted");
    }
  } catch (err) {
    setStatus(`Step error: ${err.message}`, "error");
  }
}

async function runEpisode() {
  const policy = selectedPolicy();
  setStatus(`Running episode with ${policyLabel(policy)}… (this may take a few seconds for SFT)`, "running");
  runBtn.disabled = true;
  stepBtn.disabled = true;
  resetBtn.disabled = true;
  try {
    const res = await fetch("/demo/run_episode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy, task_id: taskSel.value, seed: 0 }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status} — ${txt.slice(0, 240)}`);
    }
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    drawState(data.final_state);
    lastTrajectory = data.trajectory || [];
    renderLog(lastTrajectory);
    renderScores(data.summary, data.final_state);
    setRawSummary(data.summary);

    const tag = data.summary.llm_available ? "live LLM" : "local-best fallback";
    setStatus(
      `Done in ${data.summary.wall_time_sec}s — ${data.summary.steps_taken} steps · final score ${fmt(data.summary.final_score, 3)} (${tag})`,
      "success",
    );
  } catch (err) {
    setStatus(`Run error: ${err.message}`, "error");
  } finally {
    runBtn.disabled = false;
    stepBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runEpisode);
stepBtn.addEventListener("click", stepOnce);
resetBtn.addEventListener("click", resetEnv);
taskSel.addEventListener("change", resetEnv);

if (scrollDetailsBtn) {
  scrollDetailsBtn.addEventListener("click", () => {
    const target = document.querySelector(".log-panel");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

(async function init() {
  await fetchPolicies();
  await resetEnv();
})();
