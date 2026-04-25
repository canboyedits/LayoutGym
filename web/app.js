window.addEventListener("error", (event) => {
  document.body.innerHTML = `
    <pre style="white-space:pre-wrap;color:white;background:#111827;padding:24px;font-size:16px;">
JS ERROR:
${event.message}

File: ${event.filename}
Line: ${event.lineno}
Column: ${event.colno}
    </pre>
  `;
});

window.addEventListener("unhandledrejection", (event) => {
  document.body.innerHTML = `
    <pre style="white-space:pre-wrap;color:white;background:#111827;padding:24px;font-size:16px;">
PROMISE ERROR:
${event.reason}
    </pre>
  `;
});

let lastObservation = null;
let lastState = null;

const canvas = document.getElementById("canvas");
const briefBox = document.getElementById("brief");
const metricsBox = document.getElementById("metrics");
const historyBox = document.getElementById("history");

function unwrapObservation(payload) {
  return payload?.observation || payload?.state?.observation || payload;
}

function unwrapState(payload) {
  return payload?.state || payload?.observation?.state || payload;
}

function rectColor(role, type) {
  const key = role || type || "default";
  const colors = {
    title: "#bfdbfe",
    subtitle: "#dbeafe",
    image: "#bbf7d0",
    cta: "#fecaca",
    logo: "#fde68a",
    badge: "#ddd6fe",
    body: "#e2e8f0",
    caption: "#fef3c7",
    shape: "#ddd6fe",
    default: "#e5e7eb"
  };
  return colors[key] || colors.default;
}

function drawState(state) {
  canvas.innerHTML = "";

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", 0);
  bg.setAttribute("y", 0);
  bg.setAttribute("width", 800);
  bg.setAttribute("height", 1000);
  bg.setAttribute("fill", "#f8fafc");
  canvas.appendChild(bg);

  const elements = state?.elements || [];

  if (!elements.length) {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", 40);
    t.setAttribute("y", 60);
    t.setAttribute("fill", "#0f172a");
    t.setAttribute("font-size", "24");
    t.textContent = "No layout state yet. Click Reset.";
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
    r.setAttribute("x", x);
    r.setAttribute("y", y);
    r.setAttribute("width", w);
    r.setAttribute("height", h);
    r.setAttribute("rx", 8);
    r.setAttribute("fill", rectColor(el.role, el.type));
    r.setAttribute("stroke", "#0f172a");
    r.setAttribute("stroke-width", 2);
    canvas.appendChild(r);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x + 8);
    label.setAttribute("y", y + 22);
    label.setAttribute("fill", "#0f172a");
    label.setAttribute("font-size", "15");
    label.setAttribute("font-family", "monospace");
    label.textContent = `${el.id || "element"}`;
    canvas.appendChild(label);

    const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
    sub.setAttribute("x", x + 8);
    sub.setAttribute("y", y + 42);
    sub.setAttribute("fill", "#334155");
    sub.setAttribute("font-size", "12");
    sub.setAttribute("font-family", "monospace");
    sub.textContent = `${el.role || el.type || ""}`;
    canvas.appendChild(sub);
  }
}

async function getState() {
  const res = await fetch("/demo/state");
  const payload = await res.json();
  return payload.state || payload;
}

async function refresh() {
  const state = await getState();
  lastState = state;

  drawState(state);

  const obsLike = lastObservation || state;

  briefBox.textContent = JSON.stringify({
    task_id: state.task_id || obsLike.task_id,
    phase: state.phase || obsLike.phase,
    allowed_actions: state.allowed_actions || obsLike.allowed_actions,
    instruction_score: state.instruction_score || obsLike.instruction_score,
    brief: state.brief || obsLike.brief,
    critic_feedback: state.critic_feedback || obsLike.critic_feedback
  }, null, 2);

  metricsBox.textContent = JSON.stringify({
    score: state.current_score,
    best_score_so_far: state.best_score_so_far,
    phase_score: state.phase_score,
    reward_components: state.reward_components,
    metrics: state.metrics
  }, null, 2);

  historyBox.textContent = (state.action_history || []).join("\n");
}

async function resetEnv() {
  try {
    const task = document.getElementById("task").value;

    const res = await fetch("/demo/reset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task_id: task, seed: 0})
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Reset failed HTTP ${res.status}: ${text}`);
    }

    const payload = await res.json();
    lastObservation = payload.observation || payload;
    lastState = payload.state || null;

    await refresh();
  } catch (err) {
    document.body.innerHTML = `
      <pre style="white-space:pre-wrap;color:white;background:#111827;padding:24px;font-size:16px;">
RESET ERROR:
${err}
      </pre>
    `;
  }
}

function layoutHas(id) {
  const summary = lastObservation?.layout_summary || lastState?.layout_summary || "";
  return summary.includes(`${id}@`);
}

function simpleHeuristic(obs) {
  const worst = obs?.worst_metrics || [];
  const task = obs?.task_id || document.getElementById("task").value;
  const step = Number(obs?.step_count || 0);
  const instructionScore = Number(obs?.instruction_score || 0);

  if (step === 0) {
    if (task.includes("editorial")) {
      return {action_type: "apply_template", template_id: "editorial"};
    }
    if (task.includes("dense")) {
      return {action_type: "apply_template", template_id: "grid"};
    }
    return {action_type: "apply_template", template_id: "hero"};
  }

  if (instructionScore < 0.70) {
    if (layoutHas("cta")) {
      return {
        action_type: "anchor_to_region",
        element_id: "cta",
        region_id: "safe_lower_right",
        mode: "center"
      };
    }

    if (layoutHas("hero_image")) {
      return {
        action_type: "anchor_to_region",
        element_id: "hero_image",
        region_id: "hero_center",
        mode: "center"
      };
    }

    if (layoutHas("masthead")) {
      return {
        action_type: "anchor_to_region",
        element_id: "masthead",
        region_id: "top_band",
        mode: "center"
      };
    }
  }

  if (worst.includes("hierarchy")) {
    if (layoutHas("title")) {
      return {action_type: "promote", element_id: "title", strength: 0.04};
    }
    if (layoutHas("headline_1")) {
      return {action_type: "promote", element_id: "headline_1", strength: 0.04};
    }
  }

  if (worst.includes("spacing") || worst.includes("reading_order")) {
    if (task.includes("dense")) {
      return {action_type: "reflow_group", group_id: "support", pattern: "row"};
    }
    if (task.includes("editorial")) {
      return {action_type: "reflow_group", group_id: "stories", pattern: "stack"};
    }
    return {action_type: "reflow_group", group_id: "headline", pattern: "stack"};
  }

  if (worst.includes("alignment")) {
    if (task.includes("editorial")) {
      return {
        action_type: "align",
        element_ids: ["masthead", "headline_1", "headline_2"],
        axis: "x",
        mode: "left"
      };
    }
    if (task.includes("dense")) {
      return {
        action_type: "align",
        element_ids: ["caption_1", "caption_2"],
        axis: "y",
        mode: "top"
      };
    }
    return {
      action_type: "align",
      element_ids: ["title", "subtitle"],
      axis: "x",
      mode: "left"
    };
  }

  if (layoutHas("hero_image")) {
    return {
      action_type: "resize",
      element_id: "hero_image",
      dw: 0.02,
      dh: 0.01,
      anchor: "center"
    };
  }

  if (layoutHas("details")) {
    return {
      action_type: "resize",
      element_id: "details",
      dw: 0.02,
      dh: 0.02,
      anchor: "center"
    };
  }

  return {action_type: "finalize"};
}

async function postStep(action) {
  return await fetch("/demo/step", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({action})
  });
} 

async function stepEnv(action) {
  const res = await postStep(action);

  if (!res.ok) {
    const text = await res.text();
    console.error("Step failed", res.status, text);
    metricsBox.textContent = `Step failed: HTTP ${res.status}\n\n${text}`;
    return;
  }

  const payload = await res.json();
  lastObservation = unwrapObservation(payload);

  await refresh();
}

document.getElementById("reset").onclick = resetEnv;

document.getElementById("step").onclick = async () => {
  if (!lastObservation) {
    await resetEnv();
  }
  await stepEnv(simpleHeuristic(lastObservation));
};

document.getElementById("finalize").onclick = async () => {
  await stepEnv({action_type: "finalize"});
};

resetEnv();