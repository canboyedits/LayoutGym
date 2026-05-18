#!/usr/bin/env python3
"""Smoke-test that a DesignGym LoRA adapter loads correctly and produces
different (better) outputs than the base model on a representative prompt.

Run:  python verify_finetuned.py --adapter sft --samples 3 --compare-base
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTERS = {
    "sft": "yashvyasop/designgym2-sft-qwen05-lora",
    "grpo": "yashvyasop/designgym2-grpo-qwen05-lora",
    "smoke": "yashvyasop/designgym2-grpo-qwen05-lora-smoke",
    "base": None,
}

SYSTEM_PROMPT = (
    "You are choosing one action for a long-horizon layout design environment.\n\n"
    "Return exactly one minified JSON object:\n"
    '{"choice": <integer>}\n\n'
    "Rules:\n"
    "- Choose exactly one candidate index.\n"
    "- Do not explain.\n"
    "- Do not output markdown.\n"
    "- Prefer actions that satisfy the design brief.\n"
    "- Prefer actions that match the current design phase.\n"
    "- Improve weak metrics without ignoring the brief.\n"
    "- Avoid repeating low-gain actions.\n"
    "- Do not choose finalize unless the layout score and instruction score are both high late in the episode."
)

USER_PROMPT = (
    "Task: poster_basic_v1\n"
    "Step: 3\n"
    "Max steps: 8\n"
    "Phase: improve\n"
    'Brief: {"audience": "general", "style": "modern", "type": "poster"}\n'
    "Current score: 0.5200\n"
    "Best score so far: 0.5200\n"
    'Worst metrics: ["hierarchy", "alignment"]\n'
    'Metrics: {"alignment": 0.42, "hierarchy": 0.38, "occupancy": 0.71, "spacing": 0.60}\n'
    "Layout summary: title@(0.1,0.1,0.8,0.08) subtitle@(0.1,0.2,0.6,0.05) hero_image@(0.1,0.3,0.7,0.4) cta@(0.3,0.8,0.3,0.06)\n"
    "Recent rewards: [0.02, 0.01]\n"
    "Finalize allowed: false\n\n"
    "Candidate actions:\n"
    '0: {"action_type":"promote","element_id":"title","strength":0.04} [allowed]\n'
    '1: {"action_type":"resize","dh":0.02,"dw":0.03,"element_id":"hero_image"} [allowed]\n'
    '2: {"action_type":"align","axis":"x","element_ids":["title","subtitle"],"mode":"left"} [allowed]\n'
    '3: {"action_type":"reflow_group","group_id":"headline","pattern":"stack"} [allowed]\n'
    '4: {"action_type":"anchor_to_region","element_id":"cta","mode":"center","region_id":"safe_lower_right"} [allowed]\n'
    '5: {"action_type":"finalize"} [blocked]\n\n'
    'Return exactly one JSON object:\n'
    '{"choice": N}'
)


def pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(adapter_key: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = pick_device_and_dtype()
    adapter_id = ADAPTERS.get(adapter_key)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    active_adapter = None
    trainable_params = 0
    total_params = sum(p.numel() for p in model.parameters())

    if adapter_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_id)
        active_adapter = getattr(model, "active_adapter", None)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

    model = model.to(device)
    model.eval()
    load_seconds = round(time.time() - t0, 2)

    info = {
        "device": device,
        "dtype": str(dtype),
        "base_model": BASE_MODEL,
        "adapter_key": adapter_key,
        "adapter_id": adapter_id,
        "load_seconds": load_seconds,
        "active_adapter": str(active_adapter) if active_adapter is not None else None,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    return model, tok, info


def generate(model, tok, system: str, user: str, max_new_tokens: int = 32,
             temperature: float = 0.0) -> str:
    import torch

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tok.decode(out[0][input_len:], skip_special_tokens=True).strip()


def parse_choice(text: str):
    try:
        obj = json.loads(text)
        return int(obj["choice"])
    except Exception:
        pass
    m = re.search(r'\{\s*"choice"\s*:\s*(\d+)\s*\}', text)
    if m:
        return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Verify DesignGym LoRA adapter")
    parser.add_argument("--adapter", choices=list(ADAPTERS), default="sft")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--compare-base", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  DesignGym LoRA Verification  adapter={args.adapter}")
    print("=" * 60)

    # --- [1] Load ---
    print(f"\n[1] Loading {args.adapter} model ...")
    model, tok, info = load_model(args.adapter)
    print(f"    device        = {info['device']}")
    print(f"    dtype         = {info['dtype']}")
    print(f"    adapter_id    = {info['adapter_id']}")
    print(f"    load_seconds  = {info['load_seconds']}s")
    print(f"    active_adapter= {info['active_adapter']}")
    print(f"    trainable     = {info['trainable_params']:,}")
    print(f"    total_params  = {info['total_params']:,}")

    # --- [2] Adapter check ---
    print(f"\n[2] Adapter check")
    if args.adapter != "base":
        if info["active_adapter"] is None:
            print("    FAIL  active_adapter is None  the adapter did not load")
            sys.exit(2)
        pct = (info["trainable_params"] / info["total_params"] * 100) if info["total_params"] else 0
        print(f"    PASS  active_adapter = {info['active_adapter']!r}")
        print(f"    LoRA trainable: {pct:.2f}%")
    else:
        print("    SKIP  base model run, no adapter expected")

    # --- [3] Single greedy generation ---
    print(f"\n[3] Single greedy generation")
    raw = generate(model, tok, SYSTEM_PROMPT, USER_PROMPT, max_new_tokens=32, temperature=0.0)
    choice = parse_choice(raw)
    valid = choice is not None
    print(f"    raw output : {raw!r}")
    print(f"    parsed     : choice={choice}")
    print(f"    valid_json : {valid}")

    # --- [4] Compare base ---
    if args.compare_base and args.adapter != "base":
        print(f"\n[4] Comparing {args.adapter} vs base")
        print("    Loading base model ...")
        base_model, base_tok, base_info = load_model("base")
        base_raw = generate(base_model, base_tok, SYSTEM_PROMPT, USER_PROMPT,
                            max_new_tokens=32, temperature=0.0)
        base_choice = parse_choice(base_raw)

        print(f"    {args.adapter:5s} raw: {raw!r}")
        print(f"    base  raw: {base_raw!r}")
        differ = raw != base_raw
        print(f"    DIFFERENT: {differ}")
        if not differ:
            print("    WARNING  adapter and base produced identical output")

        del base_model, base_tok
    else:
        print(f"\n[4] Base comparison skipped")

    # --- [5] Sampling valid_json_rate ---
    print(f"\n[5] Sampling {args.samples} generations at temperature=0.3")
    valid_count = 0
    for i in range(args.samples):
        sample = generate(model, tok, SYSTEM_PROMPT, USER_PROMPT,
                          max_new_tokens=32, temperature=0.3)
        c = parse_choice(sample)
        ok = c is not None
        if ok:
            valid_count += 1
        marker = "ok" if ok else "FAIL"
        print(f"    [{i+1}] {marker}  {sample!r}")

    rate = valid_count / args.samples if args.samples else 0
    print(f"\n    valid_json_rate = {valid_count}/{args.samples} = {rate:.0%}")
    if rate < 0.6 and args.adapter != "base":
        print("    WARNING  valid_json_rate < 60% on a fine-tuned adapter")

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
