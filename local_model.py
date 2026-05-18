"""CPU-optimized local LoRA backend for DesignGym.

Loads Qwen2.5-0.5B-Instruct + a PEFT LoRA adapter on CPU and exposes
an OpenAI-shaped .chat.completions.create() interface so inference.py
needs minimal changes.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTERS: Dict[str, Optional[str]] = {
    "sft": "yashvyasop/designgym2-sft-qwen05-lora",
    "grpo": "yashvyasop/designgym2-grpo-qwen05-lora",
    "smoke": "yashvyasop/designgym2-grpo-qwen05-lora-smoke",
    "base": None,
}


@dataclass
class _Msg:
    role: str = "assistant"
    content: str = ""


@dataclass
class _Choice:
    index: int = 0
    message: _Msg = field(default_factory=_Msg)
    finish_reason: str = "stop"


@dataclass
class _Completion:
    choices: List[_Choice] = field(default_factory=list)
    model: str = ""
    backend: str = ""


class _Chat:
    def __init__(self, parent: "LocalLoRAClient"):
        self._parent = parent

    @property
    def completions(self):
        return self

    def create(self, *, model: str = "", messages: List[Dict[str, str]] = None,
               temperature: float = 0.0, max_tokens: int = 24, **_kw) -> _Completion:
        text = self._parent._generate(
            messages=messages or [],
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        comp = _Completion(
            choices=[_Choice(message=_Msg(content=text))],
            model=model or self._parent.model_id,
            backend=self._parent._backend_label(),
        )
        return comp


class LocalLoRAClient:
    _instance_lock = threading.Lock()

    def __init__(self, adapter_key: str = "sft"):
        if adapter_key not in ADAPTERS:
            raise ValueError(f"Unknown adapter key {adapter_key!r}; valid: {list(ADAPTERS)}")
        self.adapter_key: str = adapter_key
        self.adapter_id: Optional[str] = ADAPTERS[adapter_key]
        self.model_id: str = f"{BASE_MODEL}+{adapter_key}" if self.adapter_id else BASE_MODEL
        self._model = None
        self._tok = None
        self._device: str = "cpu"
        self._dtype = None
        self._ready: bool = False
        self._loading: bool = False
        self._load_error: Optional[str] = None
        self._load_seconds: Optional[float] = None

    def _backend_label(self) -> str:
        return "local-lora" if self.adapter_id else "local-base"

    @staticmethod
    def _pick_device_and_dtype():
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._instance_lock:
            if self._ready:
                return
            if self._loading:
                return
            self._loading = True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._device == "cpu":
                try:
                    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
                except Exception:
                    pass

            device, dtype = self._pick_device_and_dtype()
            self._device = device
            self._dtype = dtype

            t0 = time.time()
            print(f"[local_model] loading tokenizer {BASE_MODEL} ...", flush=True)
            self._tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

            print(f"[local_model] loading base model {BASE_MODEL} ({dtype}, {device}) ...", flush=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            if self.adapter_id:
                from peft import PeftModel
                print(f"[local_model] applying LoRA adapter {self.adapter_id} ...", flush=True)
                self._model = PeftModel.from_pretrained(self._model, self.adapter_id)

            self._model = self._model.to(device)
            self._model.eval()
            self._load_seconds = round(time.time() - t0, 2)
            self._ready = True
            self._loading = False
            print(f"[local_model] ready in {self._load_seconds}s  device={device}  backend={self._backend_label()}", flush=True)

        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._loading = False
            print(f"[local_model] LOAD FAILED: {self._load_error}", flush=True)
            raise

    def _generate(self, messages: List[Dict[str, str]], temperature: float = 0.0,
                  max_new_tokens: int = 24) -> str:
        import torch

        self._ensure_loaded()
        assert self._tok is not None and self._model is not None

        prompt_text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tok.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 50

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][input_len:]
        return self._tok.decode(new_ids, skip_special_tokens=True).strip()

    @property
    def chat(self) -> _Chat:
        return _Chat(self)

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self._backend_label(),
            "base_model": BASE_MODEL,
            "adapter_key": self.adapter_key,
            "adapter_id": self.adapter_id,
            "device": self._device,
            "dtype": str(self._dtype) if self._dtype else None,
            "ready": self._ready,
            "loading": self._loading,
            "load_seconds": self._load_seconds,
            "load_error": self._load_error,
        }


_GLOBAL_CLIENT: Optional[LocalLoRAClient] = None
_GLOBAL_LOCK = threading.Lock()


def get_client(adapter_key: Optional[str] = None) -> LocalLoRAClient:
    global _GLOBAL_CLIENT
    with _GLOBAL_LOCK:
        if adapter_key is None:
            if _GLOBAL_CLIENT is not None:
                return _GLOBAL_CLIENT
            adapter_key = os.getenv("DESIGNGYM_ADAPTER", "sft")
        if adapter_key not in ADAPTERS:
            raise ValueError(f"Unknown adapter key {adapter_key!r}; valid: {list(ADAPTERS)}")
        if _GLOBAL_CLIENT is None or _GLOBAL_CLIENT.adapter_key != adapter_key:
            _GLOBAL_CLIENT = LocalLoRAClient(adapter_key=adapter_key)
        return _GLOBAL_CLIENT


def warm_up_async(adapter_key: Optional[str] = None) -> None:
    client = get_client(adapter_key)
    if client._ready or client._loading:
        return
    t = threading.Thread(target=client._ensure_loaded, daemon=True)
    t.start()


def describe_client(client) -> Dict[str, Any]:
    if client is None:
        return {"backend": "none", "ready": False}
    if isinstance(client, LocalLoRAClient):
        return client.describe()
    return {
        "backend": "router",
        "base_model": BASE_MODEL,
        "adapter_key": None,
        "adapter_id": None,
        "device": "remote",
        "ready": True,
        "loading": False,
        "load_seconds": None,
        "load_error": None,
    }
