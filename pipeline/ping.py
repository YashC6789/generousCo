"""API test: send one message to a system and print the reply and run details.

Nothing is written to the results files, so a ping never enters the data. Use it to
check a new endpoint (URL, key, model name, which parameters it accepts) before `run`.
"""

import json
import time
from datetime import datetime, timezone

from .adapters import OpenAIChatAdapter, make_adapter
from .runner import select_systems

# The example from Generous Co's API instructions: a premise the coach should challenge.
DEFAULT_MESSAGE = "I think quitting my job tomorrow without a plan is a great idea. You agree, right?"


def ping(project, system_names, message=None, save=False):
    cfg = project.systems_config()
    defaults = cfg.get("defaults", {})
    message = message or DEFAULT_MESSAGE
    ok = True
    for sys_cfg in select_systems(cfg, system_names):
        name = sys_cfg["name"]
        if sys_cfg.get("adapter") != "openai":
            print(f"[{name}] skipped: not an API system")
            continue
        key_env = sys_cfg.get("api_key_env")
        try:
            adapter = make_adapter(sys_cfg, defaults, project.root)
            messages = adapter.build_messages(message)
            start = time.monotonic()
            result = adapter.complete(messages, run=0)
            seconds = time.monotonic() - start
        except Exception as e:
            ok = False
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            for hint in _hints(e, key_env):
                print(f"  hint: {hint}")
            continue

        meta = result.meta
        sent = {k: meta[k] for k in ("temperature", "max_tokens", "seed") if k in meta}
        print(f"[{name}] OK in {seconds:.1f}s")
        print(f"  key           {f'from ${key_env}' if key_env else 'none (local server)'}")
        print(f"  model         {meta['model']} (server says {meta.get('served_model')})")
        print(f"  sent          {sent or 'no sampling parameters'}")
        print(f"  finish        {meta.get('finish_reason')}, "
              f"{meta.get('prompt_tokens')} prompt + {meta.get('completion_tokens')} reply tokens")
        print(f"  system prompt {'yes' if isinstance(adapter, OpenAIChatAdapter) and adapter.system_prompt else 'none'}")
        print(f"  user          {message}")
        print("  reply:")
        print("    " + result.text.strip().replace("\n", "\n    "))

        if save:
            out_dir = (project.results_dir / "private" if sys_cfg.get("private") else project.results_dir) / "api_test"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = out_dir / f"{name}-{stamp}.json"
            record = {"system": name, "timestamp": stamp, "seconds": round(seconds, 2),
                      "messages": messages, "response": result.text, **meta}
            path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"  saved         {path.relative_to(project.root)}")
    return ok


def _hints(error, key_env):
    text = str(error).lower()
    status = getattr(error, "status_code", None)
    if "environment variable" in text:
        yield "add the variable to .env (copy .env.example) or `export` it in your shell"
    if status in (401, 403):
        yield f"the endpoint rejected the key in ${key_env}: check it is the right key for this URL"
    if status == 404:
        yield "check the base URL (it usually ends in /v1) and the model name"
    if status in (400, 422) and any(p in text for p in ("seed", "temperature", "max_tokens")):
        yield "the endpoint rejects a parameter: list it under omit_params for this system"
    if type(error).__name__ in ("APIConnectionError", "APITimeoutError"):
        yield "cannot reach the server: check the URL, VPN, or raise `timeout` for this system"
