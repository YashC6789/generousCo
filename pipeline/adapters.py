"""Step 2: system adapters, the swap point.

Every adapter has the same shape: `complete(messages, run)` takes a list of chat
messages and returns a Completion with the reply text and run details. Switching
to Generous means adding a config entry, not changing code.
"""

import csv
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Completion:
    text: str
    meta: dict = field(default_factory=dict)


class OpenAIChatAdapter:
    """OpenAI-style chat endpoint: Ollama locally, or a hosted/Generous API."""

    def __init__(self, cfg, defaults, client=None):
        self.name = cfg["name"]
        self.model = cfg["model"]
        self.base_url = cfg["base_url"]
        self.temperature = cfg.get("temperature", defaults.get("temperature"))
        self.max_tokens = cfg.get("max_tokens", defaults.get("max_tokens"))
        self.base_seed = cfg.get("base_seed", defaults.get("base_seed", 0))
        self.system_prompt = cfg.get("system_prompt", defaults.get("system_prompt"))
        key_env = cfg.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            raise RuntimeError(f"{self.name}: environment variable {key_env} is not set")
        if client is None:
            from openai import OpenAI

            # Ollama ignores the key, but the client requires a non-empty one.
            client = OpenAI(base_url=self.base_url, api_key=os.environ.get(key_env) if key_env else "ollama")
        self.client = client
        self._version = None
        self._context = None

    def seed_for(self, run):
        return self.base_seed + run

    def build_messages(self, opener):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": opener})
        return messages

    def model_version(self):
        """Best-effort model version: the Ollama digest when the server is Ollama."""
        if self._version is None:
            self._version = _ollama_digest(self.base_url, self.model) or self.model
        return self._version

    def complete(self, messages, run, scenario_id=None):
        settings = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed_for(run),
        }
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **settings)
        choice = resp.choices[0]
        if not (choice.message.content or "").strip():
            # e.g. a reasoning model that spent the whole token budget before answering
            raise RuntimeError(f"empty reply (finish_reason={choice.finish_reason}); not saved")
        usage = getattr(resp, "usage", None)
        return Completion(
            text=choice.message.content or "",
            meta={
                "source": "api",
                "model": self.model,
                "base_url": self.base_url,
                "model_version": self.model_version(),
                "served_model": getattr(resp, "model", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                **settings,
            },
        )

    def context_length(self):
        """The server's context window in tokens, when the server is Ollama (else None)."""
        if self._context is None:
            self._context = _ollama_context_length(self.base_url, self.model)
        return self._context


class PastedCSVAdapter:
    """Responses captured by hand (consumer apps, the human-coach reference).

    The CSV follows data/pasted/TEMPLATE.csv. Rows are matched on scenario_id and
    run, and the pasted opener must match the scenario file so a mis-filed reply
    cannot slip in.
    """

    COLUMNS = ["system", "date", "app_version", "scenario_id", "run", "opener", "response"]

    def __init__(self, cfg, project_root):
        self.name = cfg["name"]
        self.path = Path(project_root) / cfg["path"]
        self.rows = {}
        if not self.path.exists():
            return
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = [c for c in self.COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{self.path}: missing columns {missing}")
            for row in reader:
                if not row["response"].strip():
                    continue
                key = (str(row["scenario_id"]).strip(), int(row["run"]))
                if key in self.rows:
                    raise ValueError(f"{self.path}: duplicate row for scenario {key[0]} run {key[1]}")
                self.rows[key] = row

    @property
    def available(self):
        return self.path.exists()

    def build_messages(self, opener):
        return [{"role": "user", "content": opener}]

    def model_version(self):
        versions = sorted({r["app_version"] or "unknown" for r in self.rows.values()})
        return ";".join(versions) or "unknown"

    def complete(self, messages, run, scenario_id=None):
        opener = messages[-1]["content"]
        row = self.rows.get((scenario_id, run))
        if row is None:
            return None
        if _norm(row["opener"]) != _norm(opener):
            raise ValueError(
                f"{self.path}: opener for scenario {scenario_id} run {run} does not match "
                f"data/scenarios.yaml"
            )
        return Completion(
            text=row["response"],
            meta={
                "source": "pasted",
                "model": row["system"] or self.name,
                "model_version": row["app_version"] or "unknown",
                "captured_date": row["date"],
                "path": str(self.path),
            },
        )


def make_adapter(cfg, defaults, project_root, client=None):
    kind = cfg.get("adapter")
    if kind == "openai":
        return OpenAIChatAdapter(cfg, defaults, client=client)
    if kind == "pasted":
        return PastedCSVAdapter(cfg, project_root)
    raise ValueError(f"{cfg.get('name')}: unknown adapter {kind!r}")


def _norm(text):
    return " ".join((text or "").split())


def _ollama_digest(base_url, model):
    """Ollama's OpenAI endpoint does not report a digest; its native /api/tags does."""
    m = _ollama_model(base_url, model, "tags")
    return f"{model}@{m.get('digest', '')[:12]}" if m else None


def _ollama_context_length(base_url, model):
    """Context window of the loaded model, from Ollama's /api/ps (the model must be loaded)."""
    m = _ollama_model(base_url, model, "ps")
    return m.get("context_length") if m else None


def _ollama_model(base_url, model, endpoint):
    parsed = urlparse(base_url)
    if parsed.port != 11434:
        return None
    try:
        url = f"{parsed.scheme}://{parsed.netloc}/api/{endpoint}"
        with urllib.request.urlopen(url, timeout=5) as r:
            listing = json.load(r)
    except Exception:
        return None
    return next((m for m in listing.get("models", []) if model in (m.get("name"), m.get("model"))), None)
