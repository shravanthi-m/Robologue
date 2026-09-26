"""Budgeted OpenRouter JSON calls; reservations survive unknown charges."""

import base64
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from .budget import BudgetExceeded
from .store import SQLiteStore
from .perception.video import json_key

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
CLIENT_VERSION = "bounded-json-v1"
MAX_TEXT_BYTES = 16000
MAX_IMAGE_TOKENS = 4096  # conservative allowance/image at <=768x768
CALL_RESERVATION = 0.04
PROMPT_PRICE_CEILING = 0.5  # USD per million tokens
COMPLETION_PRICE_CEILING = 3.0


class CallBudget:
    def __init__(self, path, *, max_usd=10.0, max_calls=250):
        if (
            isinstance(max_usd, bool)
            or not math.isfinite(max_usd)
            or not 0 < max_usd <= 10
        ):
            raise ValueError("Budget must be >0 and <= the authorized $10 cap")
        if type(max_calls) is not int or not 1 <= max_calls <= 250:
            raise ValueError("Model call cap must be 1-250")
        self.store = SQLiteStore(path)
        self.session = "model-budget"
        self.config = {
            "max_usd": float(max_usd),
            "max_calls": max_calls,
            "reservation": CALL_RESERVATION,
        }
        state = self.store.load(self.session)
        if state is None:
            state = {"revision": 0, "config": self.config, "attempts": []}
            self.store.compare_and_swap(self.session, state, None)
        elif state["config"] != self.config:
            raise ValueError("Resume must preserve the model budget")

    def _save(self, state, revision):
        state["revision"] = revision + 1
        self.store.compare_and_swap(self.session, state, revision)

    def reserve(self, request_key):
        state = self.store.load(self.session)
        if state.get("pricing_violation"):
            raise BudgetExceeded("Provider pricing violation locked this budget")
        spent = sum(a.get("accounted_usd", CALL_RESERVATION) for a in state["attempts"])
        if (
            len(state["attempts"]) >= self.config["max_calls"]
            or spent + CALL_RESERVATION > self.config["max_usd"] + 1e-9
        ):
            raise BudgetExceeded(
                "Persistent model budget exhausted; no request dispatched"
            )
        identity = uuid.uuid4().hex
        revision = state["revision"]
        state["attempts"].append(
            {
                "attempt_id": identity,
                "request_key": request_key,
                "status": "reserved",
                "accounted_usd": CALL_RESERVATION,
            }
        )
        self._save(state, revision)
        return identity

    def finish(self, identity, usage=None, *, status="complete"):
        state = self.store.load(self.session)
        entry = next(a for a in state["attempts"] if a["attempt_id"] == identity)
        revision = state["revision"]
        entry["status"] = status
        usage = usage or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage and type(usage[key]) is int and usage[key] >= 0:
                entry[key] = usage[key]
        cost = usage.get("cost")
        if (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
        ):
            entry["accounted_usd"] = cost
            entry["provider_reported_cost"] = True
            if cost > CALL_RESERVATION:
                state["pricing_violation"] = True
        else:
            entry["provider_reported_cost"] = False
        self._save(state, revision)
        if state.get("pricing_violation"):
            raise BudgetExceeded(
                "Provider exceeded reserved cost; stop and review pricing"
            )

    def summary(self):
        state = self.store.load(self.session)
        attempts = state["attempts"]
        return {
            "max_usd": self.config["max_usd"],
            "max_calls": self.config["max_calls"],
            "dispatched_calls": len(attempts),
            "accounted_usd": round(sum(a["accounted_usd"] for a in attempts), 6),
            "reported_cost_usd": round(
                sum(
                    a["accounted_usd"]
                    for a in attempts
                    if a.get("provider_reported_cost")
                ),
                6,
            ),
            "uncertain_charge_calls": sum(
                not a.get("provider_reported_cost") for a in attempts
            ),
            "failed_calls": sum(a["status"] == "failed" for a in attempts),
        }

    def close(self):
        self.store.close()


def image_part(path):
    from PIL import Image

    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
        image.thumbnail((768, 768))
        data = io.BytesIO()
        image.save(data, format="JPEG", quality=85)
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,"
            + base64.b64encode(data.getvalue()).decode()
        },
    }


class OpenRouterClient:
    def __init__(self, budget=None, *, api_key=None, timeout=60):
        self.budget = budget
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

    def chat(self, model, system, context, images=(), *, max_tokens=2000):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required; no request made")
        if model != DEFAULT_MODEL:
            raise ValueError(
                "Only the verified-price Gemini 2.5 Flash model is enabled for this budgeted release"
            )
        text = json.dumps(context, sort_keys=True, allow_nan=False)
        if (
            len((system + text).encode()) > MAX_TEXT_BYTES
            or len(images) > 3
            or not 1 <= max_tokens <= 2000
        ):
            raise ValueError("Request exceeds bounded text/image/output limits")
        content = [{"type": "text", "text": text}] + [image_part(p) for p in images]
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_object"},
            "provider": {
                "require_parameters": True,
                "max_price": {
                    "prompt": PROMPT_PRICE_CEILING,
                    "completion": COMPLETION_PRICE_CEILING,
                },
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        request_key = json_key(payload)
        permit = self.budget.reserve(request_key) if self.budget else None
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError("Provider response must be an object")
            if result.get("error"):
                raise ValueError("Provider returned an error")
            usage = result.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("Provider usage must be an object")
            if self.budget:
                self.budget.finish(permit, usage)
            choice = result["choices"][0]
            if choice.get("finish_reason") not in ("stop", None):
                raise ValueError("Model response was incomplete")
            value = json.loads(choice["message"]["content"])
            return value, {
                "provider": "openrouter",
                "model_returned": result.get("model", model),
                "response_id": result.get("id"),
                "usage": usage,
                "budget_attempt_id": permit,
                "latency_seconds": round(time.monotonic() - started, 3),
            }
        except urllib.error.HTTPError as exc:
            if self.budget:
                self.budget.finish(permit, status="failed")
            raise ValueError(
                f"OpenRouter HTTP {exc.code}; request failed, charge reservation retained"
            ) from None
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            ValueError,
        ):
            # A response may have settled already; do not replace known cost.
            if self.budget:
                state = self.budget.store.load(self.budget.session)
                entry = next(a for a in state["attempts"] if a["attempt_id"] == permit)
                if entry["status"] == "reserved":
                    self.budget.finish(permit, status="failed")
            raise ValueError(
                "Provider request failed or returned invalid JSON; no automatic retry"
            ) from None
