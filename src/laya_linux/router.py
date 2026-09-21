# Derived from Laya via Laya-MLX (Apache-2.0); see NOTICE. Modified for laya-linux.
"""Route a request to the locally configured Laya checkpoint best suited to it.

Adapted for laya-linux's local-first model: instead of downloading named hub
repositories, the user maps checkpoint names to local model-package
directories. Nothing is ever downloaded; a configured value that is not a
local directory fails with a clear, offline error.

The upstream benchmark rationale for script-based routing carries over
unchanged: the English checkpoint collapses on non-Latin scripts (near-random
accuracy while reporting high confidence), so script detection is the primary
routing signal. ``typed-decisions`` is never selected automatically unless you
opt in with ``auto_task_detection=True`` or pass ``task="typed_decisions"``.

Precedence: explicit ``model`` > explicit ``task`` > detected workflow
(opt-in) > explicit ``lang`` > detected script/language > default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .lang import analyse

# Canonical checkpoint roles. Values are provided by the user at construction
# time; there are no defaults because there is no hub access.
MODEL_NAMES = ("english", "multilingual", "typed-decisions")
DEFAULT_MODELS: dict[str, str | None] = {name: None for name in MODEL_NAMES}

# Aliases people are likely to type.
_ALIASES = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {
        "discrepancy_severity",
        "disposition",
        "duplicate",
        "matches_order",
        "urgency",
    },
    "security_incidents": {
        "credential_compromise",
        "disposition",
        "severity",
        "true_positive",
        "urgency",
    },
}


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a dict so it serialises straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in MODEL_NAMES:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(MODEL_NAMES), sorted(_ALIASES))
        )
    return key


def match_typed_decisions_workflow(questions: dict[str, Any]) -> str | None:
    """Name of the typed-decisions workflow whose question ids these are, else None.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


class Router:
    """Lazily loads locally configured checkpoints and routes each request.

        from laya_linux import Router

        r = Router(models={
            "english": "/opt/laya/models/laya",
            "multilingual": "/opt/laya/models/laya-multilingual",
        })
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)  # -> multilingual
        r.predict({"message": "I was charged twice"}, questions)                # -> english
        r.predict(state, questions, model="typed-decisions")                    # explicit

    ``max_loaded`` caps how many checkpoints stay resident (least-recently-used
    is evicted), because each can hold hundreds of millions of parameters.
    For a server, preload instead: a cold load costs seconds while detection
    costs microseconds, so alternating languages at ``max_loaded=1`` reloads on
    every request.

        r = Router(models=..., preload=True)          # all configured models resident
        r.preload(["english", "multilingual"])        # or just the two you serve
    """

    def __init__(
        self,
        models: dict[str, str | Path] | None = None,
        device: str | None = None,
        dtype: str = "auto",
        max_loaded: int = 1,
        default: str = "english",
        auto_task_detection: bool = False,
        preload: bool = False,
    ):
        self.models: dict[str, str | None] = dict(DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): str(v) for k, v in models.items()})
        self.dtype = dtype
        self.device = device
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self._agents: dict[str, Any] = {}
        self._order: list[str] = []  # least-recently-used first
        if preload:
            self.preload()

    # ------------------------------------------------------------------ loading
    def load(self, name: str):
        """Return the Agent for ``name``, building it on first use."""
        key = normalise_name(name)
        if key in self._agents:
            self._touch(key)
            return self._agents[key]
        from .agent import Agent

        path = self.models.get(key)
        if not path:
            raise ValueError(
                f"No local model configured for {key!r}; pass models={{...}} with local "
                f"model-package directories when creating the Router"
            )
        agent = Agent(path, device=self.device, dtype=self.dtype)
        self._agents[key] = agent
        self._order.append(key)
        self._evict()
        return agent

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def _evict(self) -> None:
        while len(self._order) > self.max_loaded:
            victim = self._order.pop(0)
            self._agents.pop(victim, None)
        if len(self._order) < len(self._agents):  # keep the two views consistent
            for k in list(self._agents):
                if k not in self._order:
                    self._agents.pop(k, None)

    def attach(self, name: str, agent: Any):
        """Register an already-built Agent under ``name`` instead of loading a second copy."""
        key = normalise_name(name)
        self._agents[key] = agent
        self._touch(key)
        self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: list[str] | None = None):
        """Build configured checkpoints up front so no request pays a model load."""
        names = [normalise_name(n) for n in (names or list(self.models))]
        self.max_loaded = max(self.max_loaded, len(names), len(self._agents))
        for n in names:
            if n not in self._agents:  # an attached agent is already built
                self.load(n)
        return self

    def unload(self, name: str | None = None) -> None:
        """Free one model, or all of them."""
        if name is None:
            self._agents.clear()
            self._order.clear()
        else:
            key = normalise_name(name)
            self._agents.pop(key, None)
            if key in self._order:
                self._order.remove(key)

    @property
    def loaded(self) -> list[str]:
        return list(self._order)

    # ------------------------------------------------------------------ routing
    def route(
        self,
        state: str | dict | list | None,
        questions: dict[str, Any] | None = None,
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything."""
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(
                model=key, path=self.models.get(key), reason="explicit model=%r" % model,
                detection=None, workflow=None,
            )

        if task is not None:
            key = normalise_name(
                "typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task
            )
            return RouteDecision(
                model=key, path=self.models.get(key), reason="explicit task=%r" % task,
                detection=None, workflow=None,
            )

        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(
                model="typed-decisions",
                path=self.models.get("typed-decisions"),
                reason="question ids match the %r typed-decisions workflow" % workflow,
                detection=None,
                workflow=workflow,
            )

        if lang is not None:
            key = (
                "english" if str(lang).lower().split("-")[0] in ("en", "eng", "english") else "multilingual"
            )
            return RouteDecision(
                model=key, path=self.models.get(key), reason="explicit lang=%r" % lang,
                detection=None, workflow=workflow,
            )

        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = (
                "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                % (det["script"], 100 * float(det["non_latin_fraction"]))
            )
        elif not det["is_english"]:
            key = "multilingual"
            reason = "Latin script but language looks like %r, not English" % det["language"]
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(
            model=key, path=self.models.get(key), reason=reason, detection=det, workflow=workflow
        )

    # ------------------------------------------------------------------ running
    def predict(
        self,
        state: str | dict | list,
        questions: dict[str, Any],
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        The result is the usual ``system_one`` payload plus a ``routing`` key recording the
        decision.
        """
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        agent = self.load(decision["model"])
        result = agent.system_one(state, questions)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (
            self.loaded, self.max_loaded, self.default,
        )
