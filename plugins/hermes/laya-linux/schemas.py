"""Tool schemas — what the LLM sees.

The descriptions carry the calling discipline from
docs/harness-integration.md: state is a JSON string (may be a plain
sentence), questions reference keys of the state, and the preset defines
the question set when the caller does not supply custom questions.
"""

LAYA_PREDICT = {
    "name": "laya_predict",
    "description": (
        "Evaluate typed decision questions against a state using a local, "
        "offline laya-linux model. Returns structured answers: a choice "
        "label, a numeric score, and calibrated boolean (noul) values with "
        "confidences. Use this instead of guessing when the user asks for a "
        "classification, triage, routing, guardrail, or scoring decision "
        "and a laya-linux server is available. The state must be a JSON "
        "object (or plain text) containing the keys the questions reference "
        "(e.g. 'message', 'body', 'prompt', 'post', 'request'). Choose a "
        "preset with laya_presets, or pass custom questions as a JSON "
        "object mapping question-id to {type: choice|score|noul, "
        "instructions, criteria}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": (
                    "The state to evaluate, as a JSON object string or "
                    "plain text. Must contain the keys the questions "
                    "reference (e.g. {\"message\": \"I was charged twice\"})."
                ),
            },
            "preset": {
                "type": "string",
                "description": (
                    "Name of a question preset: triage, email, guard, "
                    "moderation, or router. Omit when 'questions' is given."
                ),
            },
            "questions": {
                "type": "string",
                "description": (
                    "Custom question set as a JSON object string mapping "
                    "question-id to its definition. Overrides 'preset'."
                ),
            },
            "preset_args": {
                "type": "string",
                "description": (
                    "Optional JSON object string of preset arguments, e.g. "
                    "{\"categories\": {...}} for the email preset."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model alias override (server-configured).",
            },
        },
        "required": ["state"],
    },
}

LAYA_PRESETS = {
    "name": "laya_presets",
    "description": (
        "List the available laya-linux question presets and the exact "
        "questions each one asks, including expected state keys. Call this "
        "before laya_predict when unsure which preset fits the task or "
        "which keys the state must contain."
    ),
    "parameters": {"type": "object", "properties": {}},
}

LAYA_SERVER_STATUS = {
    "name": "laya_server_status",
    "description": (
        "Check the laya-linux server: reachability, protocol version, "
        "loaded models, and device/dtype. Use this to diagnose a failed "
        "laya_predict call or before relying on it in a workflow."
    ),
    "parameters": {"type": "object", "properties": {}},
}
