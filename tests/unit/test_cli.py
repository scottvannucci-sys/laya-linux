"""CLI behavior via the public entry point (unit)."""

import json

import pytest

from laya_linux.cli import main

torch = pytest.importorskip("torch")


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "laya-linux" in capsys.readouterr().out


def test_verify_ok(capsys, tiny_pkg):
    assert main(["verify", str(tiny_pkg), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "strict match" in payload["report"]["weights"]


def test_verify_bad_path_structured_error(capsys, tmp_path):
    # verify reports failures as {"ok": false, error_code, error} on stdout
    assert main(["verify", str(tmp_path / "nope"), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_code"] in {"MODEL_NOT_FOUND", "NETWORK_DISABLED"}


def test_verify_remote_identifier_rejected(capsys):
    assert main(["verify", "convaiinnovations/laya", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_code"] == "NETWORK_DISABLED"


def test_doctor_reports_facts(capsys):
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "torch" in payload["facts"]
    assert "device_plan" in payload["facts"]
    assert payload["suggestions"] == []  # healthy environment: no suggestions


def test_doctor_human_output(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "doctor" in out and "torch" in out


def test_predict_with_preset(capsys, tiny_pkg):
    rc = main(["predict", str(tiny_pkg), "--state", "I was charged twice.", "--preset", "triage"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "laya-rl-agent"
    assert "intent" in payload["answers"]


def test_predict_with_files(capsys, tiny_pkg, tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"message": "hello"}))
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({
        "q": {"type": "noul", "instructions": "Is this a greeting?"}
    }))
    rc = main(["predict", str(tiny_pkg), "--state-file", str(state_file),
               "--questions-file", str(questions_file)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "q" in payload["answers"]


def test_predict_requires_state(capsys, tiny_pkg):
    with pytest.raises(SystemExit):
        main(["predict", str(tiny_pkg), "--preset", "triage"])


def test_benchmark_emits_json(capsys, tiny_pkg):
    rc = main(["benchmark", str(tiny_pkg), "--state", "x", "--preset", "triage",
               "--warmup", "1", "--iterations", "3"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["raw_samples_seconds"]) == 3
    assert payload["environment"]["iterations"] == 3
    assert "do not compare" in payload["note"]


def test_serve_is_phase4_placeholder(capsys):
    assert main(["serve"]) == 2
