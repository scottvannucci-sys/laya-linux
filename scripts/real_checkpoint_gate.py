"""Real-checkpoint gate: run the actual laya-typed-decisions checkpoint (manual script)."""
import sys, time

sys.path.insert(0, "C:/Users/scott/Sources/laya-linux/src")
import torch  # noqa: F401

PKG = "C:/Users/scott/Sources/laya-linux/models/laya-typed-decisions"

t0 = time.time()
from laya_linux import Agent

agent = Agent(PKG, device="cpu")
print(f"Agent loaded in {time.time()-t0:.1f}s | device={agent.device} dtype={agent.dtype}")
print("strict load of checkpoint tensors: OK (no name mapping)")

questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want in `message`?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "billing_question": "a question about an invoice, plan or payment method",
            "information": "general information, pricing or how-to",
            "cancellation": "wants to cancel or downgrade",
            "other": "none of the other options fits",
        },
    },
    "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound in `message`?",
        "criteria": [
            "calm and neutral",
            "concerned but civil",
            "clearly annoyed",
            "very angry or using strong language",
        ],
    },
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "churn_risk": {
        "type": "noul",
        "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?",
    },
}
state = {
    "message": "I was charged twice on my last invoice and I need this fixed today "
               "or I am cancelling my account."
}
t0 = time.time()
result = agent.predict(state, questions)
dt = time.time() - t0
usage = result["usage"]
print(f"predict: {dt*1000:.0f} ms | usage: {usage}")
for qid, a in result["answers"].items():
    if a["type"] == "choice":
        act = a["action"]["act_probability"]
        print(f"  {qid}: {a['choice']} {a['probabilities']} conf={a['confidence']} act={act}")
    elif a["type"] == "score":
        print(f"  {qid}: score={a['score']} {a['probabilities']} conf={a['confidence']}")
    else:
        act = a["action"]["act_probability"]
        print(f"  {qid}: P(true)={a['noul']} conf={a['confidence']} act={act}")

# determinism on the real checkpoint
r2 = agent.predict(state, questions)
assert r2 == result, "non-deterministic on real checkpoint!"
print("DETERMINISM OK (real checkpoint)")
print("REAL-CHECKPOINT AGENT RUN OK")
