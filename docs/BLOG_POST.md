# Teaching LLMs to Run an Incident Command Center

> *India's Biggest Mega AI Hackathon — Built on Meta OpenEnv*

**TL;DR** — I built an OpenEnv environment where a team of three specialist agents — **triage**, **investigator**, and **ops manager** — have to diagnose and resolve real-world incidents under SLA pressure, budget constraints, and executive escalation rules. I then fine-tuned a small Qwen2.5 model on rollouts from a heuristic coordinator and measured how much better it gets at running the command center.

Live environment → **[Hugging Face Space ↗](https://swapnilpatil28-multi-agent-incident-command-center.hf.space)** · [Space page](https://huggingface.co/spaces/SwapnilPatil28/Multi-Agent-Incident-Command-Center)
Source code → **[GitHub ↗](https://github.com/SwapnilPatil28/Multi-Agent-Incident-Command-Center)**
Reproducible Colab notebook → **[Open in Colab ↗](https://colab.research.google.com/drive/1vx9E5FrZZrHoRwXs2cvtom3DaI6kZ3LP?usp=sharing)**
2-min video → *coming soon*

---

## 1. The problem: incidents are multi-agent, long-horizon, and expensive to get wrong

Every real on-call rotation looks roughly like this:

1. An alert fires. Somebody on **triage** skims logs and dashboards to narrow the blast radius.
2. An **investigator** consults runbooks, past incidents, and subject-matter experts to form a hypothesis.
3. An **ops manager** decides who owns the fix, when to escalate, whether to roll back, and whether the incident needs a postmortem.

Each of those roles has different **permissions**, different information needs, and different success criteria. The hard parts aren't in any single step — they're in the **handoffs**, the **time pressure**, and the **cost of wrong guesses** (every step costs something; picking the wrong owner costs more).

Most existing RL environments for LLMs are single-agent, single-step, or turn-based games. Real enterprise work is none of those. I wanted an environment that tests an LLM's ability to:

- **Pick the right specialist for the current phase** (role-based permissions)
- **Extract signal from noisy observations** (visible signals + red-herring signals)
- **Plan over a long horizon** (3–5 incidents per episode, 20–60 steps each)
- **Balance speed vs thoroughness** (SLA budget counts down every step)
- **Write structured closure notes** (mitigation summaries, postmortem artifacts)

That's three hackathon themes in one environment: **Multi-Agent Interactions**, **Long-Horizon Planning**, and **Professional Tasks (World Modeling)**.

---

## 2. What the environment looks like

The environment runs as a standard OpenEnv FastAPI server — same Gym-style `reset / step` contract, same Pydantic observation/action schemas, same Docker image format for Hugging Face Spaces.

### Observation (partial)

```json
{
  "incident_id": "inc-cert-expiry",
  "incident_title": "mTLS cert expired — all microservices throwing 500s",
  "incident_description": "Alerting fired at 03:12 UTC ...",
  "customer_tier": "enterprise",
  "affected_users_estimate": 140000,
  "revenue_impact_usd_per_min": 4800,
  "postmortem_required": true,
  "visible_signals": ["mtls handshake errors", "5xx spike in checkout"],
  "investigation_targets": {
    "logs": ["cert-manager", "auth-service"],
    "metrics": ["dash-mesh", "dash-auth"],
    "kb": ["kb-mtls-chain", "kb-cert-rotation"]
  },
  "allowed_actors_by_action": {
    "apply_fix": ["investigator_agent"],
    "close_incident": ["ops_manager_agent"],
    ...
  },
  "budget_remaining": 18,
  "sla_minutes_remaining": 40,
  "clues_found": 2,
  "mitigation_applied": false,
  "reward_components": {"step_cost": -0.04, "clue_bonus": +0.12}
}
```

### Action space

| action_type | Typical actor | Purpose |
|---|---|---|
| `inspect_logs` / `inspect_metrics` / `consult_kb` | triage / investigator | Gather clues (reward shapes here) |
| `negotiate_handoff` | ops_manager | Route to correct owner |
| `apply_fix` | investigator | Apply mitigation (scored vs ground truth) |
| `rollback` | investigator | Revert last change |
| `escalate` | ops_manager | Engage senior staff |
| `submit_postmortem` | ops_manager | Required on tier-1 / high-revenue incidents |
| `close_incident` | ops_manager | Terminal action; final score depends on clues found + mitigation quality + postmortem + speed |

### Reward rubric (composable, not monolithic)

The reward engine emits named components at every step so judges (and training signals) can see *exactly where reward came from*:

| Component | When it fires | Sign |
|---|---|---|
| `step_cost` | Every action | − |
| `clue_bonus` | Unique log/metric/KB lookup that surfaces a real clue | + |
| `handoff_correct` / `handoff_wrong` | Ops manager routes to allowed / disallowed owner | ± |
| `mitigation_correct` / `mitigation_wrong` | Fix matches ground-truth keywords | ± |
| `closure_correct` / `closure_wrong` | Final close decision matches incident state | ± (scaled by customer tier) |
| `closure_mitigation_bonus` | Close after a correct mitigation | + |
| `speed_bonus` | Close with SLA remaining | + |
| `postmortem_bonus` / `postmortem_missing` | Postmortem filed when required | ± |
| `repeated_lookup_penalty` | Re-querying the same log/metric/KB | − |

The anti-gaming baked into this rubric: closing early with zero clues is penalized; spamming cheap `inspect_logs` racks up `repeated_lookup_penalty`; triggering `apply_fix` without investigator permissions gives `handoff_wrong`. A policy cannot shortcut its way to a high score.

---

## 3. Training: HF TRL SFT on heuristic rollouts

I first wrote a deterministic `HeuristicCoordinator` that uses the observation's `investigation_targets` and role constraints to play through the environment. On hard tasks it pulls in +5.89 reward where random scores −12.50. That gives us ~680 `(prompt, completion)` pairs of "good" behavior to imitate.

Training script: [`train_trl.py`](https://github.com/SwapnilPatil28/Multi-Agent-Incident-Command-Center/blob/main/train_trl.py). One command on Colab T4 (or [open the reproducible notebook](https://colab.research.google.com/drive/1vx9E5FrZZrHoRwXs2cvtom3DaI6kZ3LP?usp=sharing)) runs the entire pipeline:

```python
os.environ["BASE_MODEL"]        = "Qwen/Qwen2.5-1.5B-Instruct"
os.environ["EPISODES_PER_TASK"] = "8"
os.environ["TRAIN_EPOCHS"]      = "3"
os.environ["EVAL_LLM_MODELS"]   = "true"
os.environ["MAX_LLM_EVAL_STEPS"] = "120"
!python train_trl.py
```

The script:
1. Rolls out the heuristic against the live environment and collects prompts/completions.
2. Runs TRL `SFTTrainer` with a single `text` column (chat-template applied).
3. Saves the fine-tuned checkpoint to `artifacts/sft_model/`.
4. Rolls out **four** policies under identical seeds — random, heuristic, base LLM, fine-tuned LLM.
5. Writes `reward_curve.png`, `training_curve.png`, `reward_components.png`, and `summary_metrics.json`.

### Training loss

![Training curve](./artifacts/training_curve.png)

Loss drops from **~2.84** to **~0.02** over three epochs as the model learns the structured JSON action format. Mean token accuracy climbs from **~0.49 → ~0.99**.

### Reward comparison

![Reward curve with 4 policies](./artifacts/reward_curve.png)

| Task | Random | Base LLM | **Fine-tuned (SFT)** | Heuristic |
|---|---:|---:|---:|---:|
| easy | −5.96 | −2.92 | **−4.72** | −4.72 |
| medium | −11.48 | −4.00 | **−0.87** | −0.87 |
| hard | −12.50 | −4.28 | **+5.89** | +5.89 |

**Fine-tuned vs untrained base: +10.17 reward delta on hard-difficulty incidents.**

- **Random** is a flat floor on every task.
- **Base LLM** already beats random on easy because it produces well-formed-ish JSON — but it never closes a single incident, so it just racks up step-costs and SLA penalties.
- **Fine-tuned LLM** catches the heuristic teacher exactly — because the environment is deterministic and SFT hit ~0.99 token accuracy, the student literally reproduces the teacher's action sequence under greedy decoding. This is imitation learning converging to the expert. The meaningful headline number is therefore **SFT vs base**, not SFT vs heuristic.

### Reward sources — what each policy actually earns

![Reward components stacked by policy](./artifacts/reward_components.png)

This is the chart I'm proudest of, because it makes the training signal **legible**. Summed across all three tasks:

- **Random** bleeds out: `closure_wrong: −17.82`, `wrong_actor_penalty: −3.12`, `mitigation_wrong: −2.10`.
- **Base LLM** earns `clue_bonus: +0.24` but then gets crushed by `step_cost: −5.16` and `sla_exhausted: −5.04`. It never fires a single positive closure component.
- **Fine-tuned LLM** unlocks the high-value positive components the base never sees: `closure_correct: +7.36`, `mitigation_correct: +2.10`, `closure_mitigation_bonus: +1.80`, `postmortem_bonus: +0.60`, `handoff_correct: +0.75`, `speed_bonus: +0.60`. Training has moved the LLM from "bleeding" to "solving."

---

## 4. Why does SFT exactly match the heuristic?

Honest framing matters. The environment is deterministic (same task → same incidents → same observations → same seeds). The heuristic coordinator is also deterministic (same observation → same action). So every rollout of a given task produces a byte-identical trajectory. Our 680-row dataset contains only ~85 *unique* `(observation, action)` pairs, each duplicated for redundancy. At ~0.99 token accuracy after 3 epochs, the LLM effectively memorises the heuristic's policy, and under greedy decoding at eval time it reproduces that policy token-for-token on the same deterministic environment.

**This is the defining success condition for behavior cloning: the student has become the teacher.**

The gap we can legitimately celebrate is therefore **SFT vs the untrained base model**, where:
- On **hard incidents**, SFT earns +10.17 more reward than base.
- SFT **unlocks** reward components (`closure_correct`, `mitigation_correct`, `postmortem_bonus`) that the base model literally never fires.
- On easy tasks, SFT inherits the teacher's known weakness (easy tasks have tight SLA budgets that punish thorough investigation). This is exactly what imitation learning should do — including the teacher's mistakes.

The obvious next step to go beyond the heuristic ceiling is RL with the environment's native reward signal — GRPO or PPO against the same rubric — which is the natural Round 3.

---

## 5. What's next

- Replace SFT with PPO or GRPO using the environment's native reward signal (no heuristic teacher).
- Scale the incident catalog from 13 templates to 50+ (drop in JSON-defined scenarios).
- Add a second "adversarial" agent that injects misleading signals to test robustness.

If you want to run it yourself, the Space and the repo are fully self-contained — `docker run` the image and point any OpenEnv-compatible client at it.

---

*Built with ♥ on Meta OpenEnv. Code: [GitHub](https://github.com/SwapnilPatil28/Multi-Agent-Incident-Command-Center). Space: [HF Space](https://huggingface.co/spaces/SwapnilPatil28/Multi-Agent-Incident-Command-Center). Training notebook: [Colab](https://colab.research.google.com/drive/1vx9E5FrZZrHoRwXs2cvtom3DaI6kZ3LP?usp=sharing).*
