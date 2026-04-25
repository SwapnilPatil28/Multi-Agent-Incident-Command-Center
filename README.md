---
title: Multi-Agent Incident Command Center
emoji: 🚨
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
license: mit
tags:
  - openenv
  - reinforcement-learning
  - llm-agents
  - multi-agent
  - long-horizon
---

# 🚨 Multi-Agent Incident Command Center (OpenEnv Round 2)

## Problem and Motivation
This environment simulates incident management for a modern software platform under real operational constraints.

The agent must coordinate multiple specialist roles and resolve incidents over long trajectories with partial observability, action costs, and SLA pressure. This targets Round-2 themes:
- **Theme #1 Multi-Agent Interactions**: triage, investigator, and ops-manager role coordination
- **Theme #3.1 World Modeling (Professional Tasks)**: realistic logs/metrics/KB workflows
- **Theme #2 Long-Horizon Planning**: delayed rewards, carry-over constraints, budget-limited sessions

## Environment Design

### Action Space
- `inspect_logs(target)`
- `inspect_metrics(target)`
- `consult_kb(target)`
- `negotiate_handoff(target)` where target is one of:
  - `triage_agent`
  - `investigator_agent`
  - `ops_manager_agent`
- `apply_fix(resolution_summary)`
- `close_incident(root_cause, resolution_summary)`

### Observation Space
- `incident_id`, `incident_title`, `incident_description`
- `visible_signals` (partial clues)
- `available_actions`, `available_teams`
- `budget_remaining`, `sla_minutes_remaining`, `incidents_remaining`
- `terminal_output` (response from world/tool execution)

### Reward Function
- Dense shaping with delayed completion rewards:
  - Small penalty for investigation actions to discourage brute-force scanning
  - Positive reward for discovering new root-cause evidence
  - Bonus for correct specialist handoff
  - Positive reward for effective mitigation
  - Large terminal reward for correct closure (with additional speed bonus)
  - Strong negative reward for wrong closure, SLA exhaustion, or budget exhaustion

## Task Levels
- `easy`: 2 incidents
- `medium`: 3 incidents
- `hard`: 4 incidents with stricter planning requirements

## Local Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run environment
```bash
python -m server.app
```

### Run baseline inference
```bash
python inference.py
```

### OpenEnv validation
```bash
openenv validate
```

## Training Script (TRL)
This repo includes `train_trl.py` for minimum Round-2 training evidence using Hugging Face TRL.

It does:
1. Roll out trajectories from a baseline coordinator
2. Convert trajectories into SFT-style chat examples
3. Train a compact model with `SFTTrainer`
4. Evaluate random vs heuristic policy and save plots

```bash
python train_trl.py
```

Artifacts are written to `artifacts/`:
- `reward_curve.png`
- `summary_metrics.json`

## Hugging Face Space
After testing locally, deploy this repo as a Docker Space and set `app_port=8000`.

## Submission Checklist
- [ ] OpenEnv latest runtime and `openenv validate` passing
- [ ] HF Space URL live and reachable
- [ ] `train_trl.py` (or Colab equivalent) run with real outputs
- [ ] Reward/loss plot images committed and linked
- [ ] 2-minute demo video/blog link added
- [ ] README links all artifacts and references

---
*Environment ID: `incident_command_center_env`*
