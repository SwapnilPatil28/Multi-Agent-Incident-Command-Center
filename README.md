---
title: Support Ticket Routing
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
license: mit
tags:
  - openenv
  - reinforcement-learning
  - llm-agents
---

# 🎫 Customer Support Ticket Routing Environment

## 📝 Description and Motivation
This environment simulates a production-grade customer support triage system. Automated agents are tasked with analyzing raw customer queries and routing them to the appropriate department: **Billing**, **Tech**, or **Sales**. 

In real-world scenarios, misrouting leads to high churn and operational costs. This benchmark measures the ability of LLM-based agents to perform high-precision classification in a restricted environment compliant with the `openenv-core` SDK.

## 🎯 Environment Specification

### Action Space
- `action_type`: Literal["route", "search"]
- `department`: Optional[str] — Required for `route` action. Valid values: `"Billing"`, `"Tech"`, `"Sales"`.

### Observation Space
- `ticket_id`: Unique tracking ID (e.g., T1, T4).
- `content`: The raw text string of the customer's request.
- `search_result`: Contextual data retrieved from the internal database (if the `search` action is invoked).
- `available_departments`: A list of valid routing targets.

### Reward Function
To facilitate stable training and clear evaluation metrics, this environment uses **strictly bounded rewards**:
- **0.99**: Correct Department Routing.
- **0.01**: Incorrect Department Routing.
- **-0.05**: Search Penalty (Encourages efficiency unless context is truly needed).

## 🏁 Tasks and Difficulty
| Task ID | Tickets | Description |
| :--- | :--- | :--- |
| `easy` | 1 | Clear keywords (e.g., "Refund", "Invoice"). |
| `medium` | 2 | Standard conversational support language. |
| `hard` | 3 | Complex queries involving API logs and technical stack traces. |

## 🚀 Setup & Benchmarking

### 1. Installation
```bash
pip install openenv-core uvicorn openai
```

### 2. Run Local Validation
Ensure your local setup matches the competition requirements:
```bash
openenv validate
```

### 3. Run Baseline Inference
Execute the provided baseline using the Hugging Face Router and the Qwen2.5-72B model:
```bash
export HF_TOKEN="your_huggingface_token"
python inference.py
```

## 🛠️ Technical Architecture
- **Backend**: Python FastAPI serving `openenv-core` compatible endpoints.
- **Infrastructure**: Containerized deployment via Docker on Hugging Face Spaces.
- **Models**: Pydantic-based state and action validation.

---
*Submission for the Scaler Meta PyTorch Hackathon.*
*Environment ID: `support_env` | Powered by OpenEnv SDK.*
