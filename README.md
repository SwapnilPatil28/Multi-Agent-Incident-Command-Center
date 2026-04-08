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
---

# Support Ticket Routing Environment

## Description and Motivation
This environment simulates a real-world customer support triage system. Automated agents must process incoming tickets and route them to the correct department (Billing, Tech, or Sales). This is a critical task for scaling customer operations using LLM-based agents.

## Action Space
- **action_type**: "route" or "search"
- **department**: "Billing", "Tech", or "Sales" (Required for route action)

## Observation Space
- **ticket_id**: Unique identifier for the ticket.
- **content**: The raw text of the customer query.
- **search_result**: Information retrieved from the internal DB (if search action is used).
- **available_departments**: ["Billing", "Tech", "Sales"]

## Tasks and Difficulty
- **Easy**: 1 ticket with obvious keywords (e.g., Refund -> Billing).
- **Medium**: 2 tickets with standard support queries.
- **Hard**: 3 tickets involving technical logs or complex enterprise queries.

## Setup and Usage
1. Install dependencies: `pip install openenv-core`
2. Run validation: `openenv validate`
3. Run baseline: `HF_TOKEN=your_token python inference.py`

## Baseline Scores
- Easy: 1.0
- Medium: 1.0
- Hard: 1.0
(Note: Scores are generated using Qwen2.5-72B-Instruct via the Hugging Face Router.)
