import uuid
from typing import Dict, List

from openenv.core.env_server import Environment

from models import IncidentAction, IncidentObservation, IncidentState


class IncidentCommandCenterEnvironment(Environment):
    """Multi-agent, long-horizon SRE incident simulation for OpenEnv."""

    def __init__(self):
        super().__init__()
        self.tasks = self._build_tasks()
        self._task_budgets = {"easy": 24, "medium": 48, "hard": 72}
        self._task_sla = {"easy": 90, "medium": 180, "hard": 300}
        self.current_task: List[Dict[str, object]] = []

    def _build_tasks(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            "easy": [
                {
                    "id": "INC-E1",
                    "title": "Checkout timeouts",
                    "description": "Payment checkout is failing intermittently for premium users.",
                    "root_cause": "redis_connection_pool_exhausted",
                    "signals": [
                        "Spike in checkout latency for premium cohort",
                        "Error budget dropped from 99.9% to 99.2%",
                    ],
                    "logs": {
                        "payments-api": "Timeout waiting for redis write lock",
                        "checkout-worker": "Queue delay exceeds 12s under load",
                        "redis-cluster": "Connection pool exhausted at 512/512",
                    },
                    "metrics": {
                        "dash-checkout": "p99 latency 4.1s, error-rate 6.2%",
                        "dash-redis": "connections 512/512, eviction 0, cpu 74%",
                        "dash-worker": "queue_depth 440, consumer_lag 380",
                    },
                    "kb": {
                        "kb-redis-pool": "Raise redis pool and recycle stale handles in checkout-worker.",
                        "kb-checkout-fallback": "Degrade recommendation calls when payment queue > 300.",
                    },
                    "good_handoff": "investigator_agent",
                    "accepted_fixes": [
                        "increase redis pool",
                        "recycle stale connections",
                        "enable checkout fallback",
                    ],
                },
                {
                    "id": "INC-E2",
                    "title": "Login failures after deploy",
                    "description": "Users report frequent login retries after auth rollout.",
                    "root_cause": "jwt_clock_skew_mismatch",
                    "signals": [
                        "Auth errors spike immediately after deployment",
                        "Regional variance appears in mobile clients",
                    ],
                    "logs": {
                        "auth-service": "Token issued-at in future; rejected by validator",
                        "gateway": "401 bursts from auth-service route",
                        "mobile-api": "Retrying auth flow due to invalid token state",
                    },
                    "metrics": {
                        "dash-auth": "401_rate 14%, token_validation_failures high",
                        "dash-gateway": "auth_route_retries 3.2x baseline",
                    },
                    "kb": {
                        "kb-jwt-time": "Synchronize clock skew tolerance for issuer and verifier.",
                        "kb-mobile-auth": "Fallback to server timestamp for token freshness checks.",
                    },
                    "good_handoff": "ops_manager_agent",
                    "accepted_fixes": [
                        "increase jwt leeway",
                        "sync clock tolerance",
                        "roll back token validator",
                    ],
                },
            ],
            "medium": [
                {
                    "id": "INC-M1",
                    "title": "Catalog stale prices",
                    "description": "Users see old prices during flash sale windows.",
                    "root_cause": "cache_invalidation_topic_lag",
                    "signals": [
                        "Mismatch between checkout and catalog prices",
                        "Issue concentrated in high-traffic products",
                    ],
                    "logs": {
                        "catalog-api": "Read from cache generation=188, expected=193",
                        "kafka-consumer": "Lag increased on invalidation-topic partition 3",
                        "pricing-service": "Published invalidation events at 2.1k/s",
                    },
                    "metrics": {
                        "dash-catalog": "cache_hit 98%, stale_reads elevated",
                        "dash-kafka": "consumer_lag 5400 on partition 3",
                    },
                    "kb": {
                        "kb-cache-invalidation": "Scale invalidation consumers and replay stalled partition.",
                    },
                    "good_handoff": "investigator_agent",
                    "accepted_fixes": [
                        "scale invalidation consumer",
                        "replay partition 3",
                        "flush impacted cache keys",
                    ],
                },
                {
                    "id": "INC-M2",
                    "title": "Shipment ETA corruption",
                    "description": "Shipping ETAs jump unpredictably after route service update.",
                    "root_cause": "timezone_normalization_bug",
                    "signals": [
                        "ETA jumps by +24h in APAC region",
                        "Warehouse scans are on-time, only UI estimate is wrong",
                    ],
                    "logs": {
                        "route-planner": "Parsed timezone fallback=UTC for locale en-IN",
                        "eta-service": "Normalization mismatch for offset +05:30",
                    },
                    "metrics": {
                        "dash-eta": "eta_anomaly_rate 9.4%",
                        "dash-route": "parser_warnings spike post deploy",
                    },
                    "kb": {
                        "kb-timezone": "Use IANA timezone mapping and validate locale fallback path.",
                    },
                    "good_handoff": "triage_agent",
                    "accepted_fixes": [
                        "patch timezone parser",
                        "use iana timezone map",
                        "rollback route update",
                    ],
                },
                {
                    "id": "INC-M3",
                    "title": "Invoice duplicates",
                    "description": "A subset of merchants received duplicate invoices.",
                    "root_cause": "idempotency_key_regression",
                    "signals": [
                        "Duplicate invoices share same order id",
                        "Triggered after billing retry logic change",
                    ],
                    "logs": {
                        "billing-worker": "Retry path ignored idempotency token for v2 flow",
                        "billing-api": "POST /invoice executed twice for order O-92A",
                    },
                    "metrics": {
                        "dash-billing": "duplicate_invoice_rate 3.7%",
                        "dash-worker": "retry_attempts 2.4x",
                    },
                    "kb": {
                        "kb-idempotency": "Persist retry token before dispatch and enforce dedupe check.",
                    },
                    "good_handoff": "ops_manager_agent",
                    "accepted_fixes": [
                        "restore idempotency guard",
                        "persist retry token first",
                        "dedupe duplicate invoice jobs",
                    ],
                },
            ],
            "hard": [
                {
                    "id": "INC-H1",
                    "title": "Cross-service saturation cascade",
                    "description": "A sudden promo launch causes cascading failures across checkout, auth, and notification services.",
                    "root_cause": "rate_limit_misconfigured_for_promo_segment",
                    "signals": [
                        "Failure spreads from notifications to checkout within minutes",
                        "Customer segment 'promo_mega' has concentrated failures",
                    ],
                    "logs": {
                        "notification-gateway": "429 flood for promo_mega segment",
                        "checkout-api": "Retries amplified upstream failures from notification sidecar",
                        "auth-service": "Session refresh queue saturation due to retry storm",
                    },
                    "metrics": {
                        "dash-global": "error budget burn 3.7x",
                        "dash-notify": "429_rate 38%",
                        "dash-auth": "session_queue_depth 940",
                    },
                    "kb": {
                        "kb-rate-limits": "Segment-specific limits must be applied with gradual rollout and backoff.",
                    },
                    "good_handoff": "ops_manager_agent",
                    "accepted_fixes": [
                        "hotfix promo segment rate limits",
                        "enable exponential backoff",
                        "throttle notification fanout",
                    ],
                },
                {
                    "id": "INC-H2",
                    "title": "Data export corruption",
                    "description": "Enterprise customers report corrupted CSV exports from analytics dashboard.",
                    "root_cause": "schema_version_drift",
                    "signals": [
                        "Corruption only in accounts migrated last week",
                        "Export job success is high but data quality is low",
                    ],
                    "logs": {
                        "export-worker": "Schema mismatch: expected v11 got v10 on tenant shard",
                        "analytics-api": "Fallback serializer dropped nullable columns",
                    },
                    "metrics": {
                        "dash-export": "job_success 97%, data_quality_score 61%",
                        "dash-analytics": "schema_mismatch counter rising",
                    },
                    "kb": {
                        "kb-schema-drift": "Force schema negotiation at read time and backfill migrated shards.",
                    },
                    "good_handoff": "investigator_agent",
                    "accepted_fixes": [
                        "enforce schema negotiation",
                        "backfill migrated shards",
                        "pin serializer to v11",
                    ],
                },
                {
                    "id": "INC-H3",
                    "title": "On-call alert storm",
                    "description": "On-call rotations are overwhelmed by noisy duplicate alerts, masking a real outage.",
                    "root_cause": "dedupe_rule_disabled",
                    "signals": [
                        "Alert volume 10x baseline with low incident diversity",
                        "Primary outage not visible in first-page alerts",
                    ],
                    "logs": {
                        "alert-router": "Deduplication pipeline bypassed after config reload",
                        "pager-service": "Repeated notifications for identical fingerprint",
                    },
                    "metrics": {
                        "dash-alerts": "alerts_per_minute 1200",
                        "dash-pager": "notification_duplicates 87%",
                    },
                    "kb": {
                        "kb-alert-dedupe": "Restore dedupe stage and replay suppressed critical fingerprint set.",
                    },
                    "good_handoff": "triage_agent",
                    "accepted_fixes": [
                        "restore dedupe rule",
                        "replay critical fingerprints",
                        "mute duplicate alert channels",
                    ],
                },
                {
                    "id": "INC-H4",
                    "title": "Inventory phantom stock",
                    "description": "Inventory service reports available stock that does not exist in warehouse.",
                    "root_cause": "event_ordering_race_condition",
                    "signals": [
                        "Negative physical stock but positive ledger entries",
                        "Warehouse reconciliation jobs are delayed",
                    ],
                    "logs": {
                        "inventory-ledger": "Out-of-order reserve/release events for same SKU",
                        "warehouse-sync": "Late event merge exceeded ordering window",
                    },
                    "metrics": {
                        "dash-inventory": "oversell_incidents 4.2%",
                        "dash-sync": "late_event_ratio 17%",
                    },
                    "kb": {
                        "kb-event-ordering": "Use monotonic sequence guards and quarantine out-of-order events.",
                    },
                    "good_handoff": "investigator_agent",
                    "accepted_fixes": [
                        "enable sequence guards",
                        "quarantine out-of-order events",
                        "reconcile affected skus",
                    ],
                },
            ],
        }

    def reset(self, task_name: str = "easy") -> IncidentObservation:
        selected_task = task_name if task_name in self.tasks else "easy"
        self.current_task = self.tasks[selected_task]
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            task_id=selected_task,
            current_incident_index=0,
            budget_remaining=self._task_budgets[selected_task],
            sla_minutes_remaining=self._task_sla[selected_task],
        )
        return self._observation_for_current_incident(
            terminal_output=(
                "Incident Command Center initialized. "
                "Coordinate triage_agent, investigator_agent, and ops_manager_agent."
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        self._state.step_count += 1
        self._state.sla_minutes_remaining = max(0, self._state.sla_minutes_remaining - 5)
        self._state.budget_remaining -= 1

        if self._state.current_incident_index >= len(self.current_task):
            return IncidentObservation(
                done=True,
                reward=0.0,
                incident_id="EOF",
                incident_title="All incidents completed",
                incident_description="Episode ended.",
                terminal_output="No remaining incidents.",
            )

        if self._state.budget_remaining < 0:
            self._state.incidents_failed += 1
            return IncidentObservation(
                done=True,
                reward=-1.5,
                incident_id="BUDGET_EXHAUSTED",
                incident_title="Resource budget exhausted",
                incident_description="Agent used too many actions before finishing the task.",
                terminal_output="Episode terminated: investigation budget exhausted.",
                budget_remaining=0,
                sla_minutes_remaining=self._state.sla_minutes_remaining,
                incidents_remaining=len(self.current_task) - self._state.current_incident_index,
            )

        incident = self.current_task[self._state.current_incident_index]
        incident_id = str(incident["id"])
        self._state.per_incident_steps[incident_id] = (
            self._state.per_incident_steps.get(incident_id, 0) + 1
        )
        self._state.action_trace.append(f"{action.actor}:{action.action_type}:{action.target or '-'}")

        if self._state.sla_minutes_remaining <= 0:
            self._state.incidents_failed += 1
            return IncidentObservation(
                done=True,
                reward=-1.2,
                incident_id=incident_id,
                incident_title=str(incident["title"]),
                incident_description=str(incident["description"]),
                terminal_output="Episode terminated: global SLA budget reached zero.",
                budget_remaining=max(self._state.budget_remaining, 0),
                sla_minutes_remaining=0,
                incidents_remaining=len(self.current_task) - self._state.current_incident_index,
            )

        reward = 0.0
        terminal_output = ""

        if action.action_type == "inspect_logs":
            reward -= 0.04
            lookup = (action.target or "").strip()
            logs = incident["logs"]
            terminal_output = logs.get(lookup, f"No logs found for target '{lookup}'.")
            reward += self._grant_clue_reward(incident, terminal_output)

        elif action.action_type == "inspect_metrics":
            reward -= 0.04
            lookup = (action.target or "").strip()
            metrics = incident["metrics"]
            terminal_output = metrics.get(lookup, f"No metrics found for target '{lookup}'.")
            reward += self._grant_clue_reward(incident, terminal_output)

        elif action.action_type == "consult_kb":
            reward -= 0.03
            lookup = (action.target or "").strip()
            kb = incident["kb"]
            terminal_output = kb.get(lookup, f"No KB article found for key '{lookup}'.")
            reward += self._grant_clue_reward(incident, terminal_output)

        elif action.action_type == "negotiate_handoff":
            reward -= 0.02
            team = (action.target or "").strip()
            self._state.handoff_history.append(team)
            if team == incident["good_handoff"]:
                reward += 0.12
                terminal_output = (
                    f"Handoff accepted by {team}. "
                    "New hypothesis confidence increased."
                )
            else:
                reward -= 0.10
                terminal_output = (
                    f"Handoff to {team} introduced delay. "
                    "This incident likely needs a different owner."
                )

        elif action.action_type == "apply_fix":
            reward -= 0.02
            fix_text = (action.resolution_summary or "").lower()
            accepted_fixes = incident["accepted_fixes"]
            is_good_fix = any(token in fix_text for token in accepted_fixes)
            if is_good_fix:
                self._state.mitigation_applied = True
                reward += 0.35
                terminal_output = "Mitigation accepted. Error rate is stabilizing."
            else:
                reward -= 0.30
                terminal_output = "Applied mitigation appears ineffective."

        elif action.action_type == "close_incident":
            guess = (action.root_cause or "").strip().lower()
            expected = str(incident["root_cause"]).lower()
            correct = guess == expected
            episode_done = False
            if correct:
                completion_reward = 0.80
                if self._state.mitigation_applied:
                    completion_reward += 0.30
                completion_reward += self._speed_bonus(incident_id)
                reward += completion_reward
                self._state.incidents_resolved += 1
                terminal_output = (
                    "Incident resolved successfully. "
                    f"Root cause confirmed: {incident['root_cause']}."
                )
            else:
                reward -= 1.10
                self._state.incidents_failed += 1
                terminal_output = (
                    "Incident closure rejected by postmortem checker. "
                    f"Expected root cause differs from '{guess or 'unknown'}'."
                )

            self._advance_incident()
            if self._state.current_incident_index >= len(self.current_task):
                episode_done = True
                terminal_output += " All assigned incidents processed."
            else:
                next_incident = self.current_task[self._state.current_incident_index]
                terminal_output += f" Next incident: {next_incident['id']}."

            return self._observation_for_current_incident(
                terminal_output=terminal_output,
                reward=reward,
                done=episode_done,
            )

        else:
            reward -= 0.25
            terminal_output = f"Unsupported action_type: {action.action_type}"

        return self._observation_for_current_incident(
            terminal_output=terminal_output,
            reward=reward,
            done=False,
        )

    def _grant_clue_reward(self, incident: Dict[str, object], signal_text: str) -> float:
        root = str(incident["root_cause"]).lower()
        signal_key = signal_text.strip().lower()
        if root in signal_key and signal_key not in self._state.clues_found:
            self._state.clues_found.append(signal_key)
            return 0.12
        return 0.0

    def _speed_bonus(self, incident_id: str) -> float:
        steps_used = self._state.per_incident_steps.get(incident_id, 1)
        if steps_used <= 4:
            return 0.20
        if steps_used <= 7:
            return 0.10
        return 0.0

    def _advance_incident(self) -> None:
        self._state.current_incident_index += 1
        self._state.mitigation_applied = False
        self._state.clues_found = []

    def _observation_for_current_incident(
        self, terminal_output: str, reward: float, done: bool
    ) -> IncidentObservation:
        if done:
            return IncidentObservation(
                done=True,
                reward=reward,
                incident_id="EOF",
                incident_title="All incidents completed",
                incident_description="Episode ended.",
                available_actions=[],
                available_teams=[],
                visible_signals=[],
                terminal_output=terminal_output,
                budget_remaining=max(self._state.budget_remaining, 0),
                sla_minutes_remaining=self._state.sla_minutes_remaining,
                incidents_remaining=0,
            )

        incident = self.current_task[self._state.current_incident_index]
        return IncidentObservation(
            done=False,
            reward=reward,
            incident_id=str(incident["id"]),
            incident_title=str(incident["title"]),
            incident_description=str(incident["description"]),
            available_actions=[
                "inspect_logs",
                "inspect_metrics",
                "consult_kb",
                "negotiate_handoff",
                "apply_fix",
                "close_incident",
            ],
            available_teams=["triage_agent", "investigator_agent", "ops_manager_agent"],
            visible_signals=list(incident["signals"]),
            terminal_output=terminal_output,
            budget_remaining=max(self._state.budget_remaining, 0),
            sla_minutes_remaining=self._state.sla_minutes_remaining,
            incidents_remaining=len(self.current_task) - self._state.current_incident_index,
        )

    @property
    def state(self) -> IncidentState:
        return self._state
