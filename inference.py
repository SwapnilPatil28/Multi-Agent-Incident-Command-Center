import asyncio
import os
import random
from typing import Dict, List, Optional

from client import IncidentCommandEnvClient
from models import IncidentAction

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
BENCHMARK = "incident_command_center_env"
RANDOM_BASELINE = os.getenv("RANDOM_BASELINE", "false").lower() == "true"


def log_start(task: str, env: str, policy: str) -> None:
    print(f"[START] task={task} env={env} policy={policy}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


class HeuristicCoordinator:
    """Simple policy for baseline demonstrations and offline data generation."""

    def __init__(self) -> None:
        self._phase_by_incident: Dict[str, int] = {}
        self._suspects_by_incident: Dict[str, str] = {}

    def select_action(self, observation) -> IncidentAction:
        incident_id = observation.incident_id
        text = (
            f"{observation.incident_title} {observation.incident_description} "
            f"{' '.join(observation.visible_signals)} {observation.terminal_output}"
        ).lower()
        phase = self._phase_by_incident.get(incident_id, 0)

        if phase == 0:
            self._phase_by_incident[incident_id] = 1
            return IncidentAction(
                actor="triage_agent",
                action_type="inspect_logs",
                target=self._pick_log_target(text),
            )
        if phase == 1:
            self._phase_by_incident[incident_id] = 2
            return IncidentAction(
                actor="investigator_agent",
                action_type="inspect_metrics",
                target=self._pick_metric_target(text),
            )
        if phase == 2:
            self._phase_by_incident[incident_id] = 3
            owner = self._pick_owner(text)
            return IncidentAction(
                actor="ops_manager_agent",
                action_type="negotiate_handoff",
                target=owner,
            )
        if phase == 3:
            self._phase_by_incident[incident_id] = 4
            guess = self._infer_root_cause(text)
            self._suspects_by_incident[incident_id] = guess
            return IncidentAction(
                actor="investigator_agent",
                action_type="apply_fix",
                resolution_summary=self._generate_fix_plan(guess),
            )

        guess = self._suspects_by_incident.get(incident_id, self._infer_root_cause(text))
        return IncidentAction(
            actor="ops_manager_agent",
            action_type="close_incident",
            root_cause=guess,
            resolution_summary=f"Closed with hypothesis {guess}.",
        )

    def _pick_log_target(self, text: str) -> str:
        mapping = {
            "checkout": "payments-api",
            "login": "auth-service",
            "catalog": "catalog-api",
            "shipment": "route-planner",
            "invoice": "billing-worker",
            "cascade": "notification-gateway",
            "export": "export-worker",
            "alert": "alert-router",
            "inventory": "inventory-ledger",
        }
        return self._pick_from_mapping(text, mapping, "auth-service")

    def _pick_metric_target(self, text: str) -> str:
        mapping = {
            "checkout": "dash-redis",
            "login": "dash-auth",
            "catalog": "dash-kafka",
            "shipment": "dash-eta",
            "invoice": "dash-billing",
            "cascade": "dash-notify",
            "export": "dash-export",
            "alert": "dash-alerts",
            "inventory": "dash-inventory",
        }
        return self._pick_from_mapping(text, mapping, "dash-global")

    def _pick_owner(self, text: str) -> str:
        if any(token in text for token in ["deploy", "rate", "sla", "rotation"]):
            return "ops_manager_agent"
        if any(token in text for token in ["schema", "export", "cache", "inventory"]):
            return "investigator_agent"
        return "triage_agent"

    def _infer_root_cause(self, text: str) -> str:
        if "redis" in text and "pool" in text:
            return "redis_connection_pool_exhausted"
        if "jwt" in text or "token" in text:
            return "jwt_clock_skew_mismatch"
        if "cache" in text and "invalidation" in text:
            return "cache_invalidation_topic_lag"
        if "timezone" in text or "offset" in text:
            return "timezone_normalization_bug"
        if "idempotency" in text or "duplicate invoice" in text:
            return "idempotency_key_regression"
        if "429" in text or "promo" in text:
            return "rate_limit_misconfigured_for_promo_segment"
        if "schema" in text and "drift" in text:
            return "schema_version_drift"
        if "dedupe" in text or "alert storm" in text:
            return "dedupe_rule_disabled"
        if "out-of-order" in text or "oversell" in text:
            return "event_ordering_race_condition"
        return "unknown"

    def _generate_fix_plan(self, root_cause: str) -> str:
        fixes = {
            "redis_connection_pool_exhausted": "increase redis pool and recycle stale connections",
            "jwt_clock_skew_mismatch": "sync clock tolerance and increase jwt leeway",
            "cache_invalidation_topic_lag": "scale invalidation consumer and replay partition 3",
            "timezone_normalization_bug": "patch timezone parser and use iana timezone map",
            "idempotency_key_regression": "restore idempotency guard and persist retry token first",
            "rate_limit_misconfigured_for_promo_segment": "hotfix promo segment rate limits and enable exponential backoff",
            "schema_version_drift": "enforce schema negotiation and pin serializer to v11",
            "dedupe_rule_disabled": "restore dedupe rule and replay critical fingerprints",
            "event_ordering_race_condition": "enable sequence guards and quarantine out-of-order events",
        }
        return fixes.get(root_cause, "collect additional diagnostics and rollback last change")

    def _pick_from_mapping(self, text: str, mapping: Dict[str, str], default: str) -> str:
        for token, value in mapping.items():
            if token in text:
                return value
        return default


def random_action(observation) -> IncidentAction:
    action_type = random.choice(observation.available_actions or ["inspect_logs"])
    teams = observation.available_teams or ["triage_agent", "investigator_agent", "ops_manager_agent"]
    actor = random.choice(teams)
    random_target = random.choice(
        [
            "payments-api",
            "auth-service",
            "dash-auth",
            "dash-redis",
            "kb-rate-limits",
            "investigator_agent",
        ]
    )
    return IncidentAction(
        actor=actor,
        action_type=action_type,
        target=random_target,
        root_cause="unknown",
        resolution_summary="random baseline action",
    )


async def run_task(task_name: str):
    env = IncidentCommandEnvClient(base_url=ENV_URL).sync()
    policy_name = "random_baseline" if RANDOM_BASELINE else "heuristic_coordinator"
    coordinator = HeuristicCoordinator()

    log_start(task=task_name, env=BENCHMARK, policy=policy_name)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        res = env.reset(task_name=task_name)
        while not res.done:
            steps_taken += 1
            action = random_action(res.observation) if RANDOM_BASELINE else coordinator.select_action(
                res.observation
            )
            res = env.step(action)
            reward = float(res.reward or 0.0)
            rewards.append(reward)
            log_step(
                step=steps_taken,
                action=f"{action.actor}:{action.action_type}:{action.target or '-'}",
                reward=reward,
                done=res.done,
                error=None,
            )

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.2
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    for task in ["easy", "medium", "hard"]:
        asyncio.run(run_task(task))


if __name__ == "__main__":
    main()
