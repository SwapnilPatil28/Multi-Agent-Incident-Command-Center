from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import IncidentAction, IncidentObservation, IncidentState


class IncidentCommandEnvClient(EnvClient[IncidentAction, IncidentObservation, IncidentState]):
    def _step_payload(self, action: IncidentAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})

        observation = IncidentObservation(
            incident_id=obs_data.get("incident_id", ""),
            incident_title=obs_data.get("incident_title", ""),
            incident_description=obs_data.get("incident_description", ""),
            available_actions=obs_data.get("available_actions", []),
            available_teams=obs_data.get("available_teams", []),
            visible_signals=obs_data.get("visible_signals", []),
            terminal_output=obs_data.get("terminal_output", ""),
            budget_remaining=obs_data.get("budget_remaining", 0),
            sla_minutes_remaining=obs_data.get("sla_minutes_remaining", 0),
            incidents_remaining=obs_data.get("incidents_remaining", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> IncidentState:
        return IncidentState(**payload)


# Backward-compatible alias for older imports.
SREEnvClient = IncidentCommandEnvClient