from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SupportAction, SupportObservation, SupportState

class SupportEnvClient(EnvClient[SupportAction, SupportObservation, SupportState]):
    def _step_payload(self, action: SupportAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            ticket_id=obs_data.get("ticket_id", ""),
            content=obs_data.get("content", ""),
            search_result=obs_data.get("search_result"),
            available_departments=obs_data.get("available_departments", ["Billing", "Tech", "Sales"])
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> SupportState:
        return SupportState(**payload)
