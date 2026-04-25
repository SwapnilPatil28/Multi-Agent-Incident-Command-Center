from typing import Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class IncidentAction(Action):
    action_type: Literal[
        "inspect_logs",
        "inspect_metrics",
        "consult_kb",
        "negotiate_handoff",
        "apply_fix",
        "close_incident",
    ] = Field(..., description="The action selected by the acting agent.")
    target: Optional[str] = Field(
        None,
        description="Service/dashboard/knowledge id depending on action_type.",
    )
    root_cause: Optional[str] = Field(
        None,
        description="Predicted root cause when action_type=close_incident.",
    )
    resolution_summary: Optional[str] = Field(
        None,
        description="Human-readable fix summary for apply_fix/close_incident.",
    )
    actor: Literal["triage_agent", "investigator_agent", "ops_manager_agent"] = Field(
        "triage_agent",
        description="Which specialist is currently acting in the environment.",
    )


class IncidentObservation(Observation):
    incident_id: str
    incident_title: str
    incident_description: str
    available_actions: List[str] = Field(default_factory=list)
    available_teams: List[str] = Field(default_factory=list)
    visible_signals: List[str] = Field(default_factory=list)
    terminal_output: str = ""
    budget_remaining: int = 0
    sla_minutes_remaining: int = 0
    incidents_remaining: int = 0


class IncidentState(State):
    task_id: str = "easy"
    current_incident_index: int = 0
    incidents_resolved: int = 0
    incidents_failed: int = 0
    budget_remaining: int = 0
    sla_minutes_remaining: int = 0
    mitigation_applied: bool = False
    clues_found: List[str] = Field(default_factory=list)
    handoff_history: List[str] = Field(default_factory=list)
    action_trace: List[str] = Field(default_factory=list)
    per_incident_steps: Dict[str, int] = Field(default_factory=dict)
