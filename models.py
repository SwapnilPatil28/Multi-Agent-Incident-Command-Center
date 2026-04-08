from typing import List, Optional, Literal
from openenv.core.env_server import Action, Observation, State
from pydantic import Field

class SupportAction(Action):
    action_type: Literal["search", "route"] = Field(..., description="Action to take")
    department: Optional[str] = Field(None, description="Required for route action")

class SupportObservation(Observation):
    ticket_id: str
    content: str
    search_result: Optional[str] = None
    available_departments: List[str] = ["Billing", "Tech", "Sales"]

class SupportState(State):
    task_id: str = "easy"
    current_ticket_index: int = 0
