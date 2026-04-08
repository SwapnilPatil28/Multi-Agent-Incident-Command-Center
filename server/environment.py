import uuid
from typing import List, Dict
from openenv.core.env_server import Environment
from models import SupportAction, SupportObservation, SupportState

class SupportEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._db = {
            "T1": "Customer History: Frequent refunder. Status: Valid invoice.",
            "T4": "System Log: Payment gateway timed out. Card ending in 4242.",
            "T6": "Dev Log: 500 error on /api/auth. Cluster: us-east-1."
        }
        self.tasks = {
            "easy": [{"id": "T1", "text": "Refund status?", "dept": "Billing"}],
            "medium": [{"id": "T2", "text": "Upgrade account?", "dept": "Sales"},
                       {"id": "T3", "text": "App keeps crashing.", "dept": "Tech"}],
            "hard": [{"id": "T4", "text": "Payment failed!", "dept": "Billing"},
                     {"id": "T5", "text": "Bulk pricing?", "dept": "Sales"},
                     {"id": "T6", "text": "API Auth failure.", "dept": "Tech"}]
        }

    def reset(self, task_name: str = "easy") -> SupportObservation:
        self.current_task = self.tasks.get(task_name, self.tasks["easy"])
        self._state = SupportState(episode_id=str(uuid.uuid4()), task_id=task_name, current_ticket_index=0)
        t = self.current_task[0]
        return SupportObservation(done=False, reward=0.0, ticket_id=t["id"], content=t["text"])

    def step(self, action: SupportAction) -> SupportObservation:
        self._state.step_count += 1
        ticket = self.current_task[self._state.current_ticket_index]

        if action.action_type == "search":
            return SupportObservation(done=False, reward=-0.05, ticket_id=ticket["id"], content=ticket["text"], search_result=self._db.get(ticket["id"], "No record."))

        correct = action.department.strip().lower() == ticket["dept"].lower()
        reward = 1.0 if correct else 0.0
        self._state.current_ticket_index += 1

        if self._state.current_ticket_index < len(self.current_task):
            t = self.current_task[self._state.current_ticket_index]
            return SupportObservation(done=False, reward=reward, ticket_id=t["id"], content=t["text"])
        return SupportObservation(done=True, reward=reward, ticket_id="EOF", content="Done.")

    @property
    def state(self) -> SupportState:
        return self._state
