import os
import asyncio
from typing import List, Optional
from openai import OpenAI
from client import SupportEnvClient, SupportAction

# 1. Mandatory Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "https://swapnilpatil28-support-env.hf.space")
BENCHMARK = "support_env"

# 2. Logging Helpers (Exactly per Sample Script)
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# 3. Model Interaction Logic
def get_model_action(client: OpenAI, ticket_content: str) -> str:
    try:
        prompt = f"Ticket: {ticket_content}. Reply with ONE word: Billing, Tech, or Sales."
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10
        )
        return completion.choices[0].message.content.strip().strip('.')
    except Exception as e:
        return "Tech" # Fallback

async def run_task(task_name: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SupportEnvClient(base_url=ENV_URL).sync() # Sync wrapper used for simplicity
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Initial Reset
        res = env.reset(task_name=task_name)
        
        while not res.done:
            steps_taken += 1
            action_str = get_model_action(client, res.observation.content)
            
            # Step in environment
            res = env.step(SupportAction(action_type="route", department=action_str))
            
            reward = float(res.reward or 0.0)
            rewards.append(reward)
            
            log_step(step=steps_taken, action=action_str, reward=reward, done=res.done, error=None)

        # Scoring Logic (Normalized [0,1])
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.5

    finally:
        try:
            env.close()
        except:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    # Iterate through tasks sequentially
    for task in ["easy", "medium", "hard"]:
        asyncio.run(run_task(task))
