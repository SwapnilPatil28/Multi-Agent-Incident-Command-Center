"""Hugging Face TRL training + evaluation pipeline.

Pipeline:

1. **Rollout**: run the ``HeuristicCoordinator`` against the live Incident
   Command Center environment to collect ``(prompt, completion)`` pairs.
2. **SFT**: fine-tune a small instruction-tuned LLM on those pairs using
   TRL's ``SFTTrainer`` with a single ``text`` column (robust across TRL
   ≥ 0.20).
3. **Save**: persist the fine-tuned weights + tokenizer to
   ``artifacts/sft_model`` so the same script can later load them as an
   agent policy.
4. **Evaluate**: play the environment with four policies
   ``random / heuristic / base_model / sft_model`` under identical seeds
   and write a reward curve + metrics JSON into ``artifacts/``.

Designed to work on CPU for smoke checks and on Colab T4 / HF Spaces GPUs
for full runs. LLM evaluation auto-enables on CUDA and can be forced with
``EVAL_LLM_MODELS=true``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
from datasets import Dataset

from client import IncidentCommandEnvClient
from inference import HeuristicCoordinator, random_action
from models import IncidentAction, IncidentObservation


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SFT_MODEL_DIR = ARTIFACT_DIR / "sft_model"

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_ROLLOUT_STEPS = int(os.getenv("MAX_ROLLOUT_STEPS", "120"))
MAX_LLM_EVAL_STEPS = int(os.getenv("MAX_LLM_EVAL_STEPS", "60"))
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "3"))
TRAIN_EPOCHS = float(os.getenv("TRAIN_EPOCHS", "1"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
TRAIN_GRAD_ACCUM = int(os.getenv("TRAIN_GRAD_ACCUM", "2"))
TRAIN_MAX_LENGTH = int(os.getenv("TRAIN_MAX_LENGTH", "768"))
_EVAL_LLM_ENV = os.getenv("EVAL_LLM_MODELS", "auto").strip().lower()


@dataclass
class EpisodeStats:
    policy_name: str
    task_name: str
    total_reward: float
    steps: int
    success: bool
    components: Dict[str, float] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Prompt / completion formatting
# ---------------------------------------------------------------------------


def obs_to_prompt(obs: IncidentObservation) -> str:
    targets = obs.investigation_targets or {}
    return (
        "You are operating a multi-agent incident command center. "
        "Pick the next action for the appropriate specialist role.\n\n"
        f"Incident ID: {obs.incident_id}\n"
        f"Title: {obs.incident_title}\n"
        f"Description: {obs.incident_description}\n"
        f"Customer tier: {obs.customer_tier} | "
        f"Affected users: {obs.affected_users_estimate} | "
        f"Revenue impact (USD/min): {obs.revenue_impact_usd_per_min}\n"
        f"Postmortem required: {obs.postmortem_required}\n"
        f"Visible signals: {', '.join(obs.visible_signals or [])}\n"
        f"Available log targets: {', '.join(targets.get('logs', []) or [])}\n"
        f"Available metric targets: {', '.join(targets.get('metrics', []) or [])}\n"
        f"Available KB articles: {', '.join(targets.get('kb', []) or [])}\n"
        f"Budget remaining: {obs.budget_remaining} actions | "
        f"SLA remaining: {obs.sla_minutes_remaining} min | "
        f"Clues found: {obs.clues_found} | "
        f"Mitigation applied: {obs.mitigation_applied}\n"
        f"Last terminal output: {obs.terminal_output}\n\n"
        "Respond with a JSON object containing exactly these keys: "
        "actor, action_type, target, root_cause, resolution_summary, "
        "postmortem_note, confidence, reason."
    )


def action_to_json(action: IncidentAction) -> str:
    payload = action.model_dump(exclude_none=True)
    return json.dumps(payload, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Rollout / dataset construction
# ---------------------------------------------------------------------------


def rollout(
    policy_name: str,
    task_name: str,
    collect_dataset: bool = False,
    policy_callable: Optional[Callable[[IncidentObservation], IncidentAction]] = None,
    max_steps: Optional[int] = None,
):
    """Play one episode and return (stats, rows, rewards).

    If ``policy_callable`` is provided it takes precedence over
    ``policy_name`` — this is how the LLM policies plug in.
    """
    env = IncidentCommandEnvClient(base_url=ENV_URL).sync()
    coordinator = HeuristicCoordinator()
    records: List[Dict[str, str]] = []
    rewards: List[float] = []
    components_sum: Dict[str, float] = {}
    steps = 0
    step_cap = max_steps if max_steps is not None else MAX_ROLLOUT_STEPS

    try:
        result = env.reset(task_name=task_name)
        while not result.done and steps < step_cap:
            steps += 1
            if policy_callable is not None:
                action = policy_callable(result.observation)
            elif policy_name == "heuristic":
                action = coordinator.select_action(result.observation)
            else:
                action = random_action(result.observation)

            if collect_dataset:
                records.append(
                    {
                        "prompt": obs_to_prompt(result.observation),
                        "completion": action_to_json(action),
                    }
                )

            result = env.step(action)
            rewards.append(float(result.reward or 0.0))
            step_components = getattr(result.observation, "reward_components", None) or {}
            for key, value in step_components.items():
                components_sum[key] = components_sum.get(key, 0.0) + float(value)
    finally:
        try:
            env.close()
        except Exception:
            pass

    total_reward = sum(rewards)
    success = total_reward > 0.0
    stats = EpisodeStats(
        policy_name=policy_name,
        task_name=task_name,
        total_reward=total_reward,
        steps=steps,
        success=success,
        components={k: round(v, 4) for k, v in components_sum.items()},
    )
    return (stats, records, rewards)


def build_training_dataset(episodes_per_task: int = EPISODES_PER_TASK) -> Dataset:
    rows: List[Dict[str, str]] = []
    for task in ["easy", "medium", "hard"]:
        for _ in range(episodes_per_task):
            _, new_rows, _ = rollout(
                policy_name="heuristic", task_name=task, collect_dataset=True
            )
            rows.extend(new_rows)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# TRL SFT
# ---------------------------------------------------------------------------


def _dataset_to_sft_text_column(dataset: Dataset, tokenizer) -> Dataset:
    """Collapse (prompt, completion) pairs into a single `text` field."""
    from transformers import PreTrainedTokenizerBase

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        return dataset

    cols = set(dataset.column_names)
    if "completion" not in cols and "response" in cols:
        dataset = dataset.rename_column("response", "completion")
    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        raise ValueError(
            f"Expected columns 'prompt' and 'completion' (or 'response'). "
            f"Got: {dataset.column_names}"
        )

    has_template = bool(getattr(tokenizer, "chat_template", None))

    def to_text_batched(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: List[str] = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            if has_template:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                out.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            else:
                out.append(f"User: {prompt}\n\nAssistant: {completion}")
        return {"text": out}

    to_drop = [c for c in dataset.column_names if c != "text"]
    return dataset.map(to_text_batched, batched=True, remove_columns=to_drop)


def run_trl_sft(dataset: Dataset) -> Path:
    """Fine-tune ``BASE_MODEL`` on the collected dataset and save the model.

    Returns the directory of the saved SFT checkpoint (``artifacts/sft_model``).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    train_ds = _dataset_to_sft_text_column(dataset, tokenizer)

    config = SFTConfig(
        output_dir="outputs/sft_run",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
        learning_rate=2e-5,
        num_train_epochs=TRAIN_EPOCHS,
        max_length=TRAIN_MAX_LENGTH,
        dataset_text_field="text",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    SFT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SFT_MODEL_DIR))
    tokenizer.save_pretrained(str(SFT_MODEL_DIR))

    log_path = ARTIFACT_DIR / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    print(f"[train] Saved SFT checkpoint to {SFT_MODEL_DIR}")
    print(f"[train] Saved training log to {log_path}")

    del trainer, model, tokenizer
    _free_gpu_memory()
    return SFT_MODEL_DIR


# ---------------------------------------------------------------------------
# Evaluation + reporting
# ---------------------------------------------------------------------------


def _free_gpu_memory() -> None:
    try:
        import gc
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _should_evaluate_llms() -> bool:
    if _EVAL_LLM_ENV in {"1", "true", "yes", "on"}:
        return True
    if _EVAL_LLM_ENV in {"0", "false", "no", "off"}:
        return False
    # "auto" / empty: enable only when a CUDA GPU is available so CPU runs
    # stay fast.
    return _cuda_available()


def _evaluate_single_policy(
    policy_name: str,
    select_fn: Callable[[IncidentObservation], IncidentAction],
    max_steps: Optional[int] = None,
    components_accumulator: Optional[Dict[str, float]] = None,
) -> List[float]:
    scores: List[float] = []
    for task in ["easy", "medium", "hard"]:
        stats, _, _ = rollout(
            policy_name=policy_name,
            task_name=task,
            policy_callable=select_fn,
            max_steps=max_steps,
        )
        print(
            f"[eval] policy={policy_name} task={task} "
            f"reward={stats.total_reward:+.2f} steps={stats.steps}"
        )
        scores.append(round(stats.total_reward, 4))
        if components_accumulator is not None and stats.components:
            for k, v in stats.components.items():
                components_accumulator[k] = components_accumulator.get(k, 0.0) + v
    return scores


def evaluate_policies(
    seed: int = 7,
    evaluate_llms: Optional[bool] = None,
) -> Dict[str, object]:
    """Run each policy once per task under the same seed.

    Returns a dict with keys ``scores`` (mapping policy -> [easy, medium, hard])
    and ``components`` (mapping policy -> {component_name: summed_value}).
    """
    random.seed(seed)

    scores: Dict[str, List[float]] = {
        "random": [],
        "heuristic": [],
        "base_model": [],
        "sft_model": [],
    }
    components: Dict[str, Dict[str, float]] = {
        "random": {},
        "heuristic": {},
        "base_model": {},
        "sft_model": {},
    }

    for task in ["easy", "medium", "hard"]:
        random_stats, _, _ = rollout("random", task)
        heuristic_stats, _, _ = rollout("heuristic", task)
        scores["random"].append(round(random_stats.total_reward, 4))
        scores["heuristic"].append(round(heuristic_stats.total_reward, 4))
        for k, v in (random_stats.components or {}).items():
            components["random"][k] = components["random"].get(k, 0.0) + v
        for k, v in (heuristic_stats.components or {}).items():
            components["heuristic"][k] = components["heuristic"].get(k, 0.0) + v

    should_eval_llms = _should_evaluate_llms() if evaluate_llms is None else evaluate_llms
    if not should_eval_llms:
        print("[eval] Skipping LLM evaluation (no GPU or EVAL_LLM_MODELS=false).")
        return {"scores": scores, "components": components}

    try:
        from llm_policy import LLMPolicy
    except Exception as exc:  # pragma: no cover - import-time safety
        print(f"[eval] Could not import LLMPolicy ({exc}); skipping LLM eval.")
        return {"scores": scores, "components": components}

    # Base model
    try:
        print(f"[eval] Loading BASE model: {BASE_MODEL}")
        base = LLMPolicy(BASE_MODEL, label="base_model")
        scores["base_model"] = _evaluate_single_policy(
            "base_model",
            base.select_action,
            max_steps=MAX_LLM_EVAL_STEPS,
            components_accumulator=components["base_model"],
        )
        base.release()
        _free_gpu_memory()
    except Exception as exc:
        print(f"[eval] Base-model evaluation failed: {exc}")

    # SFT model
    if SFT_MODEL_DIR.exists():
        try:
            print(f"[eval] Loading SFT model: {SFT_MODEL_DIR}")
            sft = LLMPolicy(str(SFT_MODEL_DIR), label="sft_model")
            scores["sft_model"] = _evaluate_single_policy(
                "sft_model",
                sft.select_action,
                max_steps=MAX_LLM_EVAL_STEPS,
                components_accumulator=components["sft_model"],
            )
            sft.release()
            _free_gpu_memory()
        except Exception as exc:
            print(f"[eval] SFT-model evaluation failed: {exc}")
    else:
        print(f"[eval] No SFT checkpoint found at {SFT_MODEL_DIR}; skipping SFT eval.")

    return {"scores": scores, "components": components}


def plot_training_curve(
    log_path: Path = ARTIFACT_DIR / "training_log.json",
    out_path: Path = ARTIFACT_DIR / "training_curve.png",
) -> None:
    """Plot loss (and token accuracy if present) vs training step from TRL log.

    Satisfies the hackathon minimum requirement of showing BOTH loss and reward plots.
    """
    if not log_path.exists():
        return
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return

    steps: List[int] = []
    losses: List[float] = []
    accs: List[Optional[float]] = []
    for entry in log:
        if "loss" not in entry or "step" not in entry:
            continue
        try:
            steps.append(int(entry["step"]))
            losses.append(float(entry["loss"]))
            acc = entry.get("mean_token_accuracy")
            accs.append(float(acc) if acc is not None else None)
        except Exception:
            continue

    if not steps:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, losses, marker="o", color="tab:blue", label="Training loss", linewidth=2)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    if all(a is not None for a in accs):
        ax2 = ax1.twinx()
        ax2.plot(
            steps,
            accs,
            marker="^",
            color="tab:orange",
            label="Mean token accuracy",
            linewidth=2,
        )
        ax2.set_ylabel("Mean token accuracy", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0.0, 1.05)

    plt.title("TRL SFT training curve — loss & token accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_reward_components(
    components_by_policy: Dict[str, Dict[str, float]],
    out_path: Path = ARTIFACT_DIR / "reward_components.png",
) -> None:
    """Grouped bar chart of reward-component contributions per policy.

    Visualizes the rubric-based reward signal: where each policy's reward
    actually comes from (step cost, clue bonus, handoff, mitigation, closure,
    etc.). Makes the reward design visible to judges at a glance.
    """
    if not components_by_policy:
        return

    all_keys: List[str] = []
    for comps in components_by_policy.values():
        for k in comps:
            if k not in all_keys:
                all_keys.append(k)
    if not all_keys:
        return

    policies = list(components_by_policy.keys())
    n_policies = len(policies)
    n_keys = len(all_keys)

    fig, ax = plt.subplots(figsize=(max(10, n_keys * 0.6), 6))
    bar_width = 0.8 / max(n_policies, 1)
    colors = {
        "random": "tab:red",
        "heuristic": "tab:blue",
        "base_model": "tab:orange",
        "sft_model": "tab:green",
    }
    for i, policy in enumerate(policies):
        values = [components_by_policy[policy].get(k, 0.0) for k in all_keys]
        offsets = [x + i * bar_width - 0.4 + bar_width / 2 for x in range(n_keys)]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=policy,
            color=colors.get(policy, None),
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(range(n_keys))
    ax.set_xticklabels(all_keys, rotation=35, ha="right")
    ax.set_ylabel("Summed reward contribution (all tasks)")
    ax.set_title("Where each policy earns / loses reward — rubric components")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_rewards(score_map: Dict[str, List[float]]) -> None:
    labels = ["easy", "medium", "hard"]
    x = list(range(len(labels)))
    plt.figure(figsize=(9, 5))

    style = {
        "random": ("x", "tab:red", "Random baseline"),
        "heuristic": ("o", "tab:blue", "Heuristic coordinator"),
        "base_model": ("^", "tab:orange", "Base LLM (untrained)"),
        "sft_model": ("D", "tab:green", "Fine-tuned LLM (SFT)"),
    }

    for key, (marker, color, label) in style.items():
        values = score_map.get(key) or []
        if not values or len(values) != len(labels):
            continue
        plt.plot(x, values, marker=marker, color=color, label=label, linewidth=2)

    plt.xticks(x, labels)
    plt.xlabel("Task difficulty")
    plt.ylabel("Episode total reward")
    plt.title("Incident Command Center — policy comparison")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "reward_curve.png", dpi=160)
    plt.close()


def main() -> None:
    dataset = build_training_dataset(episodes_per_task=EPISODES_PER_TASK)
    dataset.save_to_disk(str(ARTIFACT_DIR / "trl_dataset"))

    run_trl_sft(dataset)
    eval_out = evaluate_policies()
    scores: Dict[str, List[float]] = eval_out["scores"]  # type: ignore[assignment]
    components: Dict[str, Dict[str, float]] = eval_out["components"]  # type: ignore[assignment]

    plot_rewards(scores)
    plot_training_curve()
    plot_reward_components(components)

    summary = {
        "base_model": BASE_MODEL,
        "dataset_rows": len(dataset),
        "episodes_per_task": EPISODES_PER_TASK,
        "random_rewards": scores.get("random", []),
        "heuristic_rewards": scores.get("heuristic", []),
        "base_model_rewards": scores.get("base_model", []),
        "sft_model_rewards": scores.get("sft_model", []),
        "improvement_sft_over_base": [
            round(s - b, 4)
            for s, b in zip(scores.get("sft_model", []), scores.get("base_model", []))
        ] if scores.get("sft_model") and scores.get("base_model") else [],
        "improvement_heuristic_over_random": [
            round(h - r, 4)
            for h, r in zip(scores.get("heuristic", []), scores.get("random", []))
        ],
        "reward_components_by_policy": {
            policy: {k: round(v, 4) for k, v in comps.items()}
            for policy, comps in components.items()
            if comps
        },
    }
    with open(ARTIFACT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training and evaluation complete.")
    print(f"Saved artifacts in: {ARTIFACT_DIR.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
