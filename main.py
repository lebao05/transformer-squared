import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from logging_utils import Metrics
from utils import eval_model, forward, load_base_params, eval_model_experts_prompt_based


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None or str(path).lower() == "none":
        return None
    return to_absolute_path(str(path))


def resolve_model_wrapper(vllm_model: Any) -> Any:
    if hasattr(vllm_model, "llm"):
        return vllm_model.llm
    if hasattr(vllm_model, "model"):
        return vllm_model.model
    return vllm_model


def clone_trainable_params(policy: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in policy.trainable_params]


def normalize_rewards(rewards: Sequence[float], clip_value: Optional[float]) -> List[float]:
    rewards_array = np.array(rewards, dtype=float)
    if clip_value is not None:
        rewards_array = np.clip(rewards_array, -clip_value, clip_value)
    if len(rewards_array) > 1:
        mean = float(rewards_array.mean())
        std = float(rewards_array.std())
        if std > 1e-6:
            rewards_array = (rewards_array - mean) / std
    return rewards_array.tolist()


class SearchOptimizer:
    def __init__(
        self,
        policy: nn.Module,
        vllm_wrapper: Any,
        vllm_model: Any,
        task: Any,
        train_eval: Any,
        train_ids: Sequence[int],
        base_params: Dict[str, torch.Tensor],
        decomposed_params: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
    ):
        self.policy = policy
        self.vllm_wrapper = vllm_wrapper
        self.vllm_model = vllm_model
        self.task = task
        self.train_eval = train_eval
        self.train_ids = list(train_ids)
        self.base_params = base_params
        self.decomposed_params = decomposed_params
        self.batch_size = batch_size
        self.best_reward = -1e9
        self.best_params = clone_trainable_params(policy)

    def _sample_train_ids(self) -> Optional[Sequence[int]]:
        if self.batch_size is None or self.batch_size <= 0:
            return self.train_ids
        if len(self.train_ids) <= self.batch_size:
            return self.train_ids
        return random.sample(self.train_ids, min(self.batch_size, len(self.train_ids)))

    def _set_params(self, params: List[torch.Tensor]) -> None:
        self.policy.set_trainable_params_values(params)
        forward(
            policy=self.policy,
            model=self.vllm_model,
            base_params=self.base_params,
            decomposed_params=self.decomposed_params,
            learnable_params=self.policy.get_learnable_params(),
        )

    def _evaluate(self, params: List[torch.Tensor]) -> float:
        original_params = clone_trainable_params(self.policy)
        self._set_params(params)
        sample_ids = self._sample_train_ids()
        result = eval_model(self.vllm_wrapper, self.train_eval, sample_ids=sample_ids)
        rewards = self.task.get_rewards(result)
        self._set_params(original_params)
        reward_value = float(np.mean(rewards)) if len(rewards) else 0.0
        if reward_value > self.best_reward:
            self.best_reward = reward_value
            self.best_params = [p.detach().clone() for p in params]
        return reward_value

    def step(self) -> Dict[str, float]:
        raise NotImplementedError


class ReinforceOptimizer(SearchOptimizer):
    def __init__(
        self,
        *args,
        lr: float = 1e-3,
        sigma: float = 0.01,
        rw_clip: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.sigma = sigma
        self.rw_clip = rw_clip
        self.baseline = 0.0
        self.momentum = 0.9

    def step(self) -> Dict[str, float]:
        current_params = clone_trainable_params(self.policy)
        noises = [torch.randn_like(p) for p in current_params]
        candidate_params = [p + self.sigma * n for p, n in zip(current_params, noises)]
        reward = self._evaluate(candidate_params)
        reward_clipped = float(np.clip(reward, -self.rw_clip, self.rw_clip)) if self.rw_clip else reward
        advantage = reward_clipped - self.baseline
        self.baseline = self.momentum * self.baseline + (1 - self.momentum) * reward_clipped
        update = [self.lr * advantage / max(self.sigma, 1e-8) * n for n in noises]
        next_params = [p + u for p, u in zip(current_params, update)]
        self._set_params(next_params)
        return {
            "train_reward": reward,
            "baseline": float(self.baseline),
        }


class CEMOptimizer(SearchOptimizer):
    def __init__(
        self,
        *args,
        pop_size: int = 32,
        elite_ratio: float = 0.2,
        min_trainable_param: float = float("-inf"),
        max_trainable_param: float = float("inf"),
        optim_ema: float = 0.0,
        re_eval_best: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pop_size = max(1, pop_size)
        self.elite_ratio = max(0.01, min(elite_ratio, 1.0))
        self.min_param = min_trainable_param
        self.max_param = max_trainable_param
        self.optim_ema = optim_ema
        self.re_eval_best = re_eval_best
        self.mean_params = clone_trainable_params(self.policy)
        self.std_params = [torch.ones_like(p) * 0.1 for p in self.mean_params]

    def _clamp(self, params: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.clamp(p, self.min_param, self.max_param) for p in params]

    def step(self) -> Dict[str, float]:
        candidates = []
        for _ in range(self.pop_size):
            sample = [m + s * torch.randn_like(s) for m, s in zip(self.mean_params, self.std_params)]
            candidates.append(self._clamp(sample))

        scored = [(self._evaluate(sample), sample) for sample in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        elite_count = max(1, int(len(scored) * self.elite_ratio))
        elites = [sample for _, sample in scored[:elite_count]]
        stacked = [torch.stack([params[i] for params in elites], dim=0) for i in range(len(elites[0]))]
        new_mean = [stack.mean(0) for stack in stacked]
        new_std = [stack.std(0) for stack in stacked]
        if self.optim_ema:
            self.mean_params = [self.optim_ema * old + (1 - self.optim_ema) * new for old, new in zip(self.mean_params, new_mean)]
            self.std_params = [self.optim_ema * old + (1 - self.optim_ema) * new for old, new in zip(self.std_params, new_std)]
        else:
            self.mean_params = new_mean
            self.std_params = new_std

        best_reward, best_params = scored[0]
        if self.re_eval_best:
            best_reward = self._evaluate(best_params)
        self._set_params(best_params)
        return {
            "train_reward": best_reward,
            "elite_mean_reward": float(np.mean([score for score, _ in scored[:elite_count]])),
        }


class RandomShootingOptimizer(CEMOptimizer):
    def __init__(
        self,
        *args,
        pop_size: int = 32,
        min_trainable_param: float = float("-inf"),
        max_trainable_param: float = float("inf"),
        **kwargs,
    ):
        super().__init__(
            *args,
            pop_size=pop_size,
            elite_ratio=1.0,
            min_trainable_param=min_trainable_param,
            max_trainable_param=max_trainable_param,
            optim_ema=0.0,
            re_eval_best=False,
            **kwargs,
        )

    def step(self) -> Dict[str, float]:
        candidates = []
        for _ in range(self.pop_size):
            sample = [m + s * torch.randn_like(s) for m, s in zip(self.mean_params, self.std_params)]
            candidates.append(self._clamp(sample))
        best_reward = -1e9
        best_sample = None
        for sample in candidates:
            reward = self._evaluate(sample)
            if reward > best_reward:
                best_reward = reward
                best_sample = sample
        if best_sample is not None:
            self._set_params(best_sample)
        return {"train_reward": best_reward}


def build_optimizer(
    cfg: DictConfig,
    policy: nn.Module,
    vllm_wrapper: Any,
    vllm_model: Any,
    task: Any,
    train_eval: Any,
    train_ids: Sequence[int],
    base_params: Dict[str, torch.Tensor],
    decomposed_params: Dict[str, torch.Tensor],
) -> SearchOptimizer:
    target = getattr(cfg, "optimization_algorithm", None)
    if target is None or "_target_" not in target:
        raise ValueError("Missing optimization_algorithm in config")
    optimizer_name = target._target_.split(".")[-1].lower()
    kwargs = {
        "policy": policy,
        "vllm_wrapper": vllm_wrapper,
        "vllm_model": vllm_model,
        "task": task,
        "train_eval": train_eval,
        "train_ids": train_ids,
        "base_params": base_params,
        "decomposed_params": decomposed_params,
        "batch_size": int(getattr(cfg, "batch_size", 0) or 0),
    }
    if optimizer_name == "reinforce":
        return ReinforceOptimizer(
            *(),
            lr=float(getattr(cfg, "lr", 1e-3)),
            sigma=float(getattr(cfg, "sigma", 0.01)),
            rw_clip=getattr(cfg, "rw_clip", None),
            **kwargs,
        )
    if optimizer_name == "cem":
        return CEMOptimizer(
            *(),
            pop_size=int(getattr(cfg, "pop_size", 32)),
            elite_ratio=float(getattr(cfg, "elite_ratio", 0.2)),
            min_trainable_param=float(getattr(cfg, "min_trainable_param", float("-inf"))),
            max_trainable_param=float(getattr(cfg, "max_trainable_param", float("inf"))),
            optim_ema=float(getattr(cfg, "optim_ema", 0.0)),
            re_eval_best=bool(getattr(cfg, "re_eval_best", True)),
            **kwargs,
        )
    if optimizer_name == "randomshooting":
        return RandomShootingOptimizer(
            *(),
            pop_size=int(getattr(cfg, "pop_size", 32)),
            min_trainable_param=float(getattr(cfg, "min_trainable_param", float("-inf"))),
            max_trainable_param=float(getattr(cfg, "max_trainable_param", float("inf"))),
            **kwargs,
        )
    raise ValueError(f"Unsupported optimization algorithm: {optimizer_name}")


def _split_names_for_evaluators(num_evaluators: int) -> List[str]:
    if num_evaluators == 2:
        return ["train", "test"]
    if num_evaluators == 3:
        return ["train", "test", "transfer"]
    if num_evaluators == 4:
        return ["train", "valid", "test", "transfer"]
    return [f"eval_{i}" for i in range(num_evaluators)]


def evaluate_policy(
    policy: nn.Module,
    vllm_wrapper: Any,
    vllm_model: Any,
    task: Any,
    evaluators: Sequence[Any],
    base_params: Dict[str, torch.Tensor],
    decomposed_params: Dict[str, torch.Tensor],
    prefix: str = "eval",
) -> Dict[str, float]:
    metrics = {}
    forward(
        policy=policy,
        model=vllm_model,
        base_params=base_params,
        decomposed_params=decomposed_params,
        learnable_params=policy.get_learnable_params(),
    )
    split_names = _split_names_for_evaluators(len(evaluators))
    for split_name, evaluator in zip(split_names, evaluators):
        try:
            result = eval_model(vllm_wrapper, evaluator)
            task_metric = getattr(task, f"target_metric_{split_name}", None)
            if task_metric is None:
                task_metric = task.target_metric_test
            metrics[f"{prefix}/{split_name}/{task_metric}"] = float(
                result.aggregate_metrics.get(task_metric, 0.0)
            )
        except Exception:
            continue
    return metrics


def save_checkpoint(
    policy: nn.Module,
    output_dir: Path,
    step: int,
    prefix: str = "policy",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_step_{step}.pt"
    torch.save(policy.state_dict(), path)
    return path


def maybe_load_checkpoint(policy: nn.Module, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    path = resolve_path(checkpoint_path)
    if path is None:
        return
    checkpoint = torch.load(path, map_location="cpu")
    try:
        policy.load_state_dict(checkpoint)
        return
    except Exception:
        if isinstance(checkpoint, dict) and not any(k.startswith("learnable_params") for k in checkpoint):
            prefixed = {f"learnable_params.{k}": v for k, v in checkpoint.items()}
            policy.load_state_dict(prefixed)
            return
        raise


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    experiment_dir = Path(to_absolute_path(cfg.out_dir))
    experiment_dir.mkdir(parents=True, exist_ok=True)

    task = hydra.utils.instantiate(cfg.task_loader)
    base_model = hydra.utils.instantiate(cfg.base_model)
    vllm_wrapper = task.get_vllm_model(base_model.get_model_id())
    vllm_model = resolve_model_wrapper(vllm_wrapper)

    checkpoint_path = resolve_path(getattr(cfg, "load_ckpt", None))
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)

    base_model_file = resolve_path(base_model.get_param_file(getattr(cfg, "model_dir", None) or "."))
    decomposed_params = torch.load(base_model_file, map_location="cpu")
    if isinstance(decomposed_params, dict) and "decomposed_params" in decomposed_params:
        if "base_params" in decomposed_params:
            base_params = decomposed_params["base_params"]
        decomposed_params = decomposed_params["decomposed_params"]
    else:
        base_params = {}

    if not base_params:
        base_params = {}
        if hasattr(vllm_model, "named_parameters"):
            for name, param in vllm_model.named_parameters():
                base_params[name] = param.detach().cpu()
        elif hasattr(vllm_model, "params"):
            for name, param in vllm_model.params.items():
                base_params[name] = param.detach().cpu()
        else:
            raise RuntimeError("Unable to extract base model parameters from the vLLM model.")

    policy = hydra.utils.instantiate(
        cfg.shakeoff_policy,
        base_params=base_params,
        decomposed_params=decomposed_params,
        gpu=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        init_val=float(cfg.init_val),
    )
    maybe_load_checkpoint(policy, checkpoint_path)
    load_base_params(vllm_model, base_params)

    train_data, train_ix, valid_ix = task.get_train_data()
    evaluators = tuple(task.get_evaluator())
    train_eval = evaluators[0]
    optimizer = build_optimizer(
        cfg,
        policy=policy,
        vllm_wrapper=vllm_wrapper,
        vllm_model=vllm_model,
        task=task,
        train_eval=train_eval,
        train_ids=train_ix,
        base_params=base_params,
        decomposed_params=decomposed_params,
    )

    if bool(getattr(cfg, "test_only", False)):
        if bool(getattr(cfg, "prompt_based_eval", False)):
            test_eval = evaluators[1] if len(evaluators) > 1 else evaluators[0]
            metrics = eval_model_experts_prompt_based(
                vllm_wrapper,
                test_eval,
                cfg.experts_path_dict,
                policy,
                vllm_model,
                base_params,
                decomposed_params,
                task.target_metric_test,
            )
        else:
            metrics = evaluate_policy(
                policy=policy,
                vllm_wrapper=vllm_wrapper,
                vllm_model=vllm_model,
                task=task,
                evaluators=evaluators,
                base_params=base_params,
                decomposed_params=decomposed_params,
            )
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")
        return

    print(f"Starting training for {int(cfg.num_iters)} iterations...")
    metrics_logger = Metrics("train_reward", "best_reward")
    best_metric = -1e9
    best_checkpoint = None
    for step in range(1, int(cfg.num_iters) + 1):
        train_stats = optimizer.step()
        train_stats["best_reward"] = float(optimizer.best_reward)
        metrics_logger.update(**train_stats)

        if step % int(getattr(cfg, "test_interval", 10)) == 0 or step == 1:
            if bool(getattr(cfg, "prompt_based_eval", False)):
                test_eval = evaluators[1] if len(evaluators) > 1 else evaluators[0]
                metrics = eval_model_experts_prompt_based(
                    vllm_wrapper,
                    test_eval,
                    cfg.experts_path_dict,
                    policy,
                    vllm_model,
                    base_params,
                    decomposed_params,
                    task.target_metric_test,
                )
            else:
                metrics = evaluate_policy(
                    policy=policy,
                    vllm_wrapper=vllm_wrapper,
                    vllm_model=vllm_model,
                    task=task,
                    evaluators=evaluators,
                    base_params=base_params,
                    decomposed_params=decomposed_params,
                )
            metrics.update({k: float(v) for k, v in metrics_logger.get().items()})
            print(f"Step {step}: " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
            if metrics.get("eval/test/" + task.target_metric_test, -1e9) > best_metric:
                best_metric = metrics.get("eval/test/" + task.target_metric_test, best_metric)
                best_checkpoint = save_checkpoint(policy, experiment_dir, step)
                print(f"Saved best checkpoint to {best_checkpoint}")

    final_ckpt = save_checkpoint(policy, experiment_dir, int(cfg.num_iters))
    print(f"Training complete. Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
