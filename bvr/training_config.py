"""Training / PPO / network settings.

Defaults match the locked classroom contract in policy.py. The online platform
admin can override these via platform config (students never see them).
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

# Locked classroom defaults (single source of truth — policy.py re-exports these).
POLICY_KWARGS = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Flat platform-config keys (stored in SQLite) → default string values.
PLATFORM_DEFAULTS: Dict[str, str] = {
    "train_seed": "0",
    "max_cycles": "260",
    "train_device": "cpu",
    "policy_hidden_size": "128",
    "policy_n_layers": "2",
    "ppo_learning_rate": "0.0003",
    "ppo_n_steps": "2048",
    "ppo_batch_size": "256",
    "ppo_n_epochs": "10",
    "ppo_gamma": "0.99",
    "ppo_gae_lambda": "0.95",
    "ppo_clip_range": "0.2",
    "ppo_ent_coef": "0.0",
    "ppo_vf_coef": "0.5",
    "ppo_max_grad_norm": "0.5",
}

PLATFORM_KEYS = tuple(PLATFORM_DEFAULTS.keys())

_BOUNDS = {
    "train_seed": (0, 999_999),
    "max_cycles": (50, 2000),
    "policy_hidden_size": (32, 512),
    "policy_n_layers": (1, 4),
    "ppo_n_steps": (256, 8192),
    "ppo_batch_size": (32, 2048),
    "ppo_n_epochs": (1, 50),
    "ppo_gamma": (0.9, 0.999),
    "ppo_gae_lambda": (0.8, 1.0),
    "ppo_clip_range": (0.05, 0.5),
    "ppo_ent_coef": (0.0, 0.1),
    "ppo_vf_coef": (0.1, 1.0),
    "ppo_max_grad_norm": (0.1, 5.0),
    "ppo_learning_rate": (1e-5, 1e-2),
}


def _f(cfg: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(cfg: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(cfg.get(key, default)))
    except (TypeError, ValueError):
        return default


def _clamp(key: str, value: float) -> float:
    lo, hi = _BOUNDS.get(key, (value, value))
    return max(lo, min(hi, value))


def sanitize_platform_updates(updates: Dict[str, str]) -> Dict[str, str]:
    """Validate and clamp admin training-config values before persisting."""
    out: Dict[str, str] = {}
    for key, raw in updates.items():
        if key not in PLATFORM_DEFAULTS:
            continue
        if key == "train_device":
            dev = str(raw).strip().lower()
            out[key] = dev if dev in ("cpu", "cuda", "auto") else "cpu"
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            out[key] = PLATFORM_DEFAULTS[key]
            continue
        if key in ("train_seed", "max_cycles", "policy_hidden_size", "policy_n_layers",
                   "ppo_n_steps", "ppo_batch_size", "ppo_n_epochs"):
            out[key] = str(int(_clamp(key, num)))
        else:
            out[key] = str(_clamp(key, num))
    return out


def training_block_from_platform(platform_cfg: Dict[str, str]) -> Dict:
    """Build the `training` object written into per-run scenario.json."""
    cfg = {**PLATFORM_DEFAULTS, **(platform_cfg or {})}
    hidden = _i(cfg, "policy_hidden_size", 128)
    n_layers = _i(cfg, "policy_n_layers", 2)
    layers = [hidden] * max(1, n_layers)
    return {
        "device": cfg.get("train_device", "cpu"),
        "policy_kwargs": {"net_arch": {"pi": layers, "vf": layers}},
        "hyperparams": {
            "learning_rate": _f(cfg, "ppo_learning_rate", 3e-4),
            "n_steps": _i(cfg, "ppo_n_steps", 2048),
            "batch_size": _i(cfg, "ppo_batch_size", 256),
            "n_epochs": _i(cfg, "ppo_n_epochs", 10),
            "gamma": _f(cfg, "ppo_gamma", 0.99),
            "gae_lambda": _f(cfg, "ppo_gae_lambda", 0.95),
            "clip_range": _f(cfg, "ppo_clip_range", 0.2),
            "ent_coef": _f(cfg, "ppo_ent_coef", 0.0),
            "vf_coef": _f(cfg, "ppo_vf_coef", 0.5),
            "max_grad_norm": _f(cfg, "ppo_max_grad_norm", 0.5),
        },
    }


def resolve_training(training: Optional[Dict] = None) -> Tuple[Dict, Dict, str]:
    """Return (hyperparams, policy_kwargs, device) for make_model."""
    if not training:
        device = "cpu"
        return copy.deepcopy(PPO_HYPERPARAMS), copy.deepcopy(POLICY_KWARGS), device

    hp = {**PPO_HYPERPARAMS, **(training.get("hyperparams") or {})}
    pk = copy.deepcopy(POLICY_KWARGS)
    custom_pk = training.get("policy_kwargs") or {}
    if custom_pk.get("net_arch"):
        pk["net_arch"] = copy.deepcopy(custom_pk["net_arch"])
    device = str(training.get("device", "cpu")).lower()
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if device not in ("cpu", "cuda"):
        device = "cpu"
    return hp, pk, device


def training_defaults_payload() -> Dict:
    """Reference defaults for the admin UI."""
    return {
        "platform_keys": PLATFORM_KEYS,
        "defaults": dict(PLATFORM_DEFAULTS),
        "bounds": {k: {"min": v[0], "max": v[1]} for k, v in _BOUNDS.items()},
        "locked_baseline": {
            "hyperparams": PPO_HYPERPARAMS,
            "policy_kwargs": POLICY_KWARGS,
            "device": "cpu",
        },
    }
