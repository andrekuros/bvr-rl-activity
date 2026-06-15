"""LOCKED model definition.

================================ DO NOT EDIT ================================
For the class competition to be fair, every student must train the *same*
network with the *same* PPO hyperparameters. The only thing you are allowed to
change is the reward configuration in `config/rewards.json` (and the enemy mix
in `config/scenario.json`). Editing anything in this file or the hyperparameters
in `train.py` will make your submission ineligible.
=============================================================================
"""

from __future__ import annotations

from typing import Dict, Optional

from stable_baselines3 import PPO

from .training_config import PPO_HYPERPARAMS, POLICY_KWARGS, resolve_training


def make_model(env, seed: Optional[int] = 0, tensorboard_log: Optional[str] = None,
               verbose: int = 0, training: Optional[Dict] = None) -> PPO:
    """Build the PPO model. Uses locked defaults unless `training` overrides (admin)."""
    hp, pk, device = resolve_training(training)
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=pk,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device=device,
        **hp,
    )
