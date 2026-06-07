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

from typing import Optional

from stable_baselines3 import PPO

# Fixed actor/critic architecture (two hidden layers of 128 units, tanh).
POLICY_KWARGS = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

# Fixed PPO hyperparameters. These are part of the "locked" contract.
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


def make_model(env, seed: Optional[int] = 0, tensorboard_log: Optional[str] = None,
               verbose: int = 0) -> PPO:
    """Build the locked PPO model around a given (vectorized) environment."""
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device="cpu",
        **PPO_HYPERPARAMS,
    )
