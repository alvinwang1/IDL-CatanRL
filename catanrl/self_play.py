"""Population-based self-play callback for CatanRL.

Maintains a pool of frozen model snapshots and periodically:
  1. Saves a snapshot of the current model to the pool.
  2. Swaps the training opponents to a fresh sample from the pool,
     mixing PPO snapshots with WeightedRandom bots for diversity.

Usage — add to your config:

    self_play:
      snapshot_dir: "snapshots/self_play_4p"
      snapshot_freq: 100_000      # steps between snapshot saves
      swap_freq:     100_000      # steps between opponent swaps
      ppo_prob:      0.7          # fraction of opponents drawn from pool
      seed_models:                # pre-seed pool before training starts
        - "vp_ablation_stage2"
"""

import glob
import os
import random
import shutil

from stable_baselines3.common.callbacks import BaseCallback
from catanrl.env_utils import _PPO_PREFIX, ensure_model_zip


class SelfPlayCallback(BaseCallback):
    """Snapshot-pool self-play: save checkpoints and rotate opponents."""

    def __init__(
        self,
        config: dict,
        snapshot_dir: str,
        snapshot_freq: int = 100_000,
        swap_freq: int = 100_000,
        ppo_prob: float = 0.7,
        seed_models: list = None,
        logging_callback=None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.config = config
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = snapshot_freq
        self.swap_freq = swap_freq
        self.ppo_prob = ppo_prob
        self.logging_callback = logging_callback

        self._last_snapshot_step = None  # set in _on_training_start
        self._last_swap_step = None
        self._swap_count = 0

        os.makedirs(snapshot_dir, exist_ok=True)

        for path in (seed_models or []):
            self._add_to_pool(path)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _add_to_pool(self, model_path: str):
        """Copy a model into the snapshot pool as a .zip file."""
        src = str(model_path).rstrip("/")
        name = os.path.basename(src)
        if name.endswith(".zip"):
            name = name[:-4]

        dst_zip = os.path.join(self.snapshot_dir, name + ".zip")
        if os.path.exists(dst_zip):
            return

        # ensure_model_zip creates src.zip from a directory if needed.
        ensure_model_zip(src)
        src_zip = src if src.endswith(".zip") else src + ".zip"

        if os.path.isfile(src_zip):
            shutil.copy2(src_zip, dst_zip)
        else:
            if self.verbose:
                print(f"[SelfPlay] WARNING: seed model not found: {src}")
            return

        if self.verbose:
            print(f"[SelfPlay] Seeded pool with: {dst_zip}")

    def _pool_zips(self) -> list:
        return glob.glob(os.path.join(self.snapshot_dir, "*.zip"))

    # ------------------------------------------------------------------
    # Snapshot saving
    # ------------------------------------------------------------------

    def _save_snapshot(self):
        path = os.path.join(self.snapshot_dir, f"snap_{self.num_timesteps}")
        self.model.save(path)
        from catanrl.env_utils import _PPO_MODEL_CACHE
        _PPO_MODEL_CACHE.pop(os.path.abspath(path), None)
        if self.verbose:
            print(f"[SelfPlay] Snapshot saved → {path}.zip")

    # ------------------------------------------------------------------
    # Opponent swapping
    # ------------------------------------------------------------------

    def _swap_opponents(self):
        import numpy as np
        from catanrl.env_utils import make_env

        pool = self._pool_zips()
        
        n_enemies = len(self.config.get("enemies", []))
        new_enemies = []
        
        # Chance to pick specific benchmark bots if configured
        benchmark_bots = self.config.get("self_play", {}).get("benchmark_bots", [])
        
        for _ in range(n_enemies):
            r = random.random()
            if r < self.ppo_prob and pool:
                new_enemies.append(f"{_PPO_PREFIX}{random.choice(pool)[:-4]}")
            elif benchmark_bots and random.random() < 0.5: # 50% of non-PPO are benchmarks
                new_enemies.append(random.choice(benchmark_bots))
            else:
                new_enemies.append("WeightedRandom")

        new_env = make_env({**self.config, "enemies": new_enemies})
        # force_reset=True (default) clears _last_obs; reset immediately to
        # repopulate it so the next rollout collection doesn't assert-fail.
        self.model.set_env(new_env)
        self.model._last_obs = self.model.env.reset()
        self.model._last_episode_starts = np.ones(
            (self.model.env.num_envs,), dtype=bool
        )

        if self.logging_callback is not None:
            self.logging_callback.reward_fn = getattr(new_env, "reward_fn", None)

        self._swap_count += 1
        if self.verbose:
            def _label(e):
                if e.startswith(_PPO_PREFIX):
                    return e.split("snap_")[-1]
                return e
            labels = [_label(e) for e in new_enemies]
            print(f"[SelfPlay] Swap #{self._swap_count} → {labels}")

    # ------------------------------------------------------------------
    # Callback hooks — trigger only between complete rollouts
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        # Initialize from current step so resumed runs don't fire immediately.
        self._last_snapshot_step = self.num_timesteps
        self._last_swap_step = self.num_timesteps

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps

        if steps - self._last_snapshot_step >= self.snapshot_freq:
            self._save_snapshot()
            self._last_snapshot_step = steps

        if steps - self._last_swap_step >= self.swap_freq:
            self._swap_opponents()
            self._last_swap_step = steps

    def _on_step(self) -> bool:
        return True
