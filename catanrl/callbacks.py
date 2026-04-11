import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CatanLoggingCallback(BaseCallback):

    def __init__(self, log_freq=10_000, reward_fn=None, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.reward_fn = reward_fn  # if PBRS, has .stats attribute
        self.start_time = None

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_won = []  # bool per episode; correct even with PBRS
        self.wins = 0
        self.losses = 0
        self.total_episodes = 0
        self._last_log_step = 0

        # Curriculum tracking
        self._stage_index = 0
        self._stage_enemies = None

        # Win rate over sliding windows for trend detection
        self._window_size = 50
        self._trend_window = 200

        # PBRS tracking
        self.ep_shaping_totals = []
        self.ep_phi_finals = []

        # Best performance tracking
        self.best_recent_wr = 0.0
        self.best_recent_wr_step = 0

        # Per-interval counters for speed tracking
        self._interval_start_time = None
        self._interval_start_step = 0

    def _on_training_start(self):
        if self.start_time is None:
            self.start_time = time.time()
            self._interval_start_time = self.start_time
            self._interval_start_step = 0
            self._print_header()

    def on_curriculum_stage(self, stage_index, stage_config):
        self._stage_index = stage_index
        self._stage_enemies = stage_config["enemies"]
        self._interval_start_time = time.time()
        self._interval_start_step = self.num_timesteps
        print(f"\n{'='*70}")
        print(f"  CURRICULUM STAGE {stage_index + 1}")
        print(f"  Enemies: {', '.join(stage_config['enemies'])}")
        print(f"  Budget: {stage_config['timesteps']:,} steps")
        print(f"{'='*70}\n")

    def _print_header(self):
        total_steps = self.locals.get("total_timesteps", 0)
        print(f"\n{'='*70}")
        print(f"  TRAINING STARTED")
        print(f"  Target: {total_steps:,} steps")
        print(f"  Logging every {self.log_freq:,} steps")
        has_pbrs = self.reward_fn is not None and hasattr(self.reward_fn, "stats")
        print(f"  Reward: {'PBRS (shaping active)' if has_pbrs else 'Sparse (+1/-1)'}")
        print(f"{'='*70}\n")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.total_episodes += 1
                # Track PBRS stats at episode end (must read before win detection
                # so last_ep_won reflects this episode's terminal outcome)
                if self.reward_fn and hasattr(self.reward_fn, "stats"):
                    stats = self.reward_fn.stats
                    won = stats.get("last_ep_won", ep_reward > 0)
                    self.ep_shaping_totals.append(stats.get("last_ep_shaping_total", 0.0))
                    self.ep_phi_finals.append(stats.get("last_ep_phi_pre_terminal", 0.0))
                else:
                    won = ep_reward > 0

                self.episode_won.append(won)
                if won:
                    self.wins += 1
                else:
                    self.losses += 1

        if self.num_timesteps - self._last_log_step >= self.log_freq:
            if self.total_episodes > 0:
                self._print_stats()
            self._last_log_step = self.num_timesteps

        return True

    def _print_stats(self):
        now = time.time()
        elapsed = now - self.start_time
        total_steps = self.locals.get("total_timesteps", 1_000_000)
        progress = self.num_timesteps / total_steps

        # Speed: use interval speed for more accurate current measurement
        interval_elapsed = now - self._interval_start_time
        interval_steps = self.num_timesteps - self._interval_start_step
        current_speed = interval_steps / interval_elapsed if interval_elapsed > 0 else 0
        avg_speed = self.num_timesteps / elapsed if elapsed > 0 else 0

        remaining = total_steps - self.num_timesteps
        eta = remaining / current_speed if current_speed > 0 else 0

        self._interval_start_time = now
        self._interval_start_step = self.num_timesteps

        # Win rates over different windows
        w = self._window_size
        recent_r = self.episode_rewards[-w:]
        recent_l = self.episode_lengths[-w:]
        recent_won = self.episode_won[-w:]
        recent_wins = sum(recent_won)
        recent_wr = recent_wins / len(recent_won) if recent_won else 0

        # Track best
        if recent_wr > self.best_recent_wr:
            self.best_recent_wr = recent_wr
            self.best_recent_wr_step = self.num_timesteps

        # Trend: compare last 50 to previous 50
        trend_str = ""
        if len(self.episode_won) >= 2 * w:
            prev_won = self.episode_won[-(2*w):-w]
            prev_wins = sum(prev_won)
            prev_wr = prev_wins / len(prev_won)
            delta = recent_wr - prev_wr
            if delta > 0.03:
                trend_str = f" [+{delta*100:.1f}pp improving]"
            elif delta < -0.03:
                trend_str = f" [{delta*100:.1f}pp declining]"
            else:
                trend_str = " [stable]"

        # Format times
        eta_h, rem = divmod(int(eta), 3600)
        eta_m, eta_s = divmod(rem, 60)
        el_h, rem = divmod(int(elapsed), 3600)
        el_m, el_s = divmod(rem, 60)

        # Progress bar
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "=" * filled + ">" + "." * (bar_width - filled - 1)

        print(f"\n[{bar}] {progress*100:.1f}%")
        if self._stage_enemies is not None:
            print(f"  Stage {self._stage_index + 1} | "
                  f"Enemies: {', '.join(self._stage_enemies)}")
        print(f"  Step {self.num_timesteps:,} / {total_steps:,} | "
              f"Ep {self.total_episodes:,}")
        print(f"  Time: {el_h}h{el_m:02d}m{el_s:02d}s elapsed | "
              f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s | "
              f"{current_speed:.0f} stp/s (avg {avg_speed:.0f})")

        overall_wr = self.wins / self.total_episodes
        print(f"  Win rate: {recent_wr*100:.1f}% (last {len(recent_r)}) | "
              f"{overall_wr*100:.1f}% (overall){trend_str}")
        print(f"  Best: {self.best_recent_wr*100:.1f}% at step {self.best_recent_wr_step:,}")
        print(f"  Avg reward: {np.mean(recent_r):.3f} | "
              f"Avg length: {np.mean(recent_l):.0f} steps")

        # PBRS diagnostics
        if self.ep_shaping_totals:
            recent_shaping = self.ep_shaping_totals[-w:]
            recent_phi = self.ep_phi_finals[-w:]
            print(f"  PBRS shaping/ep: {np.mean(recent_shaping):.3f} "
                  f"(min {np.min(recent_shaping):.3f}, max {np.max(recent_shaping):.3f})")
            if recent_phi:
                print(f"  Phi(pre-terminal): {np.mean(recent_phi):.4f} "
                      f"| range [{np.min(recent_phi):.4f}, {np.max(recent_phi):.4f}]")

        # PPO policy metrics from SB3 logger
        if hasattr(self, "logger") and self.logger is not None:
            try:
                name_to_value = self.logger.name_to_value
                metrics = {}
                for key in ["train/entropy_loss", "train/policy_gradient_loss",
                             "train/value_loss", "train/approx_kl",
                             "train/clip_fraction", "train/explained_variance"]:
                    if key in name_to_value:
                        short = key.split("/")[1]
                        metrics[short] = name_to_value[key]
                if metrics:
                    parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                    print(f"  PPO: {' | '.join(parts)}")
            except Exception:
                pass

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        el_h, rem = divmod(int(elapsed), 3600)
        el_m, el_s = divmod(rem, 60)

        overall_wr = self.wins / max(1, self.total_episodes) * 100

        print(f"\n{'#'*70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Duration: {el_h}h{el_m:02d}m{el_s:02d}s")
        print(f"  Episodes: {self.total_episodes:,} | "
              f"Steps: {self.num_timesteps:,}")
        print(f"  Final win rate: {overall_wr:.1f}% overall | "
              f"Best: {self.best_recent_wr*100:.1f}% at step {self.best_recent_wr_step:,}")

        # Final 50 vs first 50 comparison
        w = self._window_size
        if len(self.episode_won) >= 2 * w:
            first_wr = sum(self.episode_won[:w]) / w
            last_wr = sum(self.episode_won[-w:]) / w
            print(f"  Progression: {first_wr*100:.1f}% (first {w} ep) -> "
                  f"{last_wr*100:.1f}% (last {w} ep)")

        if self.ep_shaping_totals:
            print(f"  PBRS shaping/ep (final {w}): "
                  f"{np.mean(self.ep_shaping_totals[-w:]):.3f}")

        print(f"{'#'*70}")
