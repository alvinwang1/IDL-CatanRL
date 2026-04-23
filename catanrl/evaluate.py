import argparse
import time
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

import gymnasium
from sb3_contrib.ppo_mask import MaskablePPO

import catanatron_gym
from catanatron import Color, Game

from catanrl.config import load_config
from catanrl.env_utils import build_enemies, mask_fn, ENEMY_COLORS
from catanrl.ppo_player import PPOPlayer


def wilson_ci(wins, total, z=1.96):
    wr = wins / total
    d = 1 + z**2 / total
    center = (wr + z**2 / (2 * total)) / d
    spread = z * np.sqrt((wr * (1 - wr) + z**2 / (4 * total)) / total) / d
    return max(0, center - spread), min(1, center + spread)


def evaluate_gym(config, model_path):
    """Evaluate via gym environment (for Random/WeightedRandom opponents)."""
    model = MaskablePPO.load(model_path)
    enemies = build_enemies(config["enemies"])

    eval_env = gymnasium.make(
        "catanatron-v1",
        config={"enemies": enemies, "vps_to_win": config.get("vps_to_win", 10)},
    )

    n_games = config["eval_games"]
    deterministic = config.get("eval_deterministic", True)
    wins, losses, draws = 0, 0, 0
    ep_lengths = []

    print(f"Evaluating {n_games} games via gym env...")
    print(f"  Model: {model_path}")
    print(f"  Enemies: {config['enemies']}")
    start = time.time()

    for i in range(n_games):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            valid = eval_env.unwrapped.get_valid_actions()
            action_mask = np.zeros(eval_env.action_space.n, dtype=bool)
            if len(valid) > 0:
                action_mask[valid] = True
            else:
                action_mask[:] = True

            action, _ = model.predict(obs, action_masks=action_mask, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        ep_lengths.append(steps)
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            eta = (n_games - i - 1) / ((i + 1) / elapsed)
            print(f"  [{i+1}/{n_games}] Win rate: {100*wins/(i+1):.1f}% | ETA: {eta:.0f}s")

    eval_env.close()
    elapsed = time.time() - start

    wr = wins / n_games
    ci_low, ci_high = wilson_ci(wins, n_games)
    random_chance = 100.0 / (1 + len(config["enemies"]))

    print(f"\n{'='*60}")
    print(f"  RESULTS ({n_games} games, {elapsed:.0f}s)")
    print(f"  Wins: {wins} | Losses: {losses} | Draws: {draws}")
    print(f"  Win rate: {100*wr:.1f}%  (random chance: {random_chance:.0f}%)")
    print(f"  95% CI:   [{100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    print(f"  Avg episode length: {np.mean(ep_lengths):.0f}")
    print(f"{'='*60}")


def evaluate_game_api(config, model_path):
    """Evaluate via Game API directly (for AlphaBeta/ValueFunction opponents)."""
    num_players = 1 + len(config["enemies"])
    enemies = build_enemies(config["enemies"])
    ppo = PPOPlayer(Color.BLUE, model_path, num_players=num_players,
                    deterministic=config.get("eval_deterministic", True))
    players = [ppo] + enemies

    n_games = config["eval_games"]
    wins = 0

    print(f"Evaluating {n_games} games via Game API...")
    print(f"  Model: {model_path}")
    print(f"  Enemies: {config['enemies']}")
    start = time.time()

    for i in range(n_games):
        game = Game(players=players, vps_to_win=config.get("vps_to_win", 10))
        winner = game.play()
        if winner == Color.BLUE:
            wins += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            eta = (n_games - i - 1) / ((i + 1) / elapsed)
            print(f"  [{i+1}/{n_games}] Win rate: {100*wins/(i+1):.1f}% | ETA: {eta:.0f}s")

    elapsed = time.time() - start

    wr = wins / n_games
    ci_low, ci_high = wilson_ci(wins, n_games)
    random_chance = 100.0 / num_players

    print(f"\n{'='*60}")
    print(f"  RESULTS ({n_games} games, {elapsed:.0f}s)")
    print(f"  Wins: {wins} / {n_games}")
    print(f"  Win rate: {100*wr:.1f}%  (random chance: {random_chance:.0f}%)")
    print(f"  95% CI:   [{100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    print(f"{'='*60}")


def needs_game_api(config):
    advanced = {"AlphaBeta", "ValueFunction"}
    return any(e in advanced for e in config["enemies"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate CatanRL agent")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config values",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.override)

    if needs_game_api(config):
        evaluate_game_api(config, args.model)
    else:
        evaluate_gym(config, args.model)


if __name__ == "__main__":
    main()
