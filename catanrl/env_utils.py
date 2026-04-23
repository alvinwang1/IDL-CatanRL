import gymnasium
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

import catanatron_gym
from catanatron import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanrl.catanatron_ext.minimax import AlphaBetaPlayer
from catanrl.catanatron_ext.value import ValueFunctionPlayer
from catanrl.reward import build_reward


ENEMY_COLORS = [Color.RED, Color.ORANGE, Color.WHITE]

ENEMY_CONSTRUCTORS = {
    "Random": lambda c: RandomPlayer(c),
    "WeightedRandom": lambda c: WeightedRandomPlayer(c),
    "VictoryPoint": lambda c: __import__(
        "catanatron.players.search", fromlist=["VictoryPointPlayer"]
    ).VictoryPointPlayer(c),
    "ValueFunction": lambda c: ValueFunctionPlayer(c),
    "AlphaBeta": lambda c: AlphaBetaPlayer(c),
}


def build_enemies(enemy_names):
    enemies = []
    for i, name in enumerate(enemy_names):
        if name not in ENEMY_CONSTRUCTORS:
            raise ValueError(
                f"Unknown enemy type: {name}. "
                f"Available: {list(ENEMY_CONSTRUCTORS.keys())}"
            )
        enemies.append(ENEMY_CONSTRUCTORS[name](ENEMY_COLORS[i]))
    return enemies


def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    if len(valid_actions) > 0:
        mask[valid_actions] = True
    else:
        mask[:] = True
    return mask


def make_env(config):
    enemies = build_enemies(config["enemies"])
    reward_fn = build_reward(config)
    env = gymnasium.make(
        "catanatron-v1",
        config={
            "enemies": enemies,
            "reward_function": reward_fn,
            "vps_to_win": config.get("vps_to_win", 10),
        },
    )
    wrapped = ActionMasker(env, mask_fn)
    wrapped.reward_fn = reward_fn  # expose for callback logging
    return wrapped
