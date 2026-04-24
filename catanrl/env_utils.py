import os
import random as _random
import zipfile
import gymnasium
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

import catanatron_gym
from catanatron import Color, RandomPlayer
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanrl.catanatron_ext.minimax import AlphaBetaPlayer
from catanrl.catanatron_ext.value import ValueFunctionPlayer
from catanrl.reward import build_reward


def _make_safe_vp(color):
    from catanatron.players.search import VictoryPointPlayer

    class _SafeVP(VictoryPointPlayer):
        # catanatron's VictoryPointPlayer.decide() has a rare bug where it
        # indexes connected_components with a None key during tree search.
        # Fall back to a random legal action when that happens.
        def decide(self, game, playable_actions):
            try:
                return super().decide(game, playable_actions)
            except (TypeError, IndexError, KeyError):
                return _random.choice(playable_actions)

    return _SafeVP(color)


def _make_safe_vf(color):
    class _SafeVF(ValueFunctionPlayer):
        # ValueFunctionPlayer also does game_copy.execute(action) and can
        # hit the same catanatron board bug as VictoryPointPlayer.
        def decide(self, game, playable_actions):
            try:
                return super().decide(game, playable_actions)
            except (TypeError, IndexError, KeyError):
                return _random.choice(playable_actions)

    return _SafeVF(color)


ENEMY_COLORS = [Color.RED, Color.ORANGE, Color.WHITE]

ENEMY_CONSTRUCTORS = {
    "Random": lambda c: RandomPlayer(c),
    "WeightedRandom": lambda c: WeightedRandomPlayer(c),
    "VictoryPoint": _make_safe_vp,
    "ValueFunction": _make_safe_vf,
    "AlphaBeta": lambda c: AlphaBetaPlayer(c),
}

_PPO_PREFIX = "PPO:"

# Keyed by absolute path; evicted FIFO when size exceeds _MAX_CACHE_SIZE.
_PPO_MODEL_CACHE: dict = {}
_MAX_CACHE_SIZE = 20


def ensure_model_zip(path: str) -> str:
    """Return a path SB3 can load from.

    SB3's load() requires a .zip archive. If *path* points to an unzipped
    directory (the format produced by some save calls), we zip it in-place so
    that path + '.zip' exists, then return *path* (SB3 appends .zip itself).
    """
    path = str(path).rstrip("/")
    if path.endswith(".zip"):
        path = path[:-4]
    if os.path.isfile(path + ".zip"):
        return path
    if os.path.isdir(path):
        zip_dst = path + ".zip"
        with zipfile.ZipFile(zip_dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(path):
                zf.write(os.path.join(path, fname), fname)
        return path
    return path


def _load_ppo_player(color, model_path: str, num_players: int = 4):
    """Build a PPOPlayer, reusing a cached MaskablePPO model if available."""
    from catanrl.ppo_player import PPOPlayer
    from sb3_contrib.ppo_mask import MaskablePPO

    clean = ensure_model_zip(str(model_path))
    abs_path = os.path.abspath(clean)

    if abs_path not in _PPO_MODEL_CACHE:
        if len(_PPO_MODEL_CACHE) >= _MAX_CACHE_SIZE:
            _PPO_MODEL_CACHE.pop(next(iter(_PPO_MODEL_CACHE)))
        # Pass the explicit .zip path — SB3 2.4 raises IsADirectoryError
        # (not FileNotFoundError) when given a bare directory path, so its
        # fallback that appends ".zip" never fires.
        _PPO_MODEL_CACHE[abs_path] = MaskablePPO.load(clean + ".zip")

    return PPOPlayer.from_model(color, _PPO_MODEL_CACHE[abs_path], num_players)


def build_enemies(enemy_names, num_players: int = 4):
    enemies = []
    for i, name in enumerate(enemy_names):
        color = ENEMY_COLORS[i]
        if name.startswith(_PPO_PREFIX):
            model_path = name[len(_PPO_PREFIX):]
            enemies.append(_load_ppo_player(color, model_path, num_players))
        elif name in ENEMY_CONSTRUCTORS:
            enemies.append(ENEMY_CONSTRUCTORS[name](color))
        else:
            raise ValueError(
                f"Unknown enemy type: {name!r}. "
                f"Available: {list(ENEMY_CONSTRUCTORS.keys())} or PPO:<path>"
            )
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
    num_players = 1 + len(config.get("enemies", []))
    enemies = build_enemies(config["enemies"], num_players=num_players)
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
