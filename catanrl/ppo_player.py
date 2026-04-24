import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron.models.player import Player
from catanatron.game import Game
from catanatron_gym.features import create_sample, get_feature_ordering
from catanatron_gym.envs.catanatron_env import (
    to_action_space,
    from_action_space,
    ACTION_SPACE_SIZE,
)


class PPOPlayer(Player):
    """Wraps a trained MaskablePPO model as a Catanatron Player for Game API use."""

    def __init__(self, color, model_path, num_players=4, deterministic=True):
        super().__init__(color, is_bot=True)
        from catanrl.env_utils import ensure_model_zip
        self.model = MaskablePPO.load(ensure_model_zip(str(model_path)) + ".zip")
        self.deterministic = deterministic
        self.features = get_feature_ordering(num_players)

    @classmethod
    def from_model(cls, color, model, num_players=4, deterministic=True):
        """Construct a PPOPlayer from an already-loaded MaskablePPO model."""
        player = cls.__new__(cls)
        Player.__init__(player, color, is_bot=True)
        player.model = model
        player.deterministic = deterministic
        player.features = get_feature_ordering(num_players)
        return player

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        sample = create_sample(game, self.color)
        obs = np.array([float(sample.get(f, 0.0)) for f in self.features])

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for action in playable_actions:
            try:
                idx = to_action_space(action)
                mask[idx] = True
            except (ValueError, IndexError):
                pass
        if not mask.any():
            mask[:] = True

        action_int, _ = self.model.predict(
            obs, action_masks=mask, deterministic=self.deterministic
        )
        return from_action_space(int(action_int), playable_actions)
