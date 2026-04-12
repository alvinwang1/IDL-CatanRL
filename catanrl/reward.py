from catanrl.catanatron_ext.value import base_fn, DEFAULT_WEIGHTS

VP_WEIGHT = DEFAULT_WEIGHTS["public_vps"]  # 3e14


def sparse_reward(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is None:
        return 0
    else:
        return -1


def make_pbrs_reward(gamma=0.999, lam=1.0, normalize=True, phi_cap=None):
    """Create a PBRS reward function using Catanatron's hand-crafted value function.

    The shaped reward is: r_sparse + lam * (gamma * Phi(s') - Phi(s))
    where Phi(s) = base_fn(game, color), optionally normalized.
    Terminal states have Phi = 0 (PBRS requirement).

    phi_cap: if set, clamps Phi(s_pre_terminal) to this value before applying
    the terminal correction. Without a cap, a strong pre-terminal board (high Phi)
    creates a large negative terminal shaping that can make wins total-negative.
    E.g. phi_cap=3.0 ensures wins always yield at least +1 - lam*3.0 total reward.

    Exposes per-step diagnostics via reward_function.stats dict.
    """
    value_fn = base_fn(DEFAULT_WEIGHTS)
    state = {"phi_prev": None}
    # Exposed stats for logging (updated every call)
    # "last_ep_*" fields hold completed-episode data for the callback to read
    stats = {
        "phi": 0.0,
        "shaping": 0.0,
        "sparse": 0.0,
        "total": 0.0,
        "last_ep_won": False,
        "last_ep_shaping_total": 0.0,
        "last_ep_phi_max": 0.0,
        "last_ep_phi_pre_terminal": 0.0,
        "_ep_phi_values": [],
        "_ep_shaping_values": [],
    }

    def reward_function(game, p0_color):
        winning_color = game.winning_color()

        if winning_color is not None:
            # Terminal state: Phi(terminal) = 0 for PBRS correctness.
            # Apply phi_cap to prevent a strong pre-terminal board from making
            # the terminal correction so negative that wins total to < 0.
            terminal_reward = 1.0 if winning_color == p0_color else -1.0
            phi_prev = state["phi_prev"] if state["phi_prev"] is not None else 0.0
            if phi_cap is not None:
                phi_prev = min(phi_prev, phi_cap)
            shaping = gamma * 0.0 - phi_prev
            state["phi_prev"] = None
            total = terminal_reward + lam * shaping
            stats["phi"] = 0.0
            stats["shaping"] = lam * shaping
            stats["sparse"] = terminal_reward
            stats["total"] = total
            # Snapshot completed episode stats before clearing
            ep_shaping = stats["_ep_shaping_values"]
            ep_phi = stats["_ep_phi_values"]
            stats["last_ep_won"] = terminal_reward > 0
            stats["last_ep_shaping_total"] = sum(ep_shaping) + lam * shaping
            stats["last_ep_phi_max"] = max(ep_phi) if ep_phi else 0.0
            stats["last_ep_phi_pre_terminal"] = ep_phi[-1] if ep_phi else 0.0
            stats["_ep_phi_values"] = []
            stats["_ep_shaping_values"] = []
            return total

        phi_current = value_fn(game, p0_color)
        if normalize:
            phi_current = phi_current / VP_WEIGHT

        stats["phi"] = phi_current
        stats["_ep_phi_values"].append(phi_current)

        if state["phi_prev"] is None:
            # First step of episode: no shaping yet
            state["phi_prev"] = phi_current
            stats["shaping"] = 0.0
            stats["sparse"] = 0.0
            stats["total"] = 0.0
            return 0.0

        shaping = gamma * phi_current - state["phi_prev"]
        state["phi_prev"] = phi_current
        total = lam * shaping
        stats["shaping"] = lam * shaping
        stats["sparse"] = 0.0
        stats["total"] = total
        stats["_ep_shaping_values"].append(lam * shaping)
        return total

    reward_function.stats = stats
    return reward_function


def build_reward(config):
    reward_type = config.get("reward_type", "sparse")
    if reward_type == "sparse":
        return sparse_reward
    elif reward_type == "pbrs":
        return make_pbrs_reward(
            gamma=config.get("gamma", 0.999),
            lam=config.get("pbrs_lambda", 1.0),
            normalize=config.get("pbrs_normalize", True),
            phi_cap=config.get("pbrs_phi_cap", None),
        )
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
