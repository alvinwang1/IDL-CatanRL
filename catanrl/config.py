import yaml
import os

DEFAULTS = {
    # Environment
    "enemies": ["WeightedRandom", "WeightedRandom", "WeightedRandom"],
    "vps_to_win": 10,
    # Reward
    "reward_type": "sparse",
    "pbrs_lambda": 1.0,
    "pbrs_normalize": True,
    "pbrs_phi_cap": None,  # float or None; caps Phi(s_pre_terminal) at terminal step
    "vp_scale": 0.0,      # float; extra reward per VP gained (0.0 = disabled)
    # Model
    "net_arch": [256, 256],
    "learning_rate": 3e-4,
    "lr_schedule": "constant",  # "constant" or "linear"
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.999,
    "ent_coef": 0.01,
    # Training
    "total_timesteps": 1_000_000,
    "save_path": "models/default",
    "tb_log": "logs/default",
    "log_freq": 10_000,
    # Evaluation
    "eval_games": 500,
    "eval_deterministic": True,
    # Curriculum (None = single-stage training)
    "curriculum": None,
    # Resume from an existing model checkpoint (path without .zip)
    "resume_from": None,
}


def load_config(path=None, overrides=None):
    config = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            file_config = yaml.safe_load(f) or {}
        config.update(file_config)
    if overrides:
        for kv in overrides:
            key, value = kv.split("=", 1)
            config[key] = _parse_value(value)
    _coerce_types(config)
    return config


# Keys that must be specific types (YAML sometimes parses e.g. 3e-4 as string)
_FLOAT_KEYS = {"learning_rate", "gamma", "ent_coef", "pbrs_lambda", "pbrs_phi_cap", "vp_scale"}
_INT_KEYS = {"n_steps", "batch_size", "n_epochs", "total_timesteps", "log_freq",
             "eval_games", "vps_to_win"}


def _coerce_types(config):
    for key in _FLOAT_KEYS:
        if key in config and isinstance(config[key], str):
            config[key] = float(config[key])
    for key in _INT_KEYS:
        if key in config and isinstance(config[key], str):
            config[key] = int(float(config[key]))
    if config.get("curriculum"):
        for stage in config["curriculum"]:
            if "timesteps" in stage and isinstance(stage["timesteps"], str):
                stage["timesteps"] = int(float(stage["timesteps"]))


def _parse_value(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(v.strip().strip("'\"")) for v in inner.split(",")]
    return value
