import argparse
import os
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

from catanrl.config import load_config
from catanrl.env_utils import make_env
from catanrl.callbacks import CatanLoggingCallback


def _print_config_summary(model, config, total_steps, enemies):
    num_players = 1 + len(enemies)
    num_params = sum(p.numel() for p in model.policy.parameters())
    env = model.get_env().envs[0]
    print(f"\n{'='*70}")
    print(f"  CATANRL TRAINING CONFIG")
    print(f"{'='*70}")
    print(f"  Environment:")
    print(f"    Players: {num_players} (1 PPO + {len(enemies)} opponents)")
    print(f"    Enemies: {', '.join(enemies)}")
    print(f"    Obs shape: {env.observation_space.shape} | Actions: {env.action_space.n}")
    print(f"  Reward:")
    print(f"    Type: {config['reward_type']}")
    if config["reward_type"] == "pbrs":
        print(f"    Lambda: {config['pbrs_lambda']} | Normalize: {config['pbrs_normalize']}")
    print(f"  Model:")
    print(f"    Net arch: {config['net_arch']} | Params: {num_params:,}")
    print(f"    LR: {config['learning_rate']} | Gamma: {config['gamma']} | "
          f"Ent coef: {config['ent_coef']}")
    print(f"    Rollout: {config['n_steps']} steps | Batch: {config['batch_size']} | "
          f"Epochs: {config['n_epochs']}")
    print(f"  Training:")
    print(f"    Total steps: {total_steps:,}")
    print(f"    Save to: {config['save_path']}.zip")
    print(f"    TB log: {config['tb_log']}")
    print(f"    Device: {model.device}")
    print(f"{'='*70}")


def _make_model(config, env):
    return MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        tensorboard_log=config["tb_log"],
        policy_kwargs=dict(net_arch=config["net_arch"]),
    )


def _merge_stage(config, stage):
    merged = dict(config)
    merged["enemies"] = stage["enemies"]
    merged["total_timesteps"] = stage["timesteps"]
    return merged


def train(config):
    os.makedirs(os.path.dirname(config["save_path"]) or ".", exist_ok=True)
    if config.get("curriculum"):
        return train_curriculum(config)
    else:
        return train_single(config)


def train_single(config):
    env = make_env(config)
    reward_fn = getattr(env, "reward_fn", None)
    model = _make_model(config, env)

    _print_config_summary(model, config, config["total_timesteps"], config["enemies"])

    callback = CatanLoggingCallback(
        log_freq=config["log_freq"],
        reward_fn=reward_fn,
    )
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)
    model.save(config["save_path"])
    env.close()

    print(f"\nModel saved to {config['save_path']}.zip")
    return model, callback


def train_curriculum(config):
    curriculum = config["curriculum"]
    total_budget = sum(s["timesteps"] for s in curriculum)

    # Build first stage env and model
    first_config = _merge_stage(config, curriculum[0])
    env = make_env(first_config)
    reward_fn = getattr(env, "reward_fn", None)
    model = _make_model(config, env)

    _print_config_summary(model, config, total_budget, curriculum[0]["enemies"])
    print(f"  Curriculum: {len(curriculum)} stages")
    for i, s in enumerate(curriculum):
        print(f"    Stage {i+1}: {s['enemies']} for {s['timesteps']:,} steps")

    callback = CatanLoggingCallback(
        log_freq=config["log_freq"],
        reward_fn=reward_fn,
    )

    cumulative_steps = 0
    for i, stage in enumerate(curriculum):
        stage_config = _merge_stage(config, stage)

        if i > 0:
            env.close()
            env = make_env(stage_config)
            reward_fn = getattr(env, "reward_fn", None)
            model.set_env(env)
            callback.reward_fn = reward_fn

        callback.on_curriculum_stage(i, stage)

        cumulative_steps += stage["timesteps"]
        model.learn(
            total_timesteps=cumulative_steps,
            callback=callback,
            reset_num_timesteps=(i == 0),
        )

        stage_path = f"{config['save_path']}_stage{i}"
        model.save(stage_path)
        print(f"  Stage {i+1} checkpoint: {stage_path}.zip")

    model.save(config["save_path"])
    env.close()
    print(f"\nFinal model saved to {config['save_path']}.zip")
    return model, callback


def main():
    parser = argparse.ArgumentParser(description="Train CatanRL agent")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config values, e.g. --override pbrs_lambda=0.5",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    train(config)


if __name__ == "__main__":
    main()
