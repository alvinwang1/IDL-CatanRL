import argparse
import os
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

from catanrl.config import load_config
from catanrl.env_utils import make_env, ensure_model_zip
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
        cap = config.get("pbrs_phi_cap")
        cap_str = f" | Phi cap: {cap}" if cap is not None else ""
        print(f"    Lambda: {config['pbrs_lambda']} | Normalize: {config['pbrs_normalize']}{cap_str}")
    print(f"  Model:")
    print(f"    Net arch: {config['net_arch']} | Params: {num_params:,}")
    lr_str = f"{config['learning_rate']} ({config.get('lr_schedule','constant')})"
    print(f"    LR: {lr_str} | Gamma: {config['gamma']} | "
          f"Ent coef: {config['ent_coef']}")
    print(f"    Rollout: {config['n_steps']} steps | Batch: {config['batch_size']} | "
          f"Epochs: {config['n_epochs']}")
    print(f"  Training:")
    print(f"    Total steps: {total_steps:,}")
    print(f"    Save to: {config['save_path']}.zip")
    print(f"    TB log: {config['tb_log']}")
    print(f"    Device: {model.device}")
    print(f"{'='*70}")


def _make_lr(config):
    """Return a learning rate schedule callable or constant for SB3."""
    base_lr = config["learning_rate"]
    schedule = config.get("lr_schedule", "constant")
    if schedule == "linear":
        # progress_remaining: 1.0 at start → 0.0 at end
        return lambda p: base_lr * p
    return base_lr  # SB3 accepts a float directly


def _make_model(config, env):
    return MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,
        learning_rate=_make_lr(config),
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
    # Allow per-stage overrides for any key defined in the stage block
    for key in ("vps_to_win", "pbrs_lambda", "pbrs_phi_cap", "ent_coef", "vp_scale"):
        if key in stage:
            merged[key] = stage[key]
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

    resume_from = config.get("resume_from")
    if resume_from:
        # ensure_model_zip converts directory-format saves to .zip if needed.
        clean_resume = ensure_model_zip(resume_from) + ".zip"
        model = MaskablePPO.load(
            clean_resume,
            env=env,
            learning_rate=_make_lr(config),
            tensorboard_log=config["tb_log"],
        )
        reset_steps = False
        print(f"\nResuming from: {resume_from}")
    else:
        model = _make_model(config, env)
        reset_steps = True

    _print_config_summary(model, config, config["total_timesteps"], config["enemies"])

    log_callback = CatanLoggingCallback(
        log_freq=config["log_freq"],
        reward_fn=reward_fn,
    )

    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_cb = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 2_000_000),
        save_path=os.path.dirname(config["save_path"]) or ".",
        name_prefix=os.path.basename(config["save_path"]),
        verbose=1,
    )
    callbacks = [log_callback, checkpoint_cb]

    sp_cfg = config.get("self_play")
    if sp_cfg:
        from catanrl.self_play import SelfPlayCallback
        sp = SelfPlayCallback(
            config=config,
            snapshot_dir=sp_cfg["snapshot_dir"],
            snapshot_freq=sp_cfg.get("snapshot_freq", 100_000),
            swap_freq=sp_cfg.get("swap_freq", 100_000),
            ppo_prob=sp_cfg.get("ppo_prob", 0.7),
            seed_models=sp_cfg.get("seed_models", []),
            logging_callback=log_callback,
            verbose=1,
        )
        callbacks.append(sp)

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=reset_steps,
    )
    model.save(config["save_path"])
    env.close()

    print(f"\nModel saved to {config['save_path']}.zip")
    return model, log_callback


def train_curriculum(config):
    curriculum = config["curriculum"]
    total_budget = sum(s["timesteps"] for s in curriculum)

    # Build first stage env and model (optionally resuming from a checkpoint)
    first_config = _merge_stage(config, curriculum[0])
    env = make_env(first_config)
    reward_fn = getattr(env, "reward_fn", None)
    resume_from = config.get("resume_from")
    if resume_from:
        clean_resume = ensure_model_zip(resume_from) + ".zip"
        model = MaskablePPO.load(
            clean_resume,
            env=env,
            learning_rate=_make_lr(config),
            tensorboard_log=config["tb_log"],
        )
        print(f"\nResuming curriculum from: {resume_from}")
    else:
        model = _make_model(config, env)

    _print_config_summary(model, config, total_budget, curriculum[0]["enemies"])
    print(f"  Curriculum: {len(curriculum)} stages")
    for i, s in enumerate(curriculum):
        print(f"    Stage {i+1}: {s['enemies']} for {s['timesteps']:,} steps")

    callback = CatanLoggingCallback(
        log_freq=config["log_freq"],
        reward_fn=reward_fn,
    )

    for i, stage in enumerate(curriculum):
        stage_config = _merge_stage(config, stage)

        if i > 0:
            env.close()
            env = make_env(stage_config)
            reward_fn = getattr(env, "reward_fn", None)
            model.set_env(env)
            callback.reward_fn = reward_fn

        # Reset LR schedule each stage so linear decay covers exactly this stage's budget.
        # Pass stage["timesteps"] directly (not cumulative): SB3 adds self.num_timesteps
        # internally when reset_num_timesteps=False, so passing the stage budget gives
        # the correct number of additional steps. Using cumulative totals caused each
        # stage to compound all prior stages' steps, making later stages exponentially
        # longer than intended.
        model.learning_rate = _make_lr(stage_config)

        callback.on_curriculum_stage(i, stage)

        model.learn(
            total_timesteps=stage["timesteps"],
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
