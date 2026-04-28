import modal
import os

app = modal.App("pretrain-alpha-beta")

volume = modal.Volume.from_name("catan-rl-storage", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "catanatron>=3.2.1",
        "catanatron_gym>=4.0.0",
        "sb3-contrib>=2.2.1",
        "stable-baselines3>=2.2.1",
        "gymnasium>=0.29.1",
        "torch>=2.0.0",
        "pyyaml>=6.0",
    )
    .add_local_dir("catanrl", remote_path="/root/catanrl")
    .add_local_dir("configs", remote_path="/root/configs")
)

@app.function(image=image, volumes={"/storage": volume}, timeout=3600)
def pretrain_remote(config_path="configs/alpha_beta_self_play.yaml", n_steps=20000):
    import gymnasium
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import sys
    sys.path.insert(0, "/root")
    
    from sb3_contrib.ppo_mask import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from catanrl.env_utils import make_env
    from catanrl.config import load_config
    from catanrl.models import CatanTopologyExtractor
    from catanrl.catanatron_ext.value import get_value_fn, DEFAULT_WEIGHTS
    import catanrl.catanatron_ext.features as local_features
    import catanatron_gym.features
    
    # Monkey-patch to force the gym environment to use our expanded feature set
    catanatron_gym.features.feature_extractors = local_features.feature_extractors
    catanatron_gym.features.create_sample = local_features.create_sample
    catanatron_gym.features.get_feature_ordering = local_features.get_feature_ordering

    config = load_config(config_path)
    config["structured_encoder"] = True
    
    print(f"Loading environment with config: {config_path}")
    env = make_env(config)
    
    policy_kwargs = dict(
        net_arch=config["net_arch"],
        features_extractor_class=CatanTopologyExtractor,
        features_extractor_kwargs=dict(features_dim=config.get("features_dim", 512))
    )
    
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device="auto"
    )
    
    value_fn = get_value_fn("base_fn", DEFAULT_WEIGHTS)
    optimizer = optim.Adam(model.policy.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Collecting {n_steps} samples and training to predict Alpha-Beta scores...")
    
    from catanrl.env_utils import ENEMY_COLORS
    obs, _ = env.reset()
    for step in range(n_steps):
        game = env.unwrapped.game
        p0_color = next(c for c in game.state.colors if c not in ENEMY_COLORS)
        target_value = value_fn(game, p0_color) / 3e14
        
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
        predicted_value = model.policy.predict_values(obs_tensor)
        
        loss = criterion(predicted_value, torch.as_tensor([[target_value]], dtype=torch.float32).to(model.device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        valid_actions = env.unwrapped.get_valid_actions()
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        if step % 1000 == 0:
            print(f"Step {step}/{n_steps} | Loss: {loss.item():.6f} | Target: {target_value:.4f} | Pred: {predicted_value.item():.4f}")

    # Save to storage
    save_path = f"/storage/models/alpha_beta_self_play_pretrained"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Pre-trained model saved to {save_path}.zip")

@app.local_entrypoint()
def main(config_path: str = "configs/alpha_beta_self_play.yaml", n_steps: int = 20000):
    pretrain_remote.remote(config_path, n_steps)
