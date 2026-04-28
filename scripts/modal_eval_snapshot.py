import modal
import os

# Define the same environment as modal_train
volume = modal.Volume.from_name("catan-rl-storage")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "stable-baselines3",
        "sb3-contrib",
        "catanatron-gym",
        "pyyaml",
        "gymnasium",
        "shimmy",
    )
    .add_local_dir("catanrl", remote_path="/root/catanrl")
)
app = modal.App("catan-rl-eval")

@app.function(image=image, volumes={"/storage": volume}, cpu=4)
def evaluate_snapshot(snapshot_path: str, enemies_str: str = "ValueFunction,WeightedRandom,WeightedRandom", n_games: int = 100):
    enemies = [e.strip() for e in enemies_str.split(",")]
    import torch
    from sb3_contrib.ppo_mask import MaskablePPO
    from catanrl.ppo_player import PPOPlayer
    from catanrl.env_utils import build_enemies, ActionMasker, mask_fn
    from catanrl.models import CatanTopologyExtractor
    import gymnasium
    import numpy as np

    # Config for evaluation
    config = {
        "enemies": enemies,
        "vps_to_win": 10,
        "structured_encoder": True,
        "features_dim": 256
    }
    
    # Load model
    print(f"Loading snapshot from {snapshot_path}...")
    model = MaskablePPO.load(snapshot_path, custom_objects={
        "features_extractor_class": CatanTopologyExtractor
    })
    
    # Setup environment
    enemies_objs = build_enemies(enemies, config)
    env = gymnasium.make(
        "catanatron-v1",
        config={"enemies": enemies_objs, "vps_to_win": 10}
    )
    env = ActionMasker(env, mask_fn)
    
    wins = 0
    lengths = []
    print(f"Starting evaluation of {n_games} games...")
    
    for i in range(n_games):
        obs, info = env.reset()
        done = False
        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        if reward > 0:
            wins += 1
        lengths.append(info.get("episode", {}).get("l", 0))
        
        if (i+1) % 10 == 0:
            print(f"  Game {i+1}/{n_games} | Current WR: {wins/(i+1)*100:.1f}%")

    wr = wins / n_games
    print(f"\n{'='*40}")
    print(f"EVALUATION COMPLETE")
    print(f"Win Rate: {wr*100:.1f}%")
    print(f"Avg Length: {np.mean(lengths):.1f} steps")
    print(f"{'='*40}")
    return {"win_rate": wr, "avg_length": float(np.mean(lengths))}

@app.local_entrypoint()
def main():
    # Checking 6.15M snapshot
    snapshot = "/storage/models/snapshots/snapshot_6150000"
    
    print("\n--- QUICK CHECK: 6.15M SNAPSHOT AGAINST WEIGHTED RANDOM ---")
    evaluate_snapshot.remote(
        snapshot, 
        enemies_str="WeightedRandom,WeightedRandom,WeightedRandom",
        n_games=20
    )
