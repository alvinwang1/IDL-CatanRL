import modal
import os

# Define the Modal App
app = modal.App("catan-rl-training-parallel")

# Define the Image with required dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "catanatron>=3.2.1",
        "catanatron_gym>=4.0.0",
        "sb3-contrib>=2.2.1",
        "stable-baselines3>=2.2.1",
        "gymnasium>=0.29.1",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "networkx>=3.2.1",
        "tensorboard>=2.14.0",
    )
    .add_local_dir(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "catanrl")),
        remote_path="/root/catanrl"
    )
    .add_local_dir(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs")),
        remote_path="/root/configs"
    )
)

# Persistent volume for checkpoints and logs
volume = modal.Volume.from_name("catan-rl-storage", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="t4",  # Most cost-efficient GPU for this scale
    cpu=16,    # Increase CPU for faster environment simulation
    timeout=86400, # 24 hours
)
def train_remote(config_path="configs/curriculum_4p.yaml", overrides=[]):
    import sys
    import os
    sys.path.insert(0, "/root") # Ensure local catanrl takes precedence
    
    from catanrl.train import train
    from catanrl.config import load_config
    
    print(f"Starting remote training with config: {config_path}")
    
    # Load and adjust config for remote paths
    config = load_config(config_path, overrides)
    
    # Redirect outputs to persistent volume
    original_save = config["save_path"]
    original_tb = config["tb_log"]
    config["save_path"] = f"/storage/{original_save}"
    config["tb_log"] = f"/storage/{original_tb}"
    
    if config.get("self_play"):
        sp_dir = config.get("self_play", {}).get("snapshot_dir", "models/snapshots")
        config["self_play"]["snapshot_dir"] = f"/storage/{sp_dir}"
        os.makedirs(config["self_play"]["snapshot_dir"], exist_ok=True)
        
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    os.makedirs(config["tb_log"], exist_ok=True)
    
    print(f"Saving models to: {config['save_path']}")
    print(f"TensorBoard logs at: {config['tb_log']}")
    
    try:
        train(config)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.local_entrypoint()
def main(config_path: str = "configs/curriculum_4p.yaml"):
    train_remote.remote(config_path)
