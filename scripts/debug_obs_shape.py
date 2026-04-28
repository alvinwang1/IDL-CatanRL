import modal

volume = modal.Volume.from_name("catan-rl-storage")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "stable-baselines3",
        "sb3-contrib",
        "catanatron-gym",
        "gymnasium",
    )
    .add_local_dir("catanrl", remote_path="/root/catanrl")
)
app = modal.App("catan-rl-debug")

@app.function(image=image, volumes={"/storage": volume})
def debug_env():
    import gymnasium
    import catanatron_gym
    import numpy as np
    from catanrl.env_utils import build_enemies
    
    config = {"enemies": ["ValueFunction", "WeightedRandom"], "vps_to_win": 10}
    enemies = build_enemies(config["enemies"], config)
    
    env = gymnasium.make("catanatron-v1", config={"enemies": enemies, "vps_to_win": 10})
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Sample observation shape: {obs.shape}")
    return obs.shape

if __name__ == "__main__":
    with modal.Retrying():
        with app.run():
            debug_env.remote()
