import modal
import os

volume = modal.Volume.from_name("catan-rl-storage")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("stable-baselines3", "sb3-contrib", "catanatron-gym")
    .add_local_dir("catanrl", remote_path="/root/catanrl")
)

app = modal.App("inspect-model")

@app.function(image=image, volumes={"/storage": volume})
def inspect_model(path: str):
    import sys
    sys.path.insert(0, "/root")
    from sb3_contrib.ppo_mask import MaskablePPO
    from catanrl.models import CatanTopologyExtractor
    
    print(f"Loading model from {path}...")
    model = MaskablePPO.load(path, custom_objects={
        "features_extractor_class": CatanTopologyExtractor
    })
    print(f"Total timesteps: {model.num_timesteps}")
    return model.num_timesteps

@app.local_entrypoint()
def main(path: str = "/storage/models/parallel_recovery_stage0.zip"):
    inspect_model.remote(path)
