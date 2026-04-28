import torch
import gymnasium as gym
from catanrl.models import CatanTopologyExtractor

def test_encoder():
    # 1002 dimensional observation space
    obs_space = gym.spaces.Box(low=0, high=1, shape=(1002,), dtype=float)
    
    # Initialize extractor
    extractor = CatanTopologyExtractor(obs_space, features_dim=256)
    print("Successfully initialized CatanTopologyExtractor")
    
    # Create dummy observation
    dummy_obs = torch.randn(2, 1002) # Batch size 2
    
    # Forward pass
    with torch.no_grad():
        output = extractor(dummy_obs)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 256)
    print("Test passed!")

if __name__ == "__main__":
    test_encoder()
