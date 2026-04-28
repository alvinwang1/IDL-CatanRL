import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CatanTopologyExtractor(BaseFeaturesExtractor):
    """
    A structured feature extractor for Catan board topology.
    Splits the 1002-dimensional flat vector into semantically meaningful 
    components and processes them with dedicated sub-networks.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Mapping ranges (based on alphabetical sorting of catanatron-gym features)
        self.bank_range = (0, 6)
        self.edge_range = (6, 294)
        self.is_range = (294, 296)
        self.node_range = (296, 728)
        self.player_range = (728, 796)
        self.port_range = (796, 850)
        self.tile_range = (850, 1002)
        # 160 extra expert features (production, expansion, reachability)
        self.expert_range = (1002, 1162)

        # Architectural components
        # 1. Edge processing: 72 edges * 4 players = 288
        self.edge_net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        
        # 2. Node processing: 54 nodes * 8 features = 432
        self.node_net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        
        # 3. Tile processing: 19 tiles * 8 features = 152
        self.tile_net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        
        # 4. Port processing: 9 ports * 6 features = 54
        self.port_net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )

        # 5. Expert Features: 160 features
        self.expert_net = nn.Sequential(
            nn.Linear(160, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )

        # Latent dimensions after local processing:
        # Edges: 576, Nodes: 864, Tiles: 304, Ports: 72, Global: 76, Expert: 64
        total_latent = 576 + 864 + 304 + 72 + 76 + 64
        
        self.final_net = nn.Sequential(
            nn.Linear(total_latent, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Slice the input tensor
        bank = observations[:, self.bank_range[0]:self.bank_range[1]]
        edges = observations[:, self.edge_range[0]:self.edge_range[1]]
        is_flags = observations[:, self.is_range[0]:self.is_range[1]]
        nodes = observations[:, self.node_range[0]:self.node_range[1]]
        players = observations[:, self.player_range[0]:self.player_range[1]]
        ports = observations[:, self.port_range[0]:self.port_range[1]]
        tiles = observations[:, self.tile_range[0]:self.tile_range[1]]
        expert = observations[:, self.expert_range[0]:self.expert_range[1]]

        # Process structured components
        edges = edges.view(-1, 72, 4)
        edge_latent = self.edge_net(edges).view(observations.size(0), -1)
        
        nodes = nodes.view(-1, 54, 8)
        node_latent = self.node_net(nodes).view(observations.size(0), -1)
        
        tiles = tiles.view(-1, 19, 8)
        tile_latent = self.tile_net(tiles).view(observations.size(0), -1)
        
        ports = ports.view(-1, 9, 6)
        port_latent = self.port_net(ports).view(observations.size(0), -1)

        expert_latent = self.expert_net(expert)

        # Concatenate all latents + global/player features
        combined = torch.cat([
            bank,
            edge_latent,
            is_flags,
            node_latent,
            player_latent := players,
            port_latent,
            tile_latent,
            expert_latent
        ], dim=1)

        return self.final_net(combined)


class CatanGraphEncoder(BaseFeaturesExtractor):
    """
    A Graph Neural Network feature extractor for Catan.
    Uses precomputed adjacency matrices to perform message passing between
    vertices, tiles, and edges.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, hidden_dim: int = 32):
        super().__init__(observation_space, features_dim)
        
        # Load topology
        try:
            topology = torch.load("catanrl/topology.pt", map_location="cpu")
        except FileNotFoundError:
            topology = {
                "adj_vv": torch.eye(54),
                "adj_vt": torch.zeros(54, 19),
                "adj_ve": torch.zeros(54, 72)
            }
            
        self.register_buffer("adj_vv", topology["adj_vv"])
        self.register_buffer("adj_vt", topology["adj_vt"])
        self.register_buffer("adj_ve", topology["adj_ve"])
        
        # Mapping ranges (based on catanatron-gym features)
        self.bank_range = (0, 6)
        self.edge_range = (6, 294)
        self.is_range = (294, 296)
        self.node_range = (296, 728)
        self.player_range = (728, 796)
        self.port_range = (796, 850)
        self.tile_range = (850, 1002)
        self.expert_range = (1002, 1162)

        # 1. Projections
        self.node_proj = nn.Linear(8, hidden_dim)
        self.edge_proj = nn.Linear(4, hidden_dim)
        self.tile_proj = nn.Linear(8, hidden_dim)
        
        # 2. Message Passing Layer
        self.msg_vv = nn.Linear(hidden_dim, hidden_dim)
        self.msg_vt = nn.Linear(hidden_dim, hidden_dim)
        self.msg_ve = nn.Linear(hidden_dim, hidden_dim)
        self.msg_self = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

        # 3. Global processing
        self.port_net = nn.Sequential(nn.Linear(54, 32), nn.ReLU())
        
        # Expert features might be missing (standard 4p obs is 1002)
        expert_latent_dim = 0
        self.has_expert = observation_space.shape[0] >= self.expert_range[1]
        if self.has_expert:
            self.expert_net = nn.Sequential(nn.Linear(160, 64), nn.ReLU())
            expert_latent_dim = 64
        
        # Final aggregation: Max pooling over nodes + Global features
        # Global features: bank(6) + is_flags(2) + players(68) + ports(32) + expert(64?)
        global_dim = 6 + 2 + 68 + 32 + expert_latent_dim
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Slice and project
        nodes = observations[:, self.node_range[0]:self.node_range[1]].reshape(-1, 54, 8)
        edges = observations[:, self.edge_range[0]:self.edge_range[1]].reshape(-1, 72, 4)
        tiles = observations[:, self.tile_range[0]:self.tile_range[1]].reshape(-1, 19, 8)
        
        h_v = self.node_proj(nodes)  # (B, 54, H)
        h_e = self.edge_proj(edges)  # (B, 72, H)
        h_t = self.tile_proj(tiles)  # (B, 19, H)
        
        # 2. Message Passing (Heterogeneous)
        m_vv = torch.matmul(self.adj_vv, h_v) # Neighbor vertices
        m_vt = torch.matmul(self.adj_vt, h_t) # Adjacent tiles
        m_ve = torch.matmul(self.adj_ve, h_e) # Adjacent edges
        
        h_v_next = self.msg_self(h_v) + self.msg_vv(m_vv) + self.msg_vt(m_vt) + self.msg_ve(m_ve)
        h_v = self.activation(self.layer_norm(h_v_next))
        
        # 3. Global Features
        bank = observations[:, self.bank_range[0]:self.bank_range[1]]
        is_flags = observations[:, self.is_range[0]:self.is_range[1]]
        players = observations[:, self.player_range[0]:self.player_range[1]]
        ports = observations[:, self.port_range[0]:self.port_range[1]]
        
        h_ports = self.port_net(ports)
        
        # 4. Pooling and Concatenation
        h_graph, _ = torch.max(h_v, dim=1) # Global Max Pooling
        
        combined_list = [h_graph, bank, is_flags, players, h_ports]
        
        if self.has_expert:
            expert = observations[:, self.expert_range[0]:self.expert_range[1]]
            h_expert = self.expert_net(expert)
            combined_list.append(h_expert)
            
        combined = torch.cat(combined_list, dim=1)
        
        return self.final_net(combined)
