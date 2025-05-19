import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

BEHAVIOR_ENCODER_BLOCKS = 1
INTERATION_ENCODER_BLOCKS = 1
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=128, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value):
        # Cross-attention
        attn_output, _ = self.attn(query, key_value, key_value)
        x = self.norm1(query + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        out = self.norm2(x + ffn_output)

        return out

class ObservationEmbedder(nn.Module):
    def __init__(self, obs_dim ,embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        self.output_dim = embedding_dim
    def forward(self, input):
        return self.input_proj(input)

class BehaviorEncoder(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=BEHAVIOR_ENCODER_BLOCKS, n_heads=1,weight_type="exp"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.weight_type = weight_type

    def forward(self, history_batch):
        # history_batch: (batch, seq_len, input_dim)
        batch_size = history_batch.size(0)
        weighted_history = self.apply_temporal_weights(history_batch)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        x = torch.cat([weighted_history, cls_tokens], dim=1)                   # (batch, seq+1, hidden)
        # Positional encoding for x
        pos_enc = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (batch, seq+1, 1)
        pos_enc = pos_enc.float() / (x.size(1) ** 0.5)  # Scale the positional encoding
        pos_enc = pos_enc * torch.sin(pos_enc)  # Apply sine function for positional encoding
        x = x + pos_enc  # Add positional encoding to the input
        out = self.encoder(x)                                   # (batch, seq+1, hidden)
        return out[:, 0]  # Return CLS token                    # (batch, hidden)
    def apply_temporal_weights(self, x):
        # x: (batch, seq_len, hidden)
        batch_size, seq_len, hidden_dim = x.size()

        if self.weight_type == "exp":
            decay_factor = 0.9  # Decay rate (adjustable)
            weights = torch.tensor([decay_factor ** (seq_len - 1 - i) for i in range(seq_len)],
                                   device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        elif self.weight_type == "lin":
            weights = torch.linspace(1.0, 2.0, steps=seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        else:
            return x  # No weighting
        return x * weights

class InteractionEncoder(nn.Module):
    def __init__(self, hidden_dim=128, n_blocks=INTERATION_ENCODER_BLOCKS, n_heads=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=hidden_dim, n_heads=n_heads)
            for _ in range(n_blocks)
        ])

    def forward(self, ego_embedding, other_embeddings):
        # ego_embedding: (batch, 1, hidden_dim)
        x = ego_embedding
        for block in self.blocks:
            x = block(x, other_embeddings)
        return x  # (batch, 1, hidden_dim)

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, env, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.env = env
        self.game_obs = env.game_obs
        obs_dim = self.game_obs.obs_dim
        self.obs_dim = obs_dim
        self.history_embedder = ObservationEmbedder(obs_dim=obs_dim)
        self.ego_embedder = ObservationEmbedder(obs_dim=obs_dim)
        self.behavior_encoder = BehaviorEncoder()
        self.interaction_encoder = InteractionEncoder()
        self.output_dim = features_dim
        self.sigmoid = nn.Sigmoid()
    def forward(self, obs):
        cars_num = self.game_obs.get_cars_num()
        ego_idx = self.game_obs.get_ego_idx()
        # obs: (batch, I, input_dim), I: number of informations from observation
        batch_size = obs.shape[0]
        history_embedding = self.history_embedder(obs[:, :ego_idx, :self.obs_dim])
        ego_embedding = self.ego_embedder(obs[:, ego_idx:, :])
        cls_tokens = self.behavior_encoder(history_embedding.view(batch_size * cars_num, -1, history_embedding.shape[-1])) 
        cls_tokens = cls_tokens.view(batch_size, cars_num, -1) 
        # Concat cls_token with cls_tokens
        context = self.interaction_encoder(ego_embedding, cls_tokens)
        # Apply sigmoid for last token
        return context[:,0]