### Trading agent with transformer baseline added

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=128, num_heads=2, num_layers=2):
        super(TransformerFeatureExtractor, self).__init__(
            observation_space, features_dim=embed_dim
        )

        self.embed_dim = embed_dim
        self.relu = nn.ReLU()
        # Ensure input is projected to the right dimension (d_model) for the transformer
        self.embedding = nn.Linear(observation_space.shape[0], embed_dim)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, observations):
        # Assume observations are [batch_size, observation_dim]
        # Project observations to the transformer input size
        embedded_obs = self.relu(self.embedding(observations))

        # Transformer expects input as [sequence_length, batch_size, embed_dim]
        embedded_obs = embedded_obs.unsqueeze(0)  # Add a sequence dimension

        # Using the same tensor for src and tgt as we're not doing sequence-to-sequence modeling
        transformer_out = self.transformer(embedded_obs, embedded_obs)

        # Reduce the sequence dimension, since we added it artificially
        transformer_out = transformer_out.squeeze(0)

        transformer_out = torch.clamp(transformer_out, -10, 10)

        # claming outputs so that it never reaches to nan
        return torch.clamp(self.fc(transformer_out), -10, 10)


class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(TransformerPolicy, self).__init__(
            *args,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(embed_dim=64, num_heads=2, num_layers=2),
            **kwargs,
        )
