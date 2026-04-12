from src.models.resblock import ResidualBlock
from src.models.prototypelayer import PrototypeLayer

import torch
import torch.nn as nn
import torch.nn.functional as F



class PPGModel(nn.Module):

    def __init__(self, representation_dimension, mask_probability = 0.2):

        super().__init__()

        self.mask_probability = mask_probability
        
        self.masking_encoder = nn.Sequential(
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 1, 'out_channels' : 4, 'kernel_size' : 3},
                    {'in_channels' : 4, 'out_channels' : 4, 'kernel_size' : 3}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 4, 'out_channels' : 8, 'kernel_size' : 3},
                    {'in_channels' : 8, 'out_channels' : 8, 'kernel_size' : 3}
                ]
            )
        )

        self.mask_token = nn.Parameter(torch.randn(1, 8, 1))
        
        self.encoder_layer_1 = nn.Sequential(
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 9, 'out_channels' : 16, 'kernel_size' : 5},
                    {'in_channels' : 16, 'out_channels' : 16, 'kernel_size' : 5}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 16, 'out_channels' : 32, 'kernel_size' : 5},
                    {'in_channels' : 32, 'out_channels' : 32, 'kernel_size' : 5}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 32, 'out_channels' : 64, 'kernel_size' : 5},
                    {'in_channels' : 64, 'out_channels' : 64, 'kernel_size' : 5}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 64, 'out_channels' : 128, 'kernel_size' : 5},
                    {'in_channels' : 128, 'out_channels' : 128, 'kernel_size' : 5}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 128, 'out_channels' : 128, 'kernel_size' : 5},
                    {'in_channels' : 128, 'out_channels' : 128, 'kernel_size' : 5}
                ]
            )
        )

        self.prototype_layer_1 = PrototypeLayer(
            n_prototypes = 10,
            prototype_len = 20,
            prototype_channels = 128,
        )

        self.temporal_mixing_layer_1 = nn.Sequential(
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 10, 'out_channels' : 20, 'kernel_size' : 3},
                    {'in_channels' : 20, 'out_channels' : 20, 'kernel_size' : 3}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 20, 'out_channels' : 40, 'kernel_size' : 5},
                    {'in_channels' : 40, 'out_channels' : 40, 'kernel_size' : 5}
                ]
            )
        )

        self.encoder_layer_2 = nn.Sequential(
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 128, 'out_channels' : 256, 'kernel_size' : 5, 'dilation' : 2},
                    {'in_channels' : 256, 'out_channels' : 256, 'kernel_size' : 5, 'dilation' : 4}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 256, 'out_channels' : 512, 'kernel_size' : 5, 'dilation' : 8},
                    {'in_channels' : 512, 'out_channels' : 512, 'kernel_size' : 5, 'dilation' : 16}
                ]
            )
        )

        self.prototype_layer_2 = PrototypeLayer(
            n_prototypes = 10,
            prototype_len = 20,
            prototype_channels = 512,
        )

        self.temporal_mixing_layer_2 = nn.Sequential(
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 10, 'out_channels' : 20, 'kernel_size' : 3},
                    {'in_channels' : 20, 'out_channels' : 20, 'kernel_size' : 3}
                ]
            ),
            ResidualBlock(
                layer_configs = [
                    {'in_channels' : 20, 'out_channels' : 40, 'kernel_size' : 5},
                    {'in_channels' : 40, 'out_channels' : 40, 'kernel_size' : 5}
                ]
            )
        )

        self.last_layer = nn.Linear(
            in_features = 80,
            out_features = representation_dimension
        )

        # model components for stagewise training 
        self.encoder_layers = [self.encoder_layer_1, self.encoder_layer_2]
        self.prototype_layers = [self.prototype_layer_1, self.prototype_layer_2]
        self.temporal_mixing_layers = [self.temporal_mixing_layer_1, self.temporal_mixing_layer_2]
        # IMPORTANT: masking_encoder, mask_token, and last_layer are required but accessed directly 

    def forward(self, x):

        z = self.masking_encoder(x)
        if self.training: # only apply masking during training
            mask = self.create_mask(batch_size = z.size(0), seq_len = z.size(-1), device = z.device)
            z = self.apply_latent_mask(z, mask)
        
        else: # add trivial mask channel during evaluation for consistent input shape
            zero_mask = torch.zeros(z.size(0), 1, z.size(-1), device = z.device, dtype = z.dtype)
            z = torch.cat([z, zero_mask], dim = 1)
        
        z = self.encoder_layer_1(z)
        prototype_layer_1_activation_tensor = self.prototype_layer_1(z)
        m1 = self.temporal_mixing_layer_1(prototype_layer_1_activation_tensor)
        f1 = F.adaptive_avg_pool1d(m1, 1).squeeze(-1)
        
        z = self.encoder_layer_2(z)
        prototype_layer_2_activation_tensor = self.prototype_layer_2(z)
        m2 = self.temporal_mixing_layer_2(prototype_layer_2_activation_tensor)
        f2 = F.adaptive_avg_pool1d(m2, 1).squeeze(-1)

        f = torch.cat([f1, f2], dim=1)
        out = self.last_layer(f)

        return out
    
    def create_mask(self, batch_size, seq_len, device = None):
        """
        Create a Bernoulli mask over timestamps.

        Returns:
            mask: Float tensor of shape [B, 1, T]
                - 1.0 means 'masked', 0.0 means 'keep'
        """

        if device is None:
            device = self.mask_token.device

        mask = torch.bernoulli(torch.full((batch_size, 1, seq_len), self.mask_probability, device = device))
        
        return mask
    
    def apply_latent_mask(self, z, mask):
        """
        Apply timestamp-level masking in latent space.

        Args:
            z: latent tensor of shape [B, C, T]
            mask: mask tensor of shape [B, 1, T]

        Returns:
            z_out: masked latent tensor, optionally with mask indicator channel
        """
        
        mask = mask.to(dtype = z.dtype, device = z.device)

        z_masked = z * (1.0 - mask) + self.mask_token * mask
        z_masked = torch.cat([z_masked, mask], dim = 1) # add mask indicator channel

        return z_masked