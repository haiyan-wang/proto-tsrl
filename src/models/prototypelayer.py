import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    """
    Prototype layer for 1D latent time-series embeddings.

    Parameters
    ----------
    n_prototypes : int
        Number of learnable prototypes.

    prototype_len : int
        Temporal length of each prototype in latent space.

    prototype_channels : int
        Number of channels in latent space. This must match the channel
        dimension of the input tensor passed to ``forward``.

    Notes
    -----
    Input shape:  (batch_size, prototype_channels, latent_length)
    Output shape: (batch_size, n_prototypes, activation_length)

    where ``activation_length = latent_length - prototype_len + 1`` for valid
    sliding-window comparison.
    """

    def __init__(
        self,
        n_prototypes,
        prototype_len,
        prototype_channels
    ):
        
        super().__init__()

        if n_prototypes <= 0:
            raise ValueError("n_prototypes must be a positive integer.")
        if prototype_len <= 0:
            raise ValueError("prototype_len must be a positive integer.")
        if prototype_channels <= 0:
            raise ValueError("prototype_channels must be a positive integer.")

        self.n_prototypes = n_prototypes
        self.prototype_len = prototype_len
        self.prototype_channels = prototype_channels
        self.prototype_shape = (n_prototypes, prototype_channels, prototype_len)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad = True)

        # misc.
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad = False) # for efficient calculation 
        self.epsilon = 1e-4 # for numerical stability

    def cosine_convolution(self, z):
        """
        Sliding cosine similarity between each prototype and patches in latent embedding.

        s(z, p) = <z, p> / (||z|| * ||p||)

        Parameters
        ----------
        z : torch.Tensor
            Shape (batch_size, prototype_channels, latent_length)

        Returns
        -------
        torch.Tensor
            Cosine similarities with shape (batch_size, n_prototypes, activation_length)
        """

        if z.dim() != 3:
            raise ValueError("Expected input z to have shape (batch_size, channels, length).")
        if z.size(1) != self.prototype_channels:
            raise ValueError(f"Expected {self.prototype_channels} input channels, got {z.size(1)}.")
        if z.size(2) < self.prototype_len:
            raise ValueError(f"Input length ({z.size(2)}) must be at least prototype_len ({self.prototype_len}).")

        zp = F.conv1d(input = z, weight = self.prototype_vectors)
        z2 = z ** 2
        z2_patch_sum = F.conv1d(input = z2, weight = self.ones)
        z_norm = torch.sqrt(z2_patch_sum)
        p2 = self.prototype_vectors ** 2
        p_norm = torch.sqrt(torch.sum(p2, dim=(1, 2))).view(1, -1, 1)

        cosine_similarities = zp / (z_norm * p_norm)

        return cosine_similarities

    def forward(self, z):
        """
        Compute the prototype activation tensor.

        Parameters
        ----------
        z : torch.Tensor
            Latent embedding of shape
            (batch_size, prototype_channels, latent_length)

        Returns
        -------
        torch.Tensor
            Activation tensor of shape
            (batch_size, n_prototypes, activation_length).
            Each channel corresponds to the full activation vector produced by a
            single prototype over time. All prototype activation vectors are
            concatenated along the prototype/channel dimension.
        """

        return self.cosine_convolution(z)