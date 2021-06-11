import torch
import torch.nn as nn

        
class TaskRepresentationBlockV1(nn.Module):
    '''
    This block is used to map the input task code vector z
    to an intermediate latent space w.
    A mapping consists of multiple fully connected layers.
    '''
    def __init__(self, n_fc, input_dim, output_dim):
        super().__init__()
        # L2 normalization of the input code vector to be of unit length
        self.l2_norm = VectorL2Normalize()
        
        layers = []
        act = nn.LeakyReLU(0.2)
        # Mapping: 
        # Linear fully connected layers followed by activation fucntions (leaky relu)
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(act)
        for _ in range(n_fc-1):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(act)
        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        # L2 normalize
        latent_z = self.l2_norm(latent_z)

        # Map the input task code vector z to an intermediate latent space w
        return self.mapping(latent_z)


class VectorL2Normalize(nn.Module):
    """
    A module that normalizes the input vector to unit length
    by dividing it with its L2 norm.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        norm = x.norm(p=2, dim=0, keepdim=False).detach() + 1e-8
        x_normalized = x / norm
        return x_normalized
