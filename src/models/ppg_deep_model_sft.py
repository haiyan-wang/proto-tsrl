from src.models.univariate_model import UnivariateModel

import torch
import torch.nn as nn
import torch.nn.functional as F



class UnivariateModelSFT(nn.Module):

    def __init__(self, untuned_model: UnivariateModel, n_classes: int):

        super().__init__()

        self.untuned_model = untuned_model

        self.linear_layer = nn.Linear(
            in_features = UnivariateModel.representation_dimension,
            out_features = n_classes
        )

    def forward(self, x):

        repr = self.untuned_model(x)
        logits = self.linear_layer(repr)

        return logits 