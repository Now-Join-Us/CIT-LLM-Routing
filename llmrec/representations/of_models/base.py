import torch.nn as nn

class ModelRepresentationEncoder(nn.Module):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def apply(self, item):
        return item
