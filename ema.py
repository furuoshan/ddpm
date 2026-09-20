import torch


class EMA:
    """Exponential moving average of trainable parameters.

    BatchNorm running stats are copied, not averaged: those buffers are already
    an EMA, and smoothing them again breaks eval()-time normalization.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.parameter_names = {name for name, _ in model.named_parameters()}
        self.shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, tensor in model.state_dict().items():
            shadow = self.shadow[name]
            if name in self.parameter_names and tensor.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(tensor)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        for name, tensor in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
