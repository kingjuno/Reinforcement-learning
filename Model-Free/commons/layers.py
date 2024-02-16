import math

import torch
import torch.nn as nn


class NoisyLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_std = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias_mean = nn.Parameter(torch.empty((out_features), **factory_kwargs))
        self.bias_std = nn.Parameter(torch.empty((out_features), **factory_kwargs))

        self.weight_noise = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias_noise = nn.Parameter(torch.empty((out_features), **factory_kwargs))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        range = math.sqrt(3 / self.in_features)
        self.weight_mean.data.uniform_(-range, range)
        self.weight_std.data.fill_(0.017)
        self.bias_mean.data.uniform_(-range, range)
        self.bias_mean.data.fill_(0.017)

    def reset_noise(self):
        range = math.sqrt(1 / self.out_features)
        self.weight_noise.data.uniform_(-range, range)
        self.bias_noise.data.fill_(0.5 * range)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, x):
        if self.training:
            w = self.weight_mean + self.weight_std.mul(self.weight_noise)
            b = self.bias_mean + self.bias_std.mul(self.bias_noise)
        else:
            w = self.weight_mean
            b = self.bias_mean
        return nn.functional.linear(x, w, b)
