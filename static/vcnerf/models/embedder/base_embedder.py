import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import EMBEDDER


@EMBEDDER.register_module()
class BaseEmbedder(nn.Module):
    def __init__(self, 
                 in_dims, 
                 nb_freqs, 
                 scale=1, 
                 use_log=False,
                 include_input=True):
        super().__init__()
        self.in_dims = in_dims
        self.nb_freqs = nb_freqs
        self.scale = scale
        self.include_input = include_input
        self.out_dims = (2 * in_dims * nb_freqs + in_dims) \
            if include_input else (2 * in_dims * nb_freqs)
        self.use_log = use_log

        self.freqs = 2 ** torch.linspace(0, self.nb_freqs-1, self.nb_freqs)

        self.funcs = [torch.sin, torch.cos]

    def __call__(self, inputs):
        device = inputs.device
        if self.use_log:
            neg_inputs = inputs<0
            inputs = (inputs.abs()+1).log()
            inputs[neg_inputs] *= -1
        embeds = [inputs] if self.include_input else []
        for freq in self.freqs:
            freq = freq.unsqueeze(0).to(device).to(inputs.dtype)
            for func in self.funcs:
                embeds.append(func(inputs/self.scale * freq))
        embeds = torch.cat(embeds, dim=-1)
        return embeds

