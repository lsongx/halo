import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import EMBEDDER


@EMBEDDER.register_module()
class NormalEmbedder(nn.Module):
    def __init__(self, 
                 in_dims, 
                 nb_freqs, 
                 std=16, 
                 lf_nb_freqs=0, 
                 lf_std=16, 
                 scale=1, 
                 use_log=False,
                 shift=0,
                 include_input=True,
                 max_scale_iter=None):
        super().__init__()
        self.in_dims = in_dims
        self.nb_freqs = nb_freqs
        self.lf_nb_freqs = lf_nb_freqs
        self.include_input = include_input
        self.scale = scale
        self.shift = shift
        self.out_dims = (2 * in_dims * nb_freqs + in_dims) \
            if include_input else (2 * in_dims * nb_freqs)
        self.use_log = use_log

        freqs = torch.normal(mean=0.0, std=std, size=(nb_freqs,))
        freqs = torch.sort(freqs.abs()).values
        if lf_nb_freqs > 0:
            lf_freqs = torch.normal(mean=0.0, std=lf_std, size=(lf_nb_freqs,))
            freqs = torch.cat([lf_freqs, freqs])
        self.freqs = nn.Parameter(freqs, requires_grad=False)

        self.funcs = [torch.sin, torch.cos]
        self.max_scale_iter = max_scale_iter

    def forward(self, inputs, iter=None):
        device = inputs.device
        if self.use_log:
            neg_inputs = inputs<0
            inputs = (inputs.abs()+1).log()
            inputs[neg_inputs] *= -1
        embeds = [inputs] if self.include_input else []

        weight = torch.ones_like(self.freqs)
        # scale the embeding (nerfie)
        if iter is not None and self.max_scale_iter is not None:
            for idx, w in enumerate(weight):
                # iter 0: 0[0]/; iter maxiter*(idx/self.nb_freqs): 1[pi]
                if idx >= self.lf_nb_freqs:
                    maxiter = self.max_scale_iter*((idx+1)/self.nb_freqs)
                    progress = 3.141592657*iter/maxiter
                    weight[idx] = w * (1-torch.cos(progress))/2
                else:
                    weight[idx] = w

        for idx, freq in enumerate(self.freqs):
            freq = freq.unsqueeze(0).to(device).to(inputs.dtype)
            for func in self.funcs:
                embeds.append(weight[idx] * func((inputs-self.shift)/self.scale*freq))
        embeds = torch.cat(embeds, dim=-1)
        return embeds

