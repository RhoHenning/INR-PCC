import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_freq, num_freqs, include_input=True, log_sample=True):
        super().__init__()
        if log_sample:
            freq_bands = 2 ** torch.linspace(0, max_freq, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2 ** 0, 2 ** max_freq, steps=num_freqs)
        self.embed_funcs = []
        if include_input:
            self.embed_funcs.append(lambda x: x)
        for freq in freq_bands:
            self.embed_funcs.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_funcs.append(lambda x, freq=freq: torch.cos(x * freq))
        self.output_dim = input_dim * len(self.embed_funcs)

    def forward(self, inputs):
        outputs = torch.cat([func(inputs) for func in self.embed_funcs], -1)
        return outputs
    
    def extra_repr(self):
        num_freqs = len(self.embed_funcs) // 2
        return f'num_freqs={num_freqs}'

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, layer_norm=True, short_cut=True):
        super().__init__()
        self.short_cut = short_cut
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.ReLU()
        if layer_norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.register_buffer('norm', None)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        if self.norm:
            outputs = self.norm(outputs)
        outputs = self.activation(outputs + inputs if self.short_cut else outputs)
        return outputs

class ResidualNet(nn.Module):
    def __init__(self, input_dim, block_dim, hidden_dim, output_dim, num_blocks, layer_norm=True, short_cut=True):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, block_dim)
        self.linear2 = nn.Linear(block_dim, output_dim)
        self.blocks = nn.ModuleList([ResidualBlock(block_dim, hidden_dim, layer_norm, short_cut) for _ in range(num_blocks)])
        if layer_norm:
            self.norm = nn.LayerNorm(block_dim)
        else:
            self.register_buffer('norm', None)
  
    def forward(self, inputs):
        outputs = self.linear1(inputs)
        if self.norm:
            outputs = self.norm(outputs)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = self.linear2(outputs)
        return outputs

class Representation(nn.Module):
    def __init__(self, num_freqs, block_dim, hidden_dim, output_dim, num_blocks, layer_norm, short_cut):
        super().__init__()
        input_dim = (2 * num_freqs + 1) * 3
        self.encoding = PositionalEncoding(3, num_freqs - 1, num_freqs) if num_freqs > 0 else nn.Identity()
        self.net = ResidualNet(input_dim, block_dim, hidden_dim, output_dim, num_blocks, layer_norm, short_cut)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.encoding(inputs)
        outputs = self.net(outputs)
        outputs = self.sigmoid(outputs)
        return outputs

    def l1_penalty(self):
        penalty = 0
        for param in self.net.parameters():
            penalty += torch.abs(param).sum()
        return penalty
