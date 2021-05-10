import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims, multiplxer=4):
        super(MLP, self).__init__()
        hidden = int(dims * multiplxer)

        self.out = nn.Sequential(
            nn.Linear(dims, hidden),
            nn.GELU(),
            nn.Linear(hidden, dims)
        )

    def forward(self, x):
        return self.out(x)


class MixerLayer(nn.Module):
    def __init__(self, seq, dims):
        super(MixerLayer, self).__init__()

        self.layer_norm1 = nn.LayerNorm(dims)
        self.mlp1 = MLP(seq, multiplxer=0.5)
        self.layer_norm2 = nn.LayerNorm(dims)
        self.mlp2 = MLP(dims)

    def forward(self, x):
        out = self.layer_norm1(x).transpose(1, 2)
        out = self.mlp1(out).transpose(1, 2)
        out += x

        out2 = self.layer_norm2(out)
        out2 = self.mlp2(out2)
        out2 += out

        return out2


class MLPMixer(nn.Module):
    def __init__(self, seq, in_dims, dims, patch=32, n_classes=10, N=12):
        super(MLPMixer, self).__init__()

        # self.embedding = nn.Linear(in_dims, dims)
        self.embedding = nn.Conv2d(3, dims, kernel_size=patch, stride=patch)

        self.layers = nn.ModuleList()
        for _ in range(N):
            self.layers.append(MixerLayer(seq, dims))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dims, n_classes)
        self.dims = dims

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 3, 1).view(x.size(0), -1, self.dims)
        for layer in self.layers:
            out = layer(out)

        out = out.mean(dim=1)
        out = self.fc(out)

        return out


class Affine(nn.Module):
    def __init__(self, dims):
        super(Affine, self).__init__()

        self.alpha = nn.Parameter(torch.ones(dims))
        self.beta = nn.Parameter(torch.zeros(dims))

    def forward(self, x):
        return self.alpha * x + self.beta


class ResMLPBlock(nn.Module):
    def __init__(self, nb_patches, dims, layerscale_init):
        super(ResMLPBlock, self).__init__()
        self.affine1 = Affine(dims)
        self.affine2 = Affine(dims)
        self.linear_patches = nn.Linear(nb_patches, nb_patches)
        self.mlp_channels = MLP(dims)
        self.layerscale1 = nn.Parameter(layerscale_init * torch.ones(dims))
        self.layerscale2 = nn.Parameter(layerscale_init * torch.ones(dims))

    def forward(self, x):
        out1 = self.linear_patches(self.affine1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.layerscale1 * out1
        out2 = self.mlp_channels(self.affine2(x))
        x = x + self.layerscale2 * out2

        return x


class ResMLP(nn.Module):
    def __init__(self, dims, layerscale_init=1e-4, size=224, patch=32, num_classes=10, N=12):
        super(ResMLP, self).__init__()
        n = (size * size) // patch ** 2
        self.dims = dims
        self.embedding = nn.Conv2d(3, dims, kernel_size=patch, stride=patch)

        self.blocks = nn.ModuleList([ResMLPBlock(n, dims, layerscale_init) for _ in range(N)])
        self.affine = Affine(dims)
        self.out = nn.Linear(dims, num_classes)

    def forward(self, x):
        out = self.embedding(x).permute(0, 2, 3, 1).view(x.size(0), -1, self.dims)
        
        for layer in self.blocks:
            out = layer(out)

        out = self.affine(out).mean(dim=1)
        return self.out(out)


def main():
    x = torch.randn(2, 12, 32)
    m = MLPMixer(12, 32, 32)
    m(x)


# if __name__ == '__main__':
#     main()
