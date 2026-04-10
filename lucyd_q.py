import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True,
                 norm=True, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = []
        if transpose:
            # upsampling conv
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose1d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias
                )
            )
        else:
            layers.append(
                nn.Conv1d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias
                )
            )
        if norm:
            layers.append(nn.BatchNorm1d(out_channel))
        if relu:
            layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Hamming1Layer(nn.Module):
    """
    Hypercube (Hamming-1) neighborhood aggregation.
    Input:  x (B, C, L) where L = 2^n
    Output: y (B, C_out, L)
    """
    def __init__(self, n_bits=9, in_channels=1, out_channels=8, agg="mean"):
        super().__init__()
        self.n_bits = n_bits
        self.L = 1 << n_bits
        self.agg = agg

        # learnable weights: self + per-bit neighbor weights
        self.w_self = nn.Parameter(torch.tensor(1.0))
        self.w_bits = nn.Parameter(torch.zeros(n_bits))  # start near 0; learns how much to pull from each bit-flip

        # channel mixing after aggregation (like 1x1 conv)
        self.mix = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # precompute neighbor indices: neigh[s, b] = s xor (1<<b)
        s = torch.arange(self.L)
        neigh = []
        for b in range(n_bits):
            neigh.append(s ^ (1 << b))
        self.register_buffer("neigh_idx", torch.stack(neigh, dim=1))  # (L, n_bits)

    def forward(self, x):
        # x: (B, C, L)
        assert x.size(-1) == self.L, f"Expected length {self.L}, got {x.size(-1)}"

        # Gather Hamming-1 neighbors: (B, C, L, n_bits)
        xn = x[:, :, self.neigh_idx]  # advanced indexing

        # Weighted sum over bits
        # weights shape (1,1,1,n_bits) broadcasts across batch/channels/L
        w = self.w_bits.view(1, 1, 1, self.n_bits)
        neigh_sum = (xn * w).sum(dim=-1)  # (B, C, L)

        y = self.w_self * x + neigh_sum   # (B, C, L)

        # optional normalization of neighbor contribution
        if self.agg == "mean":
            y = y / (1.0 + self.n_bits)

        # channel mixing
        return self.mix(y)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # concat along channels
        return self.conv(x)


class RL_DIV(nn.Module):
    def __init__(self, channel):
        super(RL_DIV, self).__init__()
        self.conv1 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(1, channel, kernel_size=3, stride=1, relu=True)

    def forward(self, x, z):
        # z: features from update encoder
        z0 = self.conv1(z)
        z1 = torch.mean(z0, dim=1, keepdim=True)      # channel-wise mean (1, L)
        z1 = z1.clamp_min(1e-3)
        q = x / z1                          # element-wise division
        z2 = self.conv2(q)
        return z2


class LUCYD_Q(nn.Module):
    def __init__(self, num_res=1, in_channels=1, n_bits=9, base_channel=4):
        super(LUCYD_Q, self).__init__()

        self.n_bits = int(n_bits)
        self.L = 1 << self.n_bits
        self.in_channels = int(in_channels)
        self.base_channel = int(base_channel)

        # Hamming-aware stems (replace the first BasicConv in each branch)
        self.ham_corr_in = Hamming1Layer(n_bits=self.n_bits, in_channels=self.in_channels,
                                         out_channels=self.base_channel)
        self.ham_upd_in  = Hamming1Layer(n_bits=self.n_bits, in_channels=self.in_channels,
                                         out_channels=self.base_channel)
        

        self.Encoder = nn.ModuleList([
            EBlock(self.base_channel, num_res),          # for correction path
            EBlock(self.base_channel, num_res),          # for update path
            EBlock(self.base_channel * 2, num_res),      # bottleneck encoder
        ])

        # CORRECTION BRANCH
        # NOTE: we removed the original [0] BasicConv(in_channels -> base_channel) and replaced with ham_corr_in
        self.correction_branch = nn.ModuleList([
            BasicConv(self.base_channel, self.base_channel * 2, kernel_size=3, relu=True, stride=2),  # downsample
            BasicConv(self.base_channel * 2, self.base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(self.base_channel, self.in_channels, kernel_size=3, relu=False, stride=1),
        ])

        # UPDATE BRANCH
        # NOTE: removed original [0] BasicConv(in_channels -> base_channel); replaced with ham_upd_in
        self.update_branch = nn.ModuleList([
            BasicConv(self.base_channel, self.base_channel * 2, kernel_size=3, relu=True, stride=2),  # downsample
            RL_DIV(self.base_channel),
            BasicConv(self.base_channel * 2, self.base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(self.base_channel, self.base_channel, kernel_size=3, relu=True, stride=1),
        ])

        # BOTTLENECK
        self.bottleneck = nn.ModuleList([
            BasicConv(self.base_channel * 4, self.base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(self.base_channel * 2, self.base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(self.base_channel * 2, self.base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        # AFF blocks for multi-scale feature fusion
        self.AFFs = nn.ModuleList([
            AFF(self.base_channel * 3, self.base_channel),
            AFF(self.base_channel * 3, self.base_channel * 2),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(self.base_channel * 2, num_res),   # bottleneck decoder
            DBlock(self.base_channel, num_res),       # correction decoder
            DBlock(self.base_channel, num_res),       # update decoder
        ])

        # Output head is identity to produce raw logits
        self.up = nn.Identity()

    def forward(self, x):
        # x shape: (B, in_channels, L)
        if x.size(-1) != self.L:
            raise ValueError(f"LUCYD_Q expected length {self.L} (=2^{self.n_bits}), got {x.size(-1)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"LUCYD_Q expected {self.in_channels} input channels, got {x.size(1)}")

        # ----- CORRECTION PATH -----
        a1 = self.ham_corr_in(x)              # (B, base, L)
        a2 = self.Encoder[0](a1)              # (B, base, L)
        a3 = self.correction_branch[0](a2)    # (B, 2*base, L/2)

        # ----- UPDATE PATH -----
        b1 = self.ham_upd_in(x)               # (B, base, L)
        b2 = self.Encoder[1](b1)              # (B, base, L)
        b3 = self.update_branch[0](b2)        # (B, 2*base, L/2)

        # ----- BOTTLENECK ENCODER -----
        z0 = torch.cat([a3, b3], dim=1)       # (B, 4*base, L/2)
        z1 = self.bottleneck[0](z0)           # (B, 2*base, L/2)
        z2 = self.Encoder[2](z1)              # (B, 2*base, L/2)

        # ----- MULTI-SCALE FUSION -----
        az = F.interpolate(a2, scale_factor=0.5, mode='linear', align_corners=False)  # (B, base, L/2)
        za = F.interpolate(z2, scale_factor=2.0, mode='linear', align_corners=False)  # (B, 2*base, L)

        res1 = self.AFFs[0](a2, za)           # (B, base, L)
        res2 = self.AFFs[1](z2, az)           # (B, 2*base, L/2)

        # ----- BOTTLENECK DECODER -----
        z3 = self.bottleneck[1](res2)         # (B, 2*base, L/2)
        z4 = self.Decoder[0](z3)              # (B, 2*base, L/2)
        zT = self.bottleneck[2](z4)           # (B, base, L)

        # ----- CORRECTION DECODER -----
        a_ = torch.cat([res1, zT], dim=1)     # (B, 2*base, L)
        a4 = self.correction_branch[1](a_)    # (B, base, L)
        a5 = self.Decoder[1](a4)              # (B, base, L)
        cor = self.correction_branch[2](a5)   # (B, in_channels, L)

        # Residual correction (note: if in_channels>1, correction applies to all channels;
        # for RO-conditioning you typically want correction only on the first channel.
        # For now assume in_channels==1.)
        y_k = x[:, :1, :] + cor[:, :1, :]     # keep it as (B,1,L) logits path

        # ----- UPDATE MODULE -----
        b4 = self.update_branch[1](x[:, :1, :], b2)   # RL_DIV expects x as (B,1,L)
        b_ = torch.cat([b4, zT], dim=1)               # (B, 2*base, L)
        b5 = self.update_branch[2](b_)                # (B, base, L)
        b6 = self.Decoder[2](b5 + b4)                 # (B, base, L)
        up = self.update_branch[3](b6)                # (B, base, L)

        up = torch.mean(up, dim=1, keepdim=True)      # (B, 1, L)
        up = 1.0 + 0.1 * torch.tanh(up) 

        y = y_k * up                                  # (B, 1, L) raw logits
        y = self.up(y)

        return y, y_k, up
    
    