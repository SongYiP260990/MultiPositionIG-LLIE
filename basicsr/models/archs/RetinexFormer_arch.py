import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"invalid mode {mode}")

    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def shift_back(inputs, step=2):
    bs, nC, row, col = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[
            :, i, :, int(step * i): int(step * i) + out_col
        ]
    return inputs[:, :, :, :out_col]


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        inp = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(inp)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class MSIFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.low_freq = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=False
        )
        self.high_freq = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False)
        self.beta_scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)

    def forward(self, feat, illu_fea):
        lf = self.low_freq(illu_fea)
        hf = self.high_freq(illu_fea)
        gamma_beta = self.fuse(torch.cat([lf, hf], dim=1))
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return feat * (1 + torch.tanh(gamma)) + beta * self.beta_scale


class DecoupledIGAR(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(dim, num_heads, 1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, num_heads, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, illu_fea):
        b, _, h, w = illu_fea.shape
        spatial_gate = self.spatial_proj(illu_fea)
        channel_gate = self.channel_proj(illu_fea)
        gate = spatial_gate * channel_gate
        return gate.view(b, self.num_heads, 1, h * w)


class IGS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, skip_fea, illu_fea):
        gate = self.gate(illu_fea)
        return skip_fea * gate


# Backward-compatible aliases for old scripts/checkpoints that import class names.
FreqAwareIFM = MSIFM
IlluGatedSkip = IGS


class IG_MSA(nn.Module):
    # Current model: regulate the per-head output after attention mixing.
    # Position control: use the same D-IGAR gate to regulate normalized Q/K
    # immediately before attention-logit formation.
    VALID_DIGAR_POSITIONS = {"post_attention", "pre_attention_qk"}
    LEGACY_DIGAR_ALIASES = {"pre_attention_qkv": "pre_attention_qk"}

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        digar_position="post_attention",
    ):
        super().__init__()
        if digar_position in self.LEGACY_DIGAR_ALIASES:
            mapped = self.LEGACY_DIGAR_ALIASES[digar_position]
            warnings.warn(
                f"digar_position={digar_position!r} is a legacy name and is "
                f"mapped to {mapped!r}. The moved variant gates normalized "
                "Q/K only; it does not apply an additional D-IGAR gate to V.",
                stacklevel=2,
            )
            digar_position = mapped
        if digar_position not in self.VALID_DIGAR_POSITIONS:
            raise ValueError(
                f"Unsupported digar_position={digar_position!r}. "
                f"Expected one of {sorted(self.VALID_DIGAR_POSITIONS)}."
            )
        self.num_heads = heads
        self.dim_head = dim_head
        self.digar_position = digar_position

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.illu_attn_gate = DecoupledIGAR(dim, heads)

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v, illu_attn = map(
            lambda t: rearrange(t, "b n (head d) -> b head n d", head=self.num_heads),
            (q_inp, k_inp, v_inp, illu_fea_trans.flatten(1, 2)),
        )

        # The same D-IGAR module and parameters are used in both placements.
        # Shape: [B, heads, 1, HW].
        illu_gate = self.illu_attn_gate(illu_fea_trans.permute(0, 3, 1, 2))

        # Official RetinexFormer value modulation.
        v = v * illu_attn

        # Official RetinexFormer attention path.
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        if self.digar_position == "pre_attention_qk":
            # Position-matched control: regulate attention formation after Q/K
            # normalization and immediately before the attention logits.
            # Applying sqrt(G) to both Q and K makes G enter the weighted
            # Q-K correlation once, rather than as G^2. The official value
            # modulation above is retained unchanged; no extra D-IGAR gate
            # is applied to V in this moved variant.
            qk_gate = torch.sqrt(illu_gate.clamp_min(1e-6))
            q = q * qk_gate
            k = k * qk_gate

        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v  # [B, heads, dim_head, HW]

        if self.digar_position == "post_attention":
            # Current paper placement: regulate the per-head attention output.
            x = x * illu_gate

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_blocks=2,
        digar_position="post_attention",
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            digar_position=digar_position,
                        ),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, illu_fea):
        x = x.permute(0, 2, 3, 1)
        illu_fea_trans = illu_fea.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea_trans) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    VALID_MSIFM_POSITIONS = {"decoder_end", "pre_last_decoder_block"}
    VALID_IGS_POSITIONS = {"pre_fusion", "post_fusion"}

    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        dim=31,
        level=2,
        num_blocks=[2, 4, 4],
        msifm_position="decoder_end",
        digar_position="post_attention",
        igs_position="pre_fusion",
    ):
        super().__init__()
        if msifm_position not in self.VALID_MSIFM_POSITIONS:
            raise ValueError(
                f"Unsupported msifm_position={msifm_position!r}. "
                f"Expected one of {sorted(self.VALID_MSIFM_POSITIONS)}."
            )
        if igs_position not in self.VALID_IGS_POSITIONS:
            raise ValueError(
                f"Unsupported igs_position={igs_position!r}. "
                f"Expected one of {sorted(self.VALID_IGS_POSITIONS)}."
            )
        if digar_position not in IG_MSA.VALID_DIGAR_POSITIONS:
            raise ValueError(
                f"Unsupported digar_position={digar_position!r}. "
                f"Expected one of {sorted(IG_MSA.VALID_DIGAR_POSITIONS)}."
            )

        self.dim = dim
        self.level = level
        self.msifm_position = msifm_position
        self.digar_position = digar_position
        self.igs_position = igs_position

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                            digar_position=digar_position,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
            digar_position=digar_position,
        )

        self.decoder_layers = nn.ModuleList([])
        self.skip_gates = nn.ModuleList([])
        for i in range(level):
            curr_dim = dim_level // 2
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            curr_dim,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, curr_dim, 1, 1, bias=False),
                        IGAB(
                            dim=curr_dim,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=curr_dim // dim,
                            digar_position=digar_position,
                        ),
                    ]
                )
            )
            self.skip_gates.append(IGS(curr_dim))
            dim_level //= 2

        self.illum_mod = MSIFM(self.dim)
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)

        fea_encoder = []
        illu_fea_list = []
        for igab, fea_down, illu_down in self.encoder_layers:
            fea = igab(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = fea_down(fea)
            illu_fea = illu_down(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (fea_up, fusion, block) in enumerate(self.decoder_layers):
            fea = fea_up(fea)

            skip = fea_encoder[self.level - 1 - i]
            illu_fea_level = illu_fea_list[self.level - 1 - i]

            if self.igs_position == "pre_fusion":
                # Current paper placement: gate only the encoder skip feature.
                skip = self.skip_gates[i](skip, illu_fea_level)

            fea = fusion(torch.cat([fea, skip], dim=1))

            if self.igs_position == "post_fusion":
                # Moved-position control: use the same gate after fusion.
                fea = self.skip_gates[i](fea, illu_fea_level)

            is_last_decoder_level = i == self.level - 1
            if (
                self.msifm_position == "pre_last_decoder_block"
                and is_last_decoder_level
            ):
                # Moved-position control: calibrate the highest-resolution feature
                # after fusion but before the final decoder IGAB.
                fea = self.illum_mod(fea, illu_fea_level)

            fea = block(fea, illu_fea_level)

        if self.msifm_position == "decoder_end":
            # Current paper placement: after the final decoder block.
            fea = self.illum_mod(fea, illu_fea_list[0])
        out = self.mapping(fea) + x
        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        level=2,
        num_blocks=[1, 1, 1],
        msifm_position="decoder_end",
        digar_position="post_attention",
        igs_position="pre_fusion",
    ):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
            msifm_position=msifm_position,
            digar_position=digar_position,
            igs_position=igs_position,
        )

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)
        return output_img


class RetinexFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        stage=3,
        num_blocks=[1, 1, 1],
        msifm_position="decoder_end",
        digar_position="post_attention",
        igs_position="pre_fusion",
    ):
        super().__init__()
        self.stage = stage
        modules_body = [
            RetinexFormer_Single_Stage(
                in_channels=in_channels,
                out_channels=out_channels,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
                msifm_position=msifm_position,
                digar_position=digar_position,
                igs_position=igs_position,
            )
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


if __name__ == "__main__":
    # Lightweight CPU smoke test. Formal experiments should use the paper
    # configuration (stage=1, n_feat=40, num_blocks=[1, 2, 2]) in YAML.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.randn((1, 3, 8, 8))

    variants = {
        "current_full": dict(
            msifm_position="decoder_end",
            digar_position="post_attention",
            igs_position="pre_fusion",
        ),
        "move_msifm": dict(
            msifm_position="pre_last_decoder_block",
            digar_position="post_attention",
            igs_position="pre_fusion",
        ),
        "move_digar": dict(
            msifm_position="decoder_end",
            digar_position="pre_attention_qk",
            igs_position="pre_fusion",
        ),
        "move_igs": dict(
            msifm_position="decoder_end",
            digar_position="post_attention",
            igs_position="post_fusion",
        ),
    }

    reference = RetinexFormer(
        stage=1,
        n_feat=8,
        num_blocks=[1, 1, 1],
        **variants["current_full"],
    ).eval()
    shared_state = reference.state_dict()
    expected_params = sum(p.numel() for p in reference.parameters())

    with torch.no_grad():
        for name, placement in variants.items():
            model = RetinexFormer(
                stage=1,
                n_feat=8,
                num_blocks=[1, 1, 1],
                **placement,
            ).eval()
            model.load_state_dict(shared_state, strict=True)
            out = model(x)
            params = sum(p.numel() for p in model.parameters())
            assert params == expected_params, (name, params, expected_params)
            assert out.shape == x.shape, (name, out.shape, x.shape)
            assert torch.isfinite(out).all(), name
            print(
                f"{name:14s} | out={tuple(out.shape)} | "
                f"params_equal=True | {placement}"
            )

    # Parameter check for the actual paper configuration; no expensive forward.
    actual_counts = {}
    for name, placement in variants.items():
        model = RetinexFormer(
            stage=1,
            n_feat=40,
            num_blocks=[1, 2, 2],
            **placement,
        )
        actual_counts[name] = sum(p.numel() for p in model.parameters())
    assert len(set(actual_counts.values())) == 1, actual_counts
    print(
        "paper_config | all variants params="
        f"{next(iter(actual_counts.values())) / 1e6:.6f}M"
    )
