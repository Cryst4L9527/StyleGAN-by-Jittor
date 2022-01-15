import jittor as jt
import numpy as np
import random
from math import sqrt
from common import *

class Generator(jt.Module):
    def __init__(self, code_dim, fused=True):
        self.progression = jt.nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),   # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
                StyledConvBlock(128,  64, 3, 1, upsample=True, fused=fused),  # 256
                StyledConvBlock( 64,  32, 3, 1, upsample=True, fused=fused),  # 512
                StyledConvBlock( 32,  16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_rgb = jt.nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

    def execute(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]
            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out
                
            out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = jt.nn.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        return out

class StyledGenerator(jt.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(jt.nn.LeakyReLU(0.2))

        self.style = jt.nn.Sequential(*layers)

    def execute(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(jt.randn(batch, 1, size, size))

        if mean_style is not None:
            styles_norm = []
            
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdims=True)

        return style
    
class Discriminator(jt.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        self.progression = jt.nn.ModuleList(
            [
                ConvBlock( 16,  32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock( 32,  64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock( 64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )
        
        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return jt.nn.Sequential(EqualConv2d(3, out_channel, 1), jt.nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = jt.nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )
        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)
        
    def execute(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = np.std(out.data, axis=0)
                mean_std = jt.array(out_std.mean())
                mean_std = mean_std.expand((out.size(0), 1, 4, 4))
                out = jt.concat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = jt.nn.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out
