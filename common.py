import jittor as jt
import numpy as np
import random
from math import sqrt
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight[0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        delattr(module, name)
        setattr(module, name + '_orig', weight)
        module.register_pre_forward_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)
    
def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class FusedUpsample(jt.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        weight = jt.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.weight=weight
        self.bias=bias
        self.pad = padding
        

    def execute(self, input):
        weight = jt.nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:] +
            weight[:, :, :-1, 1:] +
            weight[:, :, 1:, :-1] +
            weight[:, :, :-1, :-1]
        ) / 4

        out = jt.nn.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out

class FusedDownsample(jt.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        weight = jt.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.pad = padding
        self.weight = weight
        self.bias=bias

    def execute(self, input):
        weight = jt.nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]  +
            weight[:, :, :-1, 1:] +
            weight[:, :, 1:, :-1] +
            weight[:, :, :-1, :-1]
        ) / 4

        out = jt.nn.conv2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out
class PixelNorm(jt.Module):
    def __init__(self):
            pass
    def execute(self, input):
            return input / jt.sqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)

class BlurFunctionBackward(jt.Function):
    def execute(self, grad_output, kernel, kernel_flip):
        self.saved_tensors = kernel, kernel_flip

        grad_input = jt.nn.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    def grad(self, gradgrad_output):
        kernel, kernel_flip = self.saved_tensors

        grad_input = jt.nn.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(jt.Function):
    def execute(self, input, kernel, kernel_flip):
        self.saved_tensors = kernel, kernel_flip

        output = jt.nn.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    def grad(self, grad_output):
        kernel, kernel_flip = self.saved_tensors

        grad_input = BlurFunctionBackward().execute(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction().apply

class Blur(jt.Module):
    def __init__(self, channel):
        weight = jt.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float32')
        weight = weight.reshape(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = jt.flip(weight, [2, 3])

        self._weight = weight.repeat(channel, 1, 1, 1)
        self._weight_flip = weight_flip.repeat(channel, 1, 1, 1)

    def execute(self, input):
        return blur(input, self._weight, self._weight_flip)
    
class EqualConv2d(jt.Module):
    def __init__(self, *args, **kwargs):
        conv = jt.nn.Conv2d(*args, **kwargs)
        jt.init.gauss_(conv.weight, 0, 1)
        jt.init.constant_(conv.bias,0)
        
        self.conv = equal_lr(conv)

    def execute(self, input):
        return self.conv(input)

class EqualLinear(jt.Module):
    def __init__(self, in_dim, out_dim):
        linear = jt.nn.Linear(in_dim, out_dim)
        jt.init.gauss_(linear.weight, 0, 1)
        jt.init.constant_(linear.bias, 0)

        self.linear = equal_lr(linear)

    def execute(self, input):
        return self.linear(input)
    
class ConvBlock(jt.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False
    ):
        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = jt.nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            jt.nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = jt.nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    jt.nn.LeakyReLU(0.2),
                )
            else:
                self.conv2 = jt.nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    jt.nn.AvgPool2d(2),
                    jt.nn.LeakyReLU(0.2),
                )
        else:
            self.conv2 = jt.nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                jt.nn.LeakyReLU(0.2),
            )
    def execute(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out

class AdaptiveInstanceNorm(jt.nn.Module):
    def __init__(self, in_channel, style_dim):
        self.norm = jt.nn.InstanceNorm2d(in_channel, affine=False)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def execute(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
    
class NoiseInjection(jt.Module):
    def __init__(self, channel):
        self.weight = jt.zeros((1, channel, 1, 1))

    def execute(self, image, noise):
        return image + self.weight * noise
    
class ConstantInput(jt.Module):
    def __init__(self, channel, size=4):
        self.input = jt.randn(1, channel, size, size)

    def execute(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConvBlock(jt.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            if upsample:
                if fused:
                    self.conv1 = jt.nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )
                else:
                    self.conv1 = jt.nn.Sequential(
                        jt.nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )
            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = jt.nn.LeakyReLU(0.2)

        self.conv2  = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = jt.nn.LeakyReLU(0.2)
    
    def execute(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out




