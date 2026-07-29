import random

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CHANNELS = 1

# Product of the encoder strides. Input lengths must be a multiple of this for
# the decoder to land back on exactly the input length.
TOTAL_STRIDE = 8

# Lower bound on the spectral-convergence denominator. See STFTLoss.forward.
SILENT_TARGET_FLOOR = 1.0


class GeneralConv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=25, stride=1, padding=None,
                 dilation=1):
        super().__init__()
        kernel_size = kernel_size + (kernel_size - stride) % 2
        # A dilated kernel spans dilation*(k-1)+1 samples; pad against that span
        # so a stride-1 dilated conv is still length-preserving.
        effective_kernel = dilation * (kernel_size - 1) + 1
        if padding is None:
            padding = (effective_kernel - stride) // 2
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              padding_mode="zeros")

    def forward(self, x):
        conv = self.conv(x)
        return conv


class GeneralConv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                              stride=stride, padding_mode="reflect")  # TODO: padding for 2D

    def forward(self, x):
        conv = self.conv(x)
        return conv


class GeneralDeconv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=25, stride=1, padding=None):
        super().__init__()
        
        self.stride = stride
        
        self.kernel_size = kernel_size + (kernel_size - stride) % 2
        if padding is None:
            padding = (self.kernel_size - stride) // 2
        self.padding = padding
            
        self.conv = nn.Conv1d(
            in_channels=in_features, 
            out_channels=out_features, 
            kernel_size=self.kernel_size,
            stride=1, 
            padding=0 
        )

    def forward(self, x):
        L_in = x.size(-1)
        L_target = (L_in - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        if self.stride > 1:
            x = F.interpolate(
                x,
                scale_factor=self.stride,
                mode="linear",
                align_corners=False
            )
            
        L_interp = x.size(-1)
        total_padding = L_target + self.kernel_size - 1 - L_interp
        
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        
        x = F.pad(x, (pad_left, pad_right), mode="reflect")
        
        return self.conv(x)


class GeneralDeconv2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size,
                                         stride=stride, padding_mode="reflect")  # TODO: padding for 2D

    def forward(self, x):
        deconv = self.deconv(x)
        return deconv


class ResnetBlock(nn.Module):
    def __init__(self, num_features, dilation=1, kernel_size=9):
        super().__init__()
        self.resnet = nn.Sequential(
            GeneralConv1D(num_features, num_features,
                          kernel_size=kernel_size, dilation=dilation),
            nn.InstanceNorm1d(num_features),
            nn.ReLU(),

            GeneralConv1D(num_features, num_features,
                          kernel_size=kernel_size, dilation=dilation),
            nn.InstanceNorm1d(num_features)
        )

    def forward(self, x):
        resnet_result = self.resnet(x)
        return resnet_result + x


class Generator(nn.Module):

    initial_features = 32

    # Receptive field comes from dilation, not from stride. Buying it with
    # stride instead crushes the bottleneck: at 32x downsampling the middle of
    # the network runs at a 500 Hz frame rate, and the decoder cannot rebuild
    # the detail -- measured 1-2 kHz output 18.7 dB below the input, which the
    # discriminator spots instantly. At 8x with these dilations the same ~1 s of
    # context costs only 2.2 dB there.
    #
    # Small kernels are also the cheap way to reach: doubling dilations with
    # kernel 9 covers more ground per FLOP than kernel 25 ever did.
    dilations = (1, 2, 4, 8, 16, 32, 64)
    resnet_kernel = 9

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, self.initial_features, stride=2),
            nn.ReLU(),

            GeneralConv1D(self.initial_features,
                          self.initial_features * 2, stride=2),
            nn.InstanceNorm1d(self.initial_features * 2),
            nn.ReLU(),

            GeneralConv1D(self.initial_features * 2,
                          self.initial_features * 4, stride=2),
            nn.InstanceNorm1d(self.initial_features * 4),
            nn.ReLU()
        )

        self.transformer = nn.Sequential(
            *[ResnetBlock(self.initial_features * 4, dilation=dilation,
                          kernel_size=self.resnet_kernel)
              for dilation in self.dilations])

        self.decoder = nn.Sequential(
            GeneralDeconv1D(self.initial_features * 4,
                            self.initial_features * 2, stride=2),
            nn.InstanceNorm1d(self.initial_features * 2),
            nn.ReLU(),

            GeneralDeconv1D(self.initial_features * 2,
                            self.initial_features, stride=2),
            nn.InstanceNorm1d(self.initial_features),
            nn.ReLU(),

            GeneralDeconv1D(self.initial_features,
                            self.initial_features, stride=2),
            nn.InstanceNorm1d(self.initial_features),
            nn.ReLU(),

            GeneralConv1D(self.initial_features, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.transformer(self.encoder(x)))


class STFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 at one STFT resolution."""

    def __init__(self, fft_size, hop_size, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def magnitude(self, x):
        spectrogram = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            # .to() is a no-op once they match; it keeps the loss usable even
            # if the caller forgets to move this module onto the GPU.
            window=self.window.to(x.device),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        return spectrogram.abs().clamp(min=1e-7)

    def forward(self, x, target):
        x_magnitude = self.magnitude(x)
        target_magnitude = self.magnitude(target)

        # Spectral convergence divides by the target's magnitude, so a silent
        # target sends it to ~1e7 and one bad batch wrecks Adam's moments. Real
        # RMS-normalized crops measure ~500 here, so a floor of 1.0 is under
        # 0.3% of a normal target and only ever engages on near-silence.
        convergence = (
            torch.linalg.matrix_norm(target_magnitude - x_magnitude)
            / torch.linalg.matrix_norm(target_magnitude).clamp(min=SILENT_TARGET_FLOOR)
        ).mean()
        log_magnitude = F.l1_loss(x_magnitude.log(), target_magnitude.log())
        return convergence + log_magnitude


class MultiResolutionSTFTLoss(nn.Module):
    """Averages STFTLoss over three resolutions.

    A single window forces a fixed time/frequency trade-off; three lets the
    loss see both transients (short window) and pitch (long window).
    """

    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_lengths=(512, 1024, 2048)):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(fft_size, hop_size, win_length)
            for fft_size, hop_size, win_length
            in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x, target):
        # torch.stft has no half-precision CUDA kernel, so step outside autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.squeeze(1).float()
            target = target.squeeze(1).float()
            return sum(loss(x, target) for loss in self.losses) / len(self.losses)


class GeneratorLoss(nn.Module):

    def __init__(self, discriminator, opposing_generator, cycle_consistency_factor=5,
                 waveform_factor=0.1):
        super().__init__()
        self.discriminator = discriminator
        self.opposing_generator = opposing_generator
        self.cycle_consistency_factor = cycle_consistency_factor
        self.waveform_factor = waveform_factor

        self.mse = nn.MSELoss()
        self.l1Loss = nn.L1Loss()
        self.stft = MultiResolutionSTFTLoss()

    def forward(self, x, original):
        # Reach through DDP to the raw discriminator. This pass only needs the
        # verdict, not synchronised discriminator gradients, and skipping the
        # wrapper means main.py can freeze the discriminator here without
        # tripping DDP's "parameter received no gradient" check.
        discriminator = getattr(self.discriminator, "module", self.discriminator)
        disc_outputs = as_scale_list(discriminator(x))
        gan_loss = sum(self.mse(out, torch.ones_like(out))
                       for out in disc_outputs) / len(disc_outputs)

        # Cycle consistency is measured spectrally. Waveform L1 alone cannot
        # tell a harmonic signal from noise with the same envelope, so it was
        # rewarding exactly the output we were trying to get rid of. The small
        # waveform term is kept only to anchor overall amplitude.
        reconstruction = self.opposing_generator(x)
        cycle_consistency_loss = (
            self.stft(reconstruction, original)
            + self.waveform_factor * self.l1Loss(reconstruction, original)
        )
        return gan_loss + self.cycle_consistency_factor * cycle_consistency_loss


class Discriminator(nn.Module):

    # Deliberately narrower than the generator. Telling real audio from fake is
    # a much easier job than synthesising it, so matching their widths hands the
    # discriminator the argument -- at 64 it was 4.5M parameters per scale
    # against the generator's 4.7M total, and it won every time.
    initial_features = 32
    relu_factor = .2

    def __init__(self, initial_features=None):
        super().__init__()
        features = initial_features or self.initial_features
        self.nn = nn.Sequential(
            GeneralConv1D(NUM_CHANNELS, features,
                          kernel_size=25, stride=4),
            nn.LeakyReLU(self.relu_factor),

            GeneralConv1D(features,
                          features * 2, kernel_size=25, stride=4),
            nn.InstanceNorm1d(features * 2, affine=True),
            nn.LeakyReLU(self.relu_factor),

            GeneralConv1D(features * 2,
                          features * 4, kernel_size=25, stride=4),
            nn.InstanceNorm1d(features * 4, affine=True),
            nn.LeakyReLU(self.relu_factor),

            GeneralConv1D(features * 4, features *
                          8, kernel_size=25, stride=4, padding=0),
            nn.InstanceNorm1d(features * 8, affine=True),
            nn.LeakyReLU(self.relu_factor),

            GeneralConv1D(features * 8, 1,
                          kernel_size=25, stride=1, padding=0)
        )

    def forward(self, x):
        return self.nn(x)


class MultiScaleDiscriminator(nn.Module):
    """One Discriminator per time scale, judging the same audio at 1x, 2x, 4x.

    A single-scale waveform discriminator tends to fixate on local texture,
    which the generator can satisfy with the right kind of hiss. Forcing the
    same verdict at several rates makes it judge structure instead.
    """

    def __init__(self, num_scales=3, initial_features=None):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator(initial_features) for _ in range(num_scales)])
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(x))
            if index < len(self.discriminators) - 1:
                x = self.downsample(x)
        return outputs


def as_scale_list(outputs):
    """Accept either a single discriminator output or one per scale."""
    return outputs if isinstance(outputs, (list, tuple)) else [outputs]


class DiscriminatorLoss(nn.Module):

    def __init__(self, real_label=0.9):
        super().__init__()
        # Least-squares GAN. BCE saturates once the discriminator pulls ahead,
        # which starves the generator of gradient exactly when it needs it most.
        self.mse = nn.MSELoss()
        # One-sided label smoothing: real audio is labelled 0.9, not 1.0, so the
        # discriminator is never rewarded for total certainty. The generator
        # still aims at 1.0, which keeps a gradient alive even when it is losing.
        self.real_label = real_label

    def forward(self, outputs, is_real):
        target = self.real_label if is_real else 0.0
        outputs = as_scale_list(outputs)
        return sum(self.mse(out, torch.full_like(out, target))
                   for out in outputs) / len(outputs)


class ReplayPool:
    """Buffer of previously generated samples, as in the original CycleGAN.

    Showing the discriminator only the generator's newest output lets the two
    chase each other: the discriminator specialises on the current artefact,
    the generator moves, and neither settles. Mixing in older fakes keeps the
    discriminator honest about a history of outputs.
    """

    def __init__(self, capacity=50):
        self.capacity = capacity
        self.items = []

    def query(self, batch):
        if self.capacity == 0:
            return batch

        chosen = []
        for sample in batch:
            sample = sample.unsqueeze(0)
            if len(self.items) < self.capacity:
                self.items.append(sample.detach().clone())
                chosen.append(sample)
            elif random.random() < 0.5:
                index = random.randrange(self.capacity)
                chosen.append(self.items[index].clone())
                self.items[index] = sample.detach().clone()
            else:
                chosen.append(sample)
        return torch.cat(chosen, dim=0)