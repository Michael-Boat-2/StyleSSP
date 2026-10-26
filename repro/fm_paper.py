

import torch
import torch.fft as fft


def freq_paper(z, alpha=0.7, cutoff=0.3, noise_sigma=0.3, seed=1234):
    """Frequency manipulation as written in the paper (Eq. 5, 7, 9):
    2D FFT over (H, W), Gaussian low-pass filter, low frequencies scaled by alpha,
    then N(0, noise_sigma^2) * (1 - alpha) added. cutoff is not given in the paper."""
    dtype = z.dtype
    x = z.float()
    H, W = x.shape[-2], x.shape[-1]

    fy = 2 * torch.arange(H, device=x.device) / H - 1          # same frequency coordinates as the authors' filters
    fx = 2 * torch.arange(W, device=x.device) / W - 1
    d2 = fy[:, None] ** 2 + fx[None, :] ** 2
    lpf = torch.exp(-d2 / (2 * cutoff ** 2))                     # 1 at the centre, falls off outward

    f = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    f = alpha * f * lpf + f * (1 - lpf)                          # Eq. 5: alpha * f*L + f*H
    x_low_reduced = fft.ifft2(fft.ifftshift(f, dim=(-2, -1)), dim=(-2, -1)).real   # Eq. 7

    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device) * noise_sigma
    return (x_low_reduced + noise * (1 - alpha)).to(dtype)      # Eq. 9
