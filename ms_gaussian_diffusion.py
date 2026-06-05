import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


class GaussianDiffusion:
    """DDPM forward and reverse-process math implemented with MindSpore ops."""

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.betas = Tensor(betas, ms.float32)
        self.alphas_cumprod = Tensor(alphas_cumprod, ms.float32)
        self.alphas_cumprod_prev = Tensor(alphas_cumprod_prev, ms.float32)
        self.sqrt_alphas_cumprod = Tensor(np.sqrt(alphas_cumprod), ms.float32)
        self.sqrt_one_minus_alphas_cumprod = Tensor(
            np.sqrt(1.0 - alphas_cumprod), ms.float32
        )
        self.log_one_minus_alphas_cumprod = Tensor(
            np.log(1.0 - alphas_cumprod), ms.float32
        )
        self.sqrt_recip_alphas_cumprod = Tensor(
            np.sqrt(1.0 / alphas_cumprod), ms.float32
        )
        self.sqrt_recipm1_alphas_cumprod = Tensor(
            np.sqrt(1.0 / alphas_cumprod - 1.0), ms.float32
        )
        self.posterior_variance = Tensor(posterior_variance, ms.float32)
        self.posterior_log_variance_clipped = Tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), ms.float32
        )
        self.posterior_mean_coef1 = Tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            ms.float32,
        )
        self.posterior_mean_coef2 = Tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            ms.float32,
        )

    def _extract(self, values, timesteps, x_shape):
        batch_size = x_shape[0]
        out = ops.gather(values, timesteps, 0)
        return ops.reshape(out, (batch_size, 1, 1))

    def q_mean_variance(self, x_start, timesteps):
        x_shape = x_start.shape
        mean = self._extract(self.sqrt_alphas_cumprod, timesteps, x_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, timesteps, x_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, timesteps, x_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, timesteps, noise):
        x_shape = x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, timesteps, noise):
        x_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, timesteps):
        x_shape = x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, timesteps, x_shape) * x_start
            + self._extract(self.posterior_mean_coef2, timesteps, x_shape) * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance, timesteps, x_shape
        )
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, timesteps, x_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, pred_noise, x_t, timesteps, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x_t, timesteps, pred_noise)
        if clip_denoised:
            x_recon = ops.clip_by_value(x_recon, self.clip_min, self.clip_max)
        return self.q_posterior(x_recon, x_t, timesteps)

    def p_sample(self, pred_noise, x_t, timesteps, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x_t, timesteps, clip_denoised=clip_denoised
        )
        noise = ops.StandardNormal()(x_t.shape)
        nonzero_mask = 1.0 - ops.cast(ops.equal(timesteps, 0), ms.float32)
        nonzero_mask = ops.reshape(nonzero_mask, (x_t.shape[0], 1, 1))
        return model_mean + nonzero_mask * ops.exp(0.5 * model_log_variance) * noise
