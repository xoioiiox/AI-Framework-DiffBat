import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class DiffusionWithLoss(nn.Cell):
    """MindSpore training cell that mirrors the Keras custom train_step."""

    def __init__(
        self,
        network,
        gdf_util,
        timesteps,
        p_uncond=0.2,
        first_channels=8,
    ):
        super().__init__()
        self.network = network
        self.gdf_util = gdf_util
        self.timesteps = timesteps
        self.p_uncond = p_uncond
        self.first_channels = first_channels
        self.loss_fn = nn.MSELoss()

    def construct(self, images, protocol):
        batch_size = images.shape[0]
        t = ops.randint(0, self.timesteps, (batch_size,), dtype=ms.int32)
        c_mask = ops.cast(ops.uniform((batch_size,), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) >= self.p_uncond, ms.float32)
        c_mask = ops.tile(c_mask[:, None], (1, self.first_channels * 4))
        noise = ops.StandardNormal()(images.shape)
        images_t = self.gdf_util.q_sample(images, t, noise)
        pred_noise = self.network(images_t, t, protocol, c_mask)
        return self.loss_fn(pred_noise, noise)


class DiffusionSampler:
    """Inference helper for classifier-free guided DDPM sampling."""

    def __init__(
        self,
        ema_network,
        gdf_util,
        timesteps,
        first_channels=8,
        sample_length=256,
    ):
        self.ema_network = ema_network
        self.gdf_util = gdf_util
        self.timesteps = timesteps
        self.first_channels = first_channels
        self.sample_length = sample_length

    def generate(self, samples, timesteps, capacity_matrices, guide_w):
        batch_size = samples.shape[0]
        ones = ops.ones((batch_size, self.first_channels * 4), ms.float32)
        zeros = ops.zeros((batch_size, self.first_channels * 4), ms.float32)
        pred_cond = self.ema_network(samples, timesteps, capacity_matrices, ones)
        pred_uncond = self.ema_network(samples, timesteps, capacity_matrices, zeros)
        pred_noise = (1.0 + guide_w) * pred_cond - guide_w * pred_uncond
        return self.gdf_util.p_sample(
            pred_noise, samples, timesteps, clip_denoised=False
        )

    def generate_samples(
        self, capacity_matrices, guide_w=0.0, record_samples=False, progress_every=50
    ):
        if not isinstance(capacity_matrices, Tensor):
            capacity_matrices = Tensor(capacity_matrices, ms.float32)
        else:
            capacity_matrices = ops.cast(capacity_matrices, ms.float32)

        num_samples = capacity_matrices.shape[0]
        samples = ops.StandardNormal()((num_samples, self.sample_length, 1))
        record = [samples]
        for step, t in enumerate(reversed(range(self.timesteps)), start=1):
            tt = Tensor(np.full((num_samples,), t, dtype=np.int32), ms.int32)
            samples = self.generate(samples, tt, capacity_matrices, guide_w)
            if record_samples:
                record.append(samples)
            if progress_every and (step == 1 or step % progress_every == 0 or step == self.timesteps):
                print(f"sampling step {step}/{self.timesteps} (t={t})", flush=True)
        if record_samples:
            return samples, record
        return samples


def update_ema(network, ema_network, ema=0.999):
    """Update EMA network parameters in-place."""
    params = network.get_parameters()
    ema_params = ema_network.get_parameters()
    for param, ema_param in zip(params, ema_params):
        ema_param.set_data(ema_param * ema + param * (1.0 - ema))
