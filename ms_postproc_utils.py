import numpy as np
from matplotlib import pyplot as plt


class PostProcess:
    def __init__(self, test_npz_path, diffusion_sampler, max_samples=None):
        arrays = np.load(test_npz_path)
        self.reference_soh = arrays["images"].astype(np.float32)
        self.capacity_matrix = arrays["protocols"].astype(np.float32)
        if max_samples is not None:
            self.reference_soh = self.reference_soh[:max_samples]
            self.capacity_matrix = self.capacity_matrix[:max_samples]
        self.cycles = np.arange(0, 2560) + 1
        self.n = len(self.reference_soh)
        self.model = diffusion_sampler

    def gen(self, guide_w=0.0, reps=1, progress_every=50):
        protos = np.repeat(self.capacity_matrix, reps, axis=0)
        return self.model.generate_samples(
            protos, guide_w=guide_w, progress_every=progress_every
        ).asnumpy()

    @staticmethod
    def _resize_curves(curves, target_length=2560):
        src_x = np.linspace(0.0, 1.0, curves.shape[1])
        dst_x = np.linspace(0.0, 1.0, target_length)
        flat = curves.reshape((-1, curves.shape[1]))
        resized = np.stack([np.interp(dst_x, src_x, row) for row in flat], axis=0)
        return resized.reshape((*curves.shape[:-2], target_length, curves.shape[-1]))

    def pred(self, guide_w=0.0, reps=1, progress_every=50):
        refs = self._resize_curves(self.reference_soh, 2560).reshape((self.n, -1))
        refs = (refs + 1.0) / 2.0 * 100.0

        samples = self.gen(guide_w, reps=reps, progress_every=progress_every)
        samples = self._resize_curves(samples, 2560)
        samples = (samples.reshape((self.n, reps, -1)) + 1.0) / 2.0 * 100.0

        rmse_100 = []
        for i in range(reps):
            rmse_100.append(self.soh_rmse(refs[:, :100], samples[:, i, :100], 80))
        chosen = np.argmin(np.asarray(rmse_100).T, axis=1)
        preds = np.asarray([samples[i, chosen[i]] for i in range(self.n)])
        return refs, preds

    def soh_rmse(self, ref, pred, eol):
        refc = ref.copy()
        refc[refc <= 60] = np.nan
        predc = pred.copy()
        predc[predc <= eol] = np.nan
        return np.sqrt(np.nanmean((refc - predc) ** 2, axis=1))

    def eval_soh(self, refs, preds, eol=80):
        return self.soh_rmse(refs, preds, eol).mean()

    def get_rul(self, data, eol=80):
        rul_index = np.argmin(data > eol, axis=1)
        has_crossing = rul_index != 0
        rul_index = np.where(has_crossing, rul_index, -1)
        return self.cycles[rul_index]

    def eval_rul(self, refs, preds, eol=80):
        refs = refs.copy()
        refs[refs == 0.0] = np.nan
        rul_ref = self.get_rul(refs, eol=eol)
        rul_pred = self.get_rul(preds, eol=eol)
        return np.sqrt(np.mean((rul_ref - rul_pred) ** 2))

    def plot_sample(self, ref, pred):
        n = int(np.ceil(np.sqrt(ref.shape[0])))
        ref = ref.copy()
        pred = pred.copy()
        ref[ref <= 60] = np.nan
        pred[pred <= 60] = np.nan
        fig, ax = plt.subplots(n, n, figsize=(n / 2, n / 2), sharey=True)
        axes = ax.flatten() if hasattr(ax, "flatten") else [ax]
        for i in range(ref.shape[0]):
            axes[i].plot(self.cycles, ref[i], c="cyan", lw=1.5, label="Reference")
            axes[i].plot(self.cycles, pred[i], c="magenta", lw=1.5, ls="--", label="Prediction")
            axes[i].set_ylim(80, 105)
            axes[i].set_xticks([])
            axes[i].grid()
        for i in range(ref.shape[0], len(axes)):
            axes[i].set_axis_off()
        plt.show()
