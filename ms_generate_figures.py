import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from mindspore import Tensor

from ms_eval_metrics import build_sampler
from ms_postproc_utils import PostProcess


def save_soh_prediction_figure(refs, preds, output_path, max_panels=12):
    n_samples = min(refs.shape[0], max_panels)
    cols = min(4, n_samples)
    rows = int(np.ceil(n_samples / cols))
    cycles = np.arange(refs.shape[1]) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)
    for i in range(n_samples):
        axes[i].plot(cycles, refs[i], color="#00a6c8", lw=1.4, label="Reference")
        axes[i].plot(cycles, preds[i], color="#d14a9c", lw=1.4, ls="--", label="Prediction")
        axes[i].axhline(80, color="#555555", lw=0.8, ls=":")
        axes[i].set_title(f"Sample {i}", fontsize=9)
        axes[i].set_xlim(0, min(2200, refs.shape[1]))
        axes[i].set_ylim(-5, 105)
        axes[i].grid(alpha=0.25)
    for ax in axes[n_samples:]:
        ax.set_axis_off()
    axes[0].legend(fontsize=8, loc="lower left")
    fig.supxlabel("Cycle")
    fig.supylabel("SOH (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def estimate_x0(sampler, samples, timesteps, capacity, guide_w):
    batch_size = samples.shape[0]
    ones = ms.ops.ones((batch_size, sampler.first_channels * 4), ms.float32)
    zeros = ms.ops.zeros((batch_size, sampler.first_channels * 4), ms.float32)
    pred_cond = sampler.ema_network(samples, timesteps, capacity, ones)
    pred_uncond = sampler.ema_network(samples, timesteps, capacity, zeros)
    pred_noise = (1.0 + guide_w) * pred_cond - guide_w * pred_uncond
    return sampler.gdf_util.predict_start_from_noise(samples, timesteps, pred_noise)


def save_denoising_figure(
    sampler,
    capacity_matrix,
    reference_curve,
    output_path,
    checkpoints,
    mode="sample",
    guide_w=0.0,
):
    capacity = Tensor(capacity_matrix[None, ...].astype(np.float32), ms.float32)
    samples = ms.ops.StandardNormal()((1, sampler.sample_length, 1))
    snapshots = {0: samples.asnumpy()[0, :, 0]}
    reference_soh = (reference_curve.reshape(-1) + 1.0) / 2.0 * 100.0

    wanted = set(checkpoints)
    for step, t in enumerate(reversed(range(sampler.timesteps)), start=1):
        tt = Tensor(np.full((1,), t, dtype=np.int32), ms.int32)
        if step in wanted or step == sampler.timesteps:
            if mode == "x0":
                x0_est = estimate_x0(sampler, samples, tt, capacity, guide_w)
                snapshots[step] = x0_est.asnumpy()[0, :, 0]
            else:
                snapshots[step] = samples.asnumpy()[0, :, 0]
        samples = sampler.generate(samples, tt, capacity, guide_w=guide_w)
        if step % 50 == 0 or step == sampler.timesteps:
            print(f"denoising figure step {step}/{sampler.timesteps}", flush=True)

    keys = sorted(snapshots.keys())
    cols = len(keys)
    cycles = np.linspace(1, 2560, sampler.sample_length)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.55, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        soh = (snapshots[key] + 1.0) / 2.0 * 100.0
        print(
            f"snapshot {key}: SOH min={np.nanmin(soh):.2f}, max={np.nanmax(soh):.2f}, std={np.nanstd(soh):.2f}",
            flush=True,
        )
        soh = np.clip(soh, 45, 110)
        ax.plot(cycles, soh, color="#e052b5", lw=0.8, alpha=0.9)
        ax.plot(cycles, reference_soh, color="#00b8c8", lw=1.8)
        ax.set_title(f"t = {key}", fontsize=9, fontstyle="italic")
        ax.set_ylim(-5, 105)
        ax.set_xlim(cycles[0], cycles[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.25)
    axes[0].set_yticks([0, 50, 80, 100])
    axes[0].set_ylabel("SOH(%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate(args):
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = build_sampler(args)
    post_process = PostProcess(args.test_npz, sampler, max_samples=args.max_samples)
    refs, preds = post_process.pred(
        guide_w=args.guide_w, reps=args.reps, progress_every=args.progress_every
    )

    soh_path = output_dir / "matr1_soh_prediction.png"
    save_soh_prediction_figure(refs, preds, soh_path, max_panels=args.max_panels)
    print(f"saved {soh_path}")

    denoise_path = output_dir / "matr1_denoising_steps.png"
    checkpoints = [int(x) for x in args.denoise_checkpoints.split(",") if x.strip()]
    save_denoising_figure(
        sampler,
        post_process.capacity_matrix[0],
        post_process.reference_soh[0],
        denoise_path,
        checkpoints,
        mode=args.denoise_mode,
        guide_w=args.guide_w,
    )
    print(f"saved {denoise_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate minimum-report DiffBatt figures.")
    parser.add_argument("--test-npz", default="data_npz/matr_1_test.npz")
    parser.add_argument("--ema-ckpt", default="checkpoints_ms/matr_1_v2")
    parser.add_argument("--output-dir", default="figures_ms")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--guide-w", type=float, default=0.0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--denoise-checkpoints", default="200,400,600,800,1000")
    parser.add_argument("--denoise-mode", choices=["sample", "x0"], default="x0")
    parser.add_argument("--clip-denoised", action="store_true")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
