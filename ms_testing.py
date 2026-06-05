import argparse
from datetime import datetime
from pathlib import Path

import mindspore as ms

from ms_diffusion_model import DiffusionSampler
from ms_gaussian_diffusion import GaussianDiffusion
from ms_postproc_utils import PostProcess
from ms_unet import build_model


def make_logger(log_file):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")

    def log(message):
        print(message)
        handle.write(f"{message}\n")
        handle.flush()

    return log, handle


def resolve_ema_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("ema_network_epoch*.ckpt"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"Cannot find EMA checkpoint from: {path}")


def evaluate(args):
    if args.log_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs_ms/test_{stamp}.txt"
    log, log_handle = make_logger(args.log_file)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(args.device)

    try:
        log("DiffBatt MindSpore testing")
        log(f"test_npz={args.test_npz}")
        log(f"ema_ckpt={args.ema_ckpt}")
        log(f"device={args.device}, timesteps={args.timesteps}, reps={args.reps}, guide_w={args.guide_w}")
        first_conv_channels = 8
        widths = [first_conv_channels * mult for mult in [1, 2, 4, 8]]
        has_attention = [False, False, True, True]
        ema_network = build_model(
            sequence_length=args.sequence_length,
            widths=widths,
            has_attention=has_attention,
            first_conv_channels=first_conv_channels,
            num_res_blocks=args.num_res_blocks,
        )
        ema_ckpt = resolve_ema_checkpoint(args.ema_ckpt)
        log(f"loading EMA checkpoint: {ema_ckpt}")
        params = ms.load_checkpoint(str(ema_ckpt))
        ms.load_param_into_net(ema_network, params)

        diffusion = GaussianDiffusion(timesteps=args.timesteps)
        sampler = DiffusionSampler(
            ema_network,
            diffusion,
            timesteps=args.timesteps,
            first_channels=first_conv_channels,
            sample_length=args.sequence_length,
            clip_denoised=args.clip_denoised,
        )
        post_process = PostProcess(args.test_npz, sampler, max_samples=args.max_samples)
        log(
            f"evaluating {post_process.n} samples, timesteps={args.timesteps}, reps={args.reps}"
        )
        refs, preds = post_process.pred(
            guide_w=args.guide_w,
            reps=args.reps,
            progress_every=args.progress_every,
            progress_log=log,
        )
        soh_rmse = post_process.eval_soh(refs, preds)
        rul_rmse = post_process.eval_rul(refs, preds)
        log(f"RUL RMSE {rul_rmse} SOH RMSE {soh_rmse}")
        if args.plot:
            post_process.plot_sample(refs, preds)
    finally:
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MindSpore DiffBatt checkpoint.")
    parser.add_argument("--test-npz", default="data_npz/matr_1_test.npz")
    parser.add_argument("--ema-ckpt", default="checkpoints_ms/matr_1_v2")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--guide-w", type=float, default=0.0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--clip-denoised", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
