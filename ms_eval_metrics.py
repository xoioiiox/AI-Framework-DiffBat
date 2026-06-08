import argparse
from datetime import datetime
from pathlib import Path

import mindspore as ms
import numpy as np

from ms_diffusion_model import DiffusionSampler
from ms_gaussian_diffusion import GaussianDiffusion
from ms_postproc_utils import PostProcess
from ms_testing import resolve_ema_checkpoint
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


def build_sampler(args):
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
    print(f"loading EMA checkpoint: {ema_ckpt}")
    params = ms.load_checkpoint(str(ema_ckpt))
    ms.load_param_into_net(ema_network, params)

    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    return DiffusionSampler(
        ema_network,
        diffusion,
        timesteps=args.timesteps,
        first_channels=first_conv_channels,
        sample_length=args.sequence_length,
        clip_denoised=args.clip_denoised,
    )


def evaluate(args):
    if args.log_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs_ms/eval_{stamp}.txt"
    log, log_handle = make_logger(args.log_file)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(args.device)

    try:
        log("DiffBatt MindSpore evaluation")
        log(f"test_npz={args.test_npz}")
        log(f"ema_ckpt={args.ema_ckpt}")
        log(f"device={args.device}, timesteps={args.timesteps}, reps={args.reps}, guide_w={args.guide_w}")
        sampler = build_sampler(args)
        post_process = PostProcess(args.test_npz, sampler, max_samples=args.max_samples)
        refs, preds = post_process.pred(
            guide_w=args.guide_w,
            reps=args.reps,
            progress_every=args.progress_every,
            progress_log=log,
        )
        if args.debug:
            log("\nDebug ranges")
            log(f"refs shape: {refs.shape}, min={np.nanmin(refs):.4f}, max={np.nanmax(refs):.4f}")
            log(f"preds shape: {preds.shape}, min={np.nanmin(preds):.4f}, max={np.nanmax(preds):.4f}")
            for eol in [80]:
                valid_ref = refs > 60
                valid_pred = preds > eol
                valid_both = valid_ref & valid_pred
                log(f"valid ref points (>60): {valid_ref.sum()}")
                log(f"valid pred points (>{eol}): {valid_pred.sum()}")
                log(f"valid paired points: {valid_both.sum()}")
                log(f"first ref first 20: {refs[0, :20]}")
                log(f"first pred first 20: {preds[0, :20]}")

        log("\nEvaluation summary")
        log(f"samples: {post_process.n}")
        log(f"timesteps: {args.timesteps}")
        log(f"reps: {args.reps}")
        log(f"guidance w: {args.guide_w}")
        log(f"RUL RMSE @ EOL 80%: {post_process.eval_rul(refs, preds, eol=80):.4f}")
        log(f"SOH RMSE @ EOL 80%: {post_process.eval_soh(refs, preds, eol=80):.4f}")
        log("=" * 50)

        if args.all_eol:
            for eol in [90, 80, 70, 60]:
                log(f"SOH RMSE @ EOL {eol}%: {post_process.eval_soh(refs, preds, eol=eol):.4f}")
    finally:
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiffBatt MindSpore metrics.")
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
    parser.add_argument("--all-eol", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--clip-denoised", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
