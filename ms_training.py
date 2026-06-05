import argparse
import math
from pathlib import Path
from datetime import datetime

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ms_data import create_dataset
from ms_diffusion_model import DiffusionWithLoss, update_ema
from ms_gaussian_diffusion import GaussianDiffusion
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


def train(args):
    if args.log_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs_ms/train_{stamp}.txt"
    log, log_handle = make_logger(args.log_file)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(args.device)
    log("DiffBatt MindSpore training")
    log(f"train_npz={args.train_npz}")
    log(f"val_npz={args.val_npz}")
    log(f"ckpt_dir={args.ckpt_dir}")
    log(f"device={args.device}")
    log(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    log(f"timesteps={args.timesteps}, p_uncond={args.p_uncond}, ema={args.ema}, clip_norm={args.clip_norm}")

    try:
        first_conv_channels = 8
        widths = [first_conv_channels * mult for mult in [1, 2, 4, 8]]
        has_attention = [False, False, True, True]

        network = build_model(
            sequence_length=args.sequence_length,
            widths=widths,
            has_attention=has_attention,
            first_conv_channels=first_conv_channels,
            num_res_blocks=args.num_res_blocks,
        )
        ema_network = build_model(
            sequence_length=args.sequence_length,
            widths=widths,
            has_attention=has_attention,
            first_conv_channels=first_conv_channels,
            num_res_blocks=args.num_res_blocks,
        )
        ms.load_param_into_net(ema_network, network.parameters_dict())

        diffusion = GaussianDiffusion(timesteps=args.timesteps)
        loss_cell = DiffusionWithLoss(
            network,
            diffusion,
            timesteps=args.timesteps,
            p_uncond=args.p_uncond,
            first_channels=first_conv_channels,
        )
        val_loss_cell = DiffusionWithLoss(
            network,
            diffusion,
            timesteps=args.timesteps,
            p_uncond=args.p_uncond,
            first_channels=first_conv_channels,
            drop_condition=False,
        )
        optimizer = nn.Adam(network.trainable_params(), learning_rate=args.learning_rate)
        grad_fn = ms.value_and_grad(loss_cell, None, optimizer.parameters)

        train_ds = create_dataset(args.train_npz, batch_size=args.batch_size, shuffle=True)
        val_ds = (
            create_dataset(args.val_npz, batch_size=args.batch_size, shuffle=False)
            if args.val_npz
            else None
        )
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            steps = 0
            for images, protocols in train_ds.create_tuple_iterator():
                loss, grads = grad_fn(images, protocols)
                loss_value = float(loss.asnumpy())
                if not math.isfinite(loss_value):
                    log(f"epoch {epoch} step {steps + 1}: non-finite loss={loss_value}, stop training")
                    return
                if args.clip_norm and args.clip_norm > 0:
                    grads = ops.clip_by_global_norm(grads, clip_norm=args.clip_norm)
                optimizer(grads)
                update_ema(network, ema_network, ema=args.ema)
                epoch_loss += loss_value
                steps += 1
                if steps % args.log_every == 0:
                    log(f"epoch {epoch} step {steps}: loss={loss_value:.6f}")

            avg_loss = epoch_loss / max(steps, 1)
            if val_ds is not None:
                val_total = 0.0
                val_steps = 0
                for images, protocols in val_ds.create_tuple_iterator():
                    val_loss = val_loss_cell(images, protocols)
                    val_loss_value = float(val_loss.asnumpy())
                    if not math.isfinite(val_loss_value):
                        log(f"epoch {epoch}: non-finite val_loss={val_loss_value}, stop training")
                        return
                    val_total += val_loss_value
                    val_steps += 1
                avg_val_loss = val_total / max(val_steps, 1)
                monitor_loss = avg_val_loss
                log(
                    f"epoch {epoch} finished: loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f}"
                )
            else:
                monitor_loss = avg_loss
                log(f"epoch {epoch} finished: loss={avg_loss:.6f}")

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                ckpt_suffix = f"epoch{epoch:04d}_loss{best_loss:.6f}"
                network_path = ckpt_dir / f"network_{ckpt_suffix}.ckpt"
                ema_path = ckpt_dir / f"ema_network_{ckpt_suffix}.ckpt"
                ms.save_checkpoint(network, str(network_path))
                ms.save_checkpoint(ema_network, str(ema_path))
                log(f"saved checkpoint: {network_path.name}, {ema_path.name}")
        log(f"training complete: best_monitor_loss={best_loss:.6f}")
    finally:
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Train DiffBatt with MindSpore.")
    parser.add_argument("--train-npz", default="data_npz/matr_1_train.npz")
    parser.add_argument("--val-npz", default="data_npz/matr_1_test.npz")
    parser.add_argument("--ckpt-dir", default="checkpoints_ms/matr_1_v2")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--p-uncond", type=float, default=0.2)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-file", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
