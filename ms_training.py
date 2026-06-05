import argparse
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ms_data import create_dataset
from ms_diffusion_model import DiffusionWithLoss, update_ema
from ms_gaussian_diffusion import GaussianDiffusion
from ms_unet import build_model


def train(args):
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(args.device)

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
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.learning_rate)
    grad_fn = ms.value_and_grad(loss_cell, None, optimizer.parameters)

    train_ds = create_dataset(args.train_npz, batch_size=args.batch_size, shuffle=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        steps = 0
        for images, protocols in train_ds.create_tuple_iterator():
            loss, grads = grad_fn(images, protocols)
            optimizer(grads)
            update_ema(network, ema_network, ema=args.ema)
            epoch_loss += float(loss.asnumpy())
            steps += 1
            if steps % args.log_every == 0:
                print(f"epoch {epoch} step {steps}: loss={float(loss.asnumpy()):.6f}")

        avg_loss = epoch_loss / max(steps, 1)
        print(f"epoch {epoch} finished: loss={avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_suffix = f"epoch{epoch:04d}_loss{best_loss:.6f}"
            network_path = ckpt_dir / f"network_{ckpt_suffix}.ckpt"
            ema_path = ckpt_dir / f"ema_network_{ckpt_suffix}.ckpt"
            ms.save_checkpoint(network, str(network_path))
            ms.save_checkpoint(ema_network, str(ema_path))
            print(f"saved checkpoint: {network_path.name}, {ema_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Train DiffBatt with MindSpore.")
    parser.add_argument("--train-npz", default="data_npz/matr_1_train.npz")
    parser.add_argument("--ckpt-dir", default="checkpoints_ms/matr_1")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--p-uncond", type=float, default=0.2)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=10)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
