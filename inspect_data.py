import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_snapshot(path):
    import tensorflow as tf

    return tf.data.Dataset.load(str(path))


def summarize_snapshot(path):
    dataset = load_snapshot(path)
    samples = []
    for item in dataset.unbatch():
        image, feature, protocol = item
        samples.append((image.numpy(), feature.numpy(), protocol.numpy()))

    images = np.asarray([x[0] for x in samples], dtype=np.float32)
    features = np.asarray([x[1] for x in samples], dtype=np.float32)
    protocols = np.asarray([x[2] for x in samples], dtype=np.float32)

    print(f"dataset: {path}")
    print(f"samples: {len(samples)}")
    print(f"images/SOH shape: {images.shape}")
    print(f"features shape: {features.shape}")
    print(f"capacity matrix shape: {protocols.shape}")
    print(f"images/SOH range: {images.min():.6f} to {images.max():.6f}")
    print(f"features range: {features.min():.6f} to {features.max():.6f}")
    print(f"capacity matrix range: {protocols.min():.6f} to {protocols.max():.6f}")
    print("\nfirst sample:")
    print(f"  SOH first 10 normalized values: {images[0, :10, 0]}")
    print(f"  feature vector: {features[0]}")
    print(f"  capacity matrix top-left 5x5:\n{protocols[0, :5, :5, 0]}")


def export_sample(path, index, output_dir):
    dataset = load_snapshot(path)
    samples = list(dataset.unbatch().as_numpy_iterator())
    image, feature, protocol = samples[index]

    output_dir.mkdir(parents=True, exist_ok=True)
    soh = image[:, 0]
    soh_percent = (soh + 1.0) / 2.0 * 100.0
    cycles = np.arange(len(soh)) + 1
    np.savetxt(
        output_dir / f"sample_{index}_soh.csv",
        np.column_stack([cycles, soh, soh_percent]),
        delimiter=",",
        header="cycle,soh_normalized,soh_percent",
        comments="",
    )
    np.savetxt(
        output_dir / f"sample_{index}_features.csv",
        feature.reshape(1, -1),
        delimiter=",",
    )
    np.savetxt(
        output_dir / f"sample_{index}_capacity_matrix.csv",
        protocol[:, :, 0],
        delimiter=",",
    )
    print(f"exported sample {index} to {output_dir}")


def plot_sample(path, index, output_dir):
    dataset = load_snapshot(path)
    samples = list(dataset.unbatch().as_numpy_iterator())
    image, _, protocol = samples[index]

    output_dir.mkdir(parents=True, exist_ok=True)
    soh = (image[:, 0] + 1.0) / 2.0 * 100.0
    cycles = np.linspace(1, 2560, len(soh))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(cycles, soh, color="#00a6c8", lw=1.8)
    axes[0].axhline(80, color="#555555", lw=0.9, ls=":")
    axes[0].set_title("SOH curve")
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel("SOH (%)")
    axes[0].grid(alpha=0.25)

    im = axes[1].imshow(protocol[:, :, 0], cmap="viridis", aspect="auto")
    axes[1].set_title("Capacity matrix")
    axes[1].set_xlabel("Feature index")
    axes[1].set_ylabel("Cycle index")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    output_path = output_dir / f"sample_{index}_overview.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"saved {output_path}")


def summarize_all(data_dir):
    for path in sorted(Path(data_dir).glob("*_ds")):
        dataset = load_snapshot(path)
        count = 0
        shapes = None
        for item in dataset.unbatch():
            count += 1
            if shapes is None:
                shapes = tuple(x.shape for x in item)
        print(f"{path.name:18s} samples={count:4d} shapes={shapes}")


def main():
    parser = argparse.ArgumentParser(description="Inspect TensorFlow snapshot battery datasets.")
    parser.add_argument("--dataset", default="data/matr_1_train_ds")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="data_inspection")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.all:
        summarize_all("data")
        return

    dataset_path = Path(args.dataset)
    summarize_snapshot(dataset_path)
    output_dir = Path(args.output_dir)
    if args.export:
        export_sample(dataset_path, args.index, output_dir)
    if args.plot:
        plot_sample(dataset_path, args.index, output_dir)


if __name__ == "__main__":
    main()
