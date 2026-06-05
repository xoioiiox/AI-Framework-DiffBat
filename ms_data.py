import argparse
from pathlib import Path

import numpy as np
import mindspore.dataset as ds


class NpzBatteryDataset:
    def __init__(self, npz_path):
        arrays = np.load(npz_path)
        self.images = arrays["images"].astype(np.float32)
        self.protocols = arrays["protocols"].astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.protocols[index]


def create_dataset(npz_path, batch_size=32, shuffle=True):
    dataset = ds.GeneratorDataset(
        NpzBatteryDataset(npz_path),
        column_names=["images", "protocols"],
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=False)


def export_tf_snapshot_to_npz(snapshot_path, output_path):
    import tensorflow as tf

    snapshot = tf.data.Dataset.load(str(snapshot_path))
    images = []
    protocols = []
    for image, _, protocol in snapshot.unbatch():
        images.append(image.numpy())
        protocols.append(protocol.numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=np.asarray(images, dtype=np.float32),
        protocols=np.asarray(protocols, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export the original TensorFlow Dataset snapshot to MindSpore-friendly NPZ."
    )
    parser.add_argument("--snapshot", required=True, help="Path such as data/matr_1_train_ds")
    parser.add_argument("--output", required=True, help="Output .npz path")
    args = parser.parse_args()
    export_tf_snapshot_to_npz(Path(args.snapshot), Path(args.output))


if __name__ == "__main__":
    main()
