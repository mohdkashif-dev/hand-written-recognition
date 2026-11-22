"""Train a CNN for single-letter recognition on EMNIST Letters."""
import argparse
import gzip
import json
import random
import string
import struct
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

LETTERS = string.ascii_uppercase
SEED = 42
SOURCE = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

def read_split(archive, split):
    def read_idx(kind):
        suffix = "images-idx3-ubyte" if kind == "images" else "labels-idx1-ubyte"
        name = f"emnist-letters-{split}-{suffix}.gz"
        entries = [entry for entry in archive.namelist() if Path(entry).name == name]
        if len(entries) != 1:
            raise ValueError(f"Expected one {name} in the NIST archive.")
        raw = gzip.decompress(archive.read(entries[0]))
        magic, count = struct.unpack_from(">II", raw)
        if kind == "labels":
            if magic != 2049 or len(raw) != 8 + count:
                raise ValueError(f"Invalid EMNIST label file: {name}")
            labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
            if labels.min() != 1 or labels.max() != 26:
                raise ValueError("Expected EMNIST Letters labels 1–26.")
            return torch.from_numpy(labels.astype(np.int64) - 1)
        rows, cols = struct.unpack_from(">II", raw, 8)
        if magic != 2051 or (rows, cols) != (28, 28) or len(raw) != 16 + count * rows * cols:
            raise ValueError(f"Invalid EMNIST image file: {name}")
        pixels = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28)
        # The IDX pixel layout for EMNIST is transposed relative to upright images.
        upright = pixels.transpose(0, 2, 1).copy()
        return torch.from_numpy(upright).unsqueeze(1)

    return TensorDataset(read_idx("images"), read_idx("labels"))

def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
        nn.MaxPool2d(2), nn.Conv2d(24, 48, 3, padding=1),
        nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(48 * 7 * 7, 128), nn.ReLU(), nn.Dropout(.25),
        nn.Linear(128, len(LETTERS)),
    )

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    expected, predicted = [], []
    for images, labels in loader:
        outputs = model(images.to(device, dtype=torch.float32).div(255))
        expected.append(labels.numpy())
        predicted.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(expected), np.concatenate(predicted)

def plot_results(test, truth, predictions, out):
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        truth, predictions, labels=list(range(26)), display_labels=list(LETTERS),
        include_values=False, cmap="Blues", ax=ax)
    ax.set_title("EMNIST Letters · held-out test set")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for row, match in enumerate([True, False]):
        candidates = np.flatnonzero((truth == predictions) == match)
        picked = rng.choice(candidates, min(len(candidates), 6), replace=False)
        for ax in axes[row]:
            ax.axis("off")
        for ax, index in zip(axes[row], picked):
            ax.imshow(test[int(index)][0][0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"True {LETTERS[truth[index]]} / Pred {LETTERS[predictions[index]]}",
                         fontsize=9, color="#166534" if match else "#b91c1c")
    fig.tight_layout()
    fig.savefig(out / "test_examples.png", dpi=170)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/emnist-gzip.zip"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("Epochs and batch size must be positive.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.data.with_suffix(".part")
        urlretrieve(SOURCE, temporary)
        if not zipfile.is_zipfile(temporary):
            temporary.unlink()
            raise ValueError("The downloaded file is not an EMNIST ZIP archive.")
        temporary.replace(args.data)
    with zipfile.ZipFile(args.data) as archive:
        whole_train = read_split(archive, "train")
        test = read_split(archive, "test")
    if (len(whole_train), len(test)) != (124800, 20800):
        raise ValueError("Expected the official EMNIST Letters split sizes.")
    validation_size = len(whole_train) // 10
    train, validation = random_split(
        whole_train, [len(whole_train) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "letters_cnn.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(device, dtype=torch.float32).div(255))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        expected, predicted = evaluate(model, val_loader, device)
        validation_accuracy = float((expected == predicted).mean())
        row = dict(epoch=epoch, training_loss=total_loss / len(train),
                   validation_accuracy=validation_accuracy)
        history.append(row)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        print(json.dumps(row), flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    truth, predictions = evaluate(model, test_loader, device)
    report = classification_report(truth, predictions, labels=list(range(26)),
                                   target_names=list(LETTERS), output_dict=True,
                                   zero_division=0)
    summary = dict(dataset="EMNIST Letters", device=str(device), seed=SEED,
                   best_epoch=best_epoch, train_count=len(train),
                   validation_count=len(validation), test_count=len(test),
                   test_accuracy=report["accuracy"], macro_f1=report["macro avg"]["f1-score"],
                   history=history, per_class_metrics=report)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_results(test, truth, predictions, args.out)
    print(f"Test accuracy: {summary['test_accuracy']:.4f} | Macro F1: {summary['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
