"""Regenerate evaluation artifacts. Install Pillow, then run python reproduce.py."""
import argparse
import csv
import io
import json
import math
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
INK = "#16253c"
MUTED = "#526782"


def read_rows(cache, name, suffix, limit):
    if cache:
        content = (cache / f"{name}.csv").read_text()
    else:
        with urlopen(SOURCE + "optdigits." + suffix, timeout=60) as response:
            content = response.read().decode("ascii")
    rows = [list(map(int, row)) for row in islice(csv.reader(io.StringIO(content)), limit)]
    assert len(rows) == limit and all(len(row) == 65 for row in rows)
    return rows


def predict(row, training):
    neighbors = sorted(range(len(training)), key=lambda i: sum(
        (a - b) ** 2 for a, b in zip(row[:-1], training[i][:-1])))
    votes = Counter(training[i][-1] for i in neighbors[:3])
    return max((training[i][-1] for i in neighbors[:3]), key=votes.get)


def write_csv(name, header, rows):
    with (ROOT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def label(draw, xy, value, size, color=INK, anchor="la"):
    draw.text(xy, str(value), font=ImageFont.load_default(size=size), fill=color, anchor=anchor)


def gallery(name, title, indices, test, predictions):
    cols = 5
    canvas = Image.new("RGB", (830, 88 + math.ceil(len(indices) / cols) * 155), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw, (24, 14), title, 30)
    label(draw, (24, 54), "UCI Optical Digits  /  held-out test images", 16, MUTED)
    for position, i in enumerate(indices):
        x, y = 24 + position % cols * 158, 88 + position // cols * 155
        digit = Image.new("L", (8, 8))
        digit.putdata([round(255 * p / 16) for p in test[i][:-1]])
        canvas.paste(digit.resize((88, 88), Image.Resampling.NEAREST), (x, y))
        label(draw, (x, y + 94), f"True {test[i][-1]}   Pred {predictions[i]}", 17)
        label(draw, (x, y + 118), f"Test row {i+1}", 14, MUTED)
    canvas.save(ROOT / name)


def draw_matrix(matrix):
    canvas = Image.new("RGB", (680, 660), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw, (24, 16), "Confusion matrix", 32)
    label(draw, (24, 58), "Rows show true digits. Columns show predicted digits.", 18, MUTED)
    peak = max(map(max, matrix))
    for digit in range(10):
        label(draw, (115 + digit * 54, 105), digit, 17, anchor="mm")
        label(draw, (60, 145 + digit * 50), digit, 17, anchor="mm")
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            x, y = 88 + j * 54, 120 + i * 50
            mix = (count / peak) ** 0.6
            fill = tuple(round(a + (b-a)*mix) for a,b in zip((240,245,249),(16,84,130)))
            draw.rectangle((x, y, x+52, y+48), fill=fill)
            label(draw, (x+27,y+25), count, 18, "white" if mix > 0.6 else INK, "mm")
    canvas.save(ROOT / "confusion_matrix.png")


def main(cache):
    train = read_rows(cache, "train", "tra", 200)
    test = read_rows(cache, "test", "tes", 100)
    predictions = [predict(row, train) for row in test]
    matrix = [[0] * 10 for _ in range(10)]
    for row, predicted in zip(test, predictions):
        matrix[row[-1]][predicted] += 1
    correct = sum(matrix[i][i] for i in range(10))
    per_class = []
    for i in range(10):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0
        recall = tp / support if support else 0
        f1 = 2 * tp / (support + predicted_count) if support + predicted_count else 0
        per_class.append((i, precision, recall, f1, support))
    summary = dict(model="3-nearest neighbors", train_images=len(train), test_images=len(test),
                   correct=correct, errors=len(test)-correct, accuracy=correct/len(test),
                   macro_f1=sum(row[3] for row in per_class)/10)
    (ROOT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv("classification_report.csv", ["digit","precision","recall","f1","support"], per_class)
    write_csv("predictions.csv", ["test_row","actual","predicted","correct"],
              [(i+1, row[-1], p, int(row[-1] == p)) for i,(row,p) in enumerate(zip(test,predictions))])
    write_csv("confusion_matrix.csv", ["actual/predicted", *range(10)],
              [(i,*row) for i,row in enumerate(matrix)])
    draw_matrix(matrix)
    indices = [next(i for i,row in enumerate(test) if row[-1] == digit) for digit in range(10)]
    gallery("sample_predictions.png", "One example per digit", indices, test, predictions)
    errors = [i for i,row in enumerate(test) if row[-1] != predictions[i]]
    if errors:
        gallery("misclassified_digits.png", "Misclassified digits", errors, test, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Optional directory containing train.csv and test.csv")
    main(parser.parse_args().cache)
