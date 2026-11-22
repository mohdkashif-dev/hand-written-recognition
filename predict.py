"""Classify a tightly framed photo or image of one handwritten letter."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from train import LETTERS, make_model

def prepare(path):
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        gray = Image.alpha_composite(background, rgba).convert("L")
    pixels = np.asarray(gray)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    ink = ImageOps.invert(gray) if np.median(border) > 127 else gray
    ink = ImageOps.autocontrast(ink)
    bounds = ink.point(lambda value: 255 if value > 48 else 0).getbbox()
    if bounds is None:
        raise ValueError("No visible handwriting found. Use a dark letter on plain paper.")
    glyph = ink.crop(bounds)
    glyph.thumbnail((21, 21), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(glyph, ((28 - glyph.width) // 2, (28 - glyph.height) // 2))
    tensor = torch.from_numpy(np.array(canvas)).float().div(255)[None, None]
    return canvas, tensor

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/letters_cnn.pt"))
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    model = make_model()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    canvas, tensor = prepare(args.image)
    with torch.inference_mode():
        values, indices = model(tensor).softmax(dim=1)[0].topk(3)
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"{LETTERS[index]}: {value:.1%} model score")
    if args.preview:
        canvas.resize((280, 280), Image.Resampling.NEAREST).save(args.preview)

if __name__ == "__main__":
    main()
