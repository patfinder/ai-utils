#!/usr/bin/env python3
"""Extract text from images in a folder using the Google Cloud Vision API."""

import argparse
import io
import math
import re
import sys
import time
from pathlib import Path

from google.cloud import vision
from PIL import Image as PILImage

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_text(client: vision.ImageAnnotatorClient, image_path: Path) -> str:
    content = image_path.read_bytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)
    annotations = response.text_annotations
    return annotations[0].description if annotations else ""


def parse_grid(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError("Grid must look like COLSxROWS, e.g. 3x2")
    cols, rows = int(match.group(1)), int(match.group(2))
    if cols < 1 or rows < 1:
        raise argparse.ArgumentTypeError("Grid columns/rows must be at least 1")
    return cols, rows


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_composite(
    image_paths: list[Path], cols: int, gap: int, max_dim: int
) -> tuple[bytes, list[tuple[Path, int, int, int, int]]]:
    """Tile images left-to-right, top-to-bottom into one image at native resolution.

    Returns the composite PNG bytes and, for each source image, the
    (path, x_offset, y_offset, width, height) rectangle it occupies in the
    composite, so OCR results can be mapped back to their source image.
    """
    opened = [PILImage.open(p).convert("RGB") for p in image_paths]
    rows = math.ceil(len(opened) / cols)

    col_widths = [0] * cols
    row_heights = [0] * rows
    for idx, img in enumerate(opened):
        r, c = divmod(idx, cols)
        col_widths[c] = max(col_widths[c], img.width)
        row_heights[r] = max(row_heights[r], img.height)

    canvas_w = sum(col_widths) + gap * (cols - 1)
    canvas_h = sum(row_heights) + gap * (rows - 1)
    if canvas_w > max_dim or canvas_h > max_dim:
        raise ValueError(
            f"Composite size {canvas_w}x{canvas_h} exceeds --tile-max-dim {max_dim}; "
            "use a smaller --tile-grid or raise --tile-max-dim"
        )

    canvas = PILImage.new("RGB", (canvas_w, canvas_h), color="white")
    placements = []
    col_x = [sum(col_widths[:c]) + gap * c for c in range(cols)]
    row_y = [sum(row_heights[:r]) + gap * r for r in range(rows)]
    for idx, img in enumerate(opened):
        r, c = divmod(idx, cols)
        x, y = col_x[c], row_y[r]
        canvas.paste(img, (x, y))
        placements.append((image_paths[idx], x, y, img.width, img.height))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue(), placements


def paragraph_text(paragraph) -> str:
    return " ".join(
        "".join(symbol.text for symbol in word.symbols) for word in paragraph.words
    )


def block_bounds(block) -> tuple[float, float, float, float]:
    xs = [v.x for v in block.bounding_box.vertices]
    ys = [v.y for v in block.bounding_box.vertices]
    return min(xs), min(ys), sum(xs) / len(xs), sum(ys) / len(ys)


def assign_block_to_placement(cx, cy, placements):
    for path, x, y, w, h in placements:
        if x <= cx < x + w and y <= cy < y + h:
            return path
    # Fallback: nearest placement center (e.g. text detected in the gutter)
    def dist(p):
        _, x, y, w, h = p
        return (cx - (x + w / 2)) ** 2 + (cy - (y + h / 2)) ** 2

    return min(placements, key=dist)[0]


def extract_tiled_text(
    client: vision.ImageAnnotatorClient,
    composite_bytes: bytes,
    placements: list[tuple[Path, int, int, int, int]],
) -> dict[Path, str]:
    image = vision.Image(content=composite_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    results: dict[Path, list[tuple[float, float, str]]] = {p: [] for p, *_ in placements}
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            text = "\n".join(paragraph_text(p) for p in block.paragraphs)
            if not text.strip():
                continue
            min_x, min_y, cx, cy = block_bounds(block)
            path = assign_block_to_placement(cx, cy, placements)
            results[path].append((min_y, min_x, text))

    return {
        path: "\n\n".join(text for _, _, text in sorted(blocks))
        for path, blocks in results.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from images in a folder using Google Cloud Vision OCR."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing images")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Folder to write .txt files to (default: same as input folder)",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recurse into subfolders",
    )
    parser.add_argument(
        "--combined", type=Path, default=None,
        help="Also write all results to a single combined text file at this path",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .txt output files (default: skip)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to sleep between API calls (helps avoid rate limits)",
    )
    parser.add_argument(
        "--tile-grid", type=parse_grid, default=None, metavar="COLSxROWS",
        help=(
            "Combine multiple images into one composite image per API call to cut "
            "billed units, e.g. 3x2 tiles up to 6 images per request. Images are "
            "placed at native resolution; OCR results are mapped back to each "
            "source image automatically."
        ),
    )
    parser.add_argument(
        "--tile-gap", type=int, default=20,
        help="Pixels of white space between tiled images (default: 20)",
    )
    parser.add_argument(
        "--tile-max-dim", type=int, default=15000,
        help="Max composite width/height in pixels; aborts the batch if exceeded (default: 15000)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(args.input_dir, args.recursive)
    if not images:
        print(f"No images found in {args.input_dir}")
        return 0

    client = vision.ImageAnnotatorClient()
    combined_lines = []

    def out_path_for(image_path: Path) -> Path:
        rel_path = image_path.relative_to(args.input_dir)
        out_path = (output_dir / rel_path).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    def write_result(image_path: Path, out_path: Path, text: str) -> None:
        out_path.write_text(text, encoding="utf-8")
        if args.combined:
            rel_path = image_path.relative_to(args.input_dir)
            combined_lines.append(f"----- {rel_path} -----\n{text}\n")

    if args.tile_grid:
        cols, rows = args.tile_grid
        batch_size = cols * rows

        pending = []
        for image_path in images:
            out_path = out_path_for(image_path)
            if out_path.exists() and not args.overwrite:
                print(f"Skipping {image_path.relative_to(args.input_dir)} (output exists)")
                continue
            pending.append((image_path, out_path))

        batches = list(chunked(pending, batch_size))
        for i, batch in enumerate(batches, start=1):
            batch_paths = [p for p, _ in batch]
            names = ", ".join(p.name for p in batch_paths)
            print(f"[batch {i}/{len(batches)}] Tiling {len(batch_paths)} image(s): {names}")
            try:
                composite_bytes, placements = build_composite(
                    batch_paths, cols, args.tile_gap, args.tile_max_dim
                )
                texts = extract_tiled_text(client, composite_bytes, placements)
            except Exception as exc:
                print(f"  Error: {exc}", file=sys.stderr)
                continue

            for image_path, out_path in batch:
                write_result(image_path, out_path, texts.get(image_path, ""))

            if args.delay:
                time.sleep(args.delay)
    else:
        for i, image_path in enumerate(images, start=1):
            out_path = out_path_for(image_path)
            rel_path = image_path.relative_to(args.input_dir)

            if out_path.exists() and not args.overwrite:
                print(f"[{i}/{len(images)}] Skipping {rel_path} (output exists)")
                continue

            print(f"[{i}/{len(images)}] Processing {rel_path}...")
            try:
                text = extract_text(client, image_path)
            except Exception as exc:
                print(f"  Error: {exc}", file=sys.stderr)
                continue

            write_result(image_path, out_path, text)

            if args.delay:
                time.sleep(args.delay)

    if args.combined and combined_lines:
        args.combined.write_text("\n".join(combined_lines), encoding="utf-8")
        print(f"Combined output written to {args.combined}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
