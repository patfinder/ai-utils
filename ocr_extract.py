#!/usr/bin/env python3
"""Extract text from images in a folder using the Google Cloud Vision API."""

import argparse
import sys
import time
from pathlib import Path

from google.cloud import vision

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
    for i, image_path in enumerate(images, start=1):
        rel_path = image_path.relative_to(args.input_dir)
        out_path = (output_dir / rel_path).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(images)}] Skipping {rel_path} (output exists)")
            continue

        print(f"[{i}/{len(images)}] Processing {rel_path}...")
        try:
            text = extract_text(client, image_path)
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            continue

        out_path.write_text(text, encoding="utf-8")

        if args.combined:
            combined_lines.append(f"----- {rel_path} -----\n{text}\n")

        if args.delay:
            time.sleep(args.delay)

    if args.combined and combined_lines:
        args.combined.write_text("\n".join(combined_lines), encoding="utf-8")
        print(f"Combined output written to {args.combined}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
