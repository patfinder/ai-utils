# ocr_extract

Extract text from images in a folder using Google Cloud Vision's OCR (`text_detection`).

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Enable the **Cloud Vision API** in a GCP project (Console → APIs & Services → search "Cloud Vision API" → Enable).

3. Create a service account with the "Cloud Vision AI Service Agent" (or a basic Viewer/Editor) role, download its JSON key, then point the client at it:
   ```
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

## Usage

```
./ocr_extract.py /path/to/images
```

Writes a `.txt` file next to each image containing its detected text.

### Options

| Flag | Description |
|---|---|
| `-o, --output-dir DIR` | Write `.txt` files here instead of next to the source images (mirrors subfolder structure) |
| `-r, --recursive` | Recurse into subfolders |
| `--combined FILE` | Also write all results, concatenated, to a single file |
| `--overwrite` | Overwrite `.txt` files that already exist (default: skip images already processed) |
| `--delay SECONDS` | Sleep between API calls, to stay under rate limits |
| `--tile-grid COLSxROWS` | Combine multiple images into one composite per API call (see [Tiling mode](#tiling-mode-cost-saving)) |
| `--tile-gap PIXELS` | White space between tiled images (default: `20`) |
| `--tile-max-dim PIXELS` | Max composite width/height; batch aborts if exceeded (default: `15000`) |

### Examples

```
# Basic: process all images in a folder
./ocr_extract.py ./scans

# Recurse into subfolders, write output elsewhere, also produce one combined file
./ocr_extract.py ./scans -o ./scans_text -r --combined ./scans_text/all.txt

# Re-run and overwrite previous results, throttling requests
./ocr_extract.py ./scans --overwrite --delay 0.2
```

## Tiling mode (cost saving)

Cloud Vision bills per **image sent per request** (1 unit), not per request, regardless of that image's resolution. `--tile-grid COLSxROWS` exploits this: it tiles up to `COLS * ROWS` source images into one composite image at native resolution, sends it as a single request, and maps the OCR results back to each original image automatically — so N images can cost as little as `ceil(N / (COLS*ROWS))` units instead of N.

```
# Tile up to 6 images per request (3 columns x 2 rows)
./ocr_extract.py ./receipts --tile-grid 3x2
```

Trade-offs to be aware of:

- **No downscaling** — images are placed at full native size, so accuracy isn't sacrificed, but a big grid of large images can exceed Vision's request size limits. If a batch's composite exceeds `--tile-max-dim`, that batch is skipped with an error — use a smaller grid or raise the limit.
- **Best for small/uniform images** — works well for things like receipts, ID crops, or screenshots. Very large or high-res source images will hit size limits with fewer tiles per grid cell.
- **Text mapping is automatic but heuristic** — each detected text block is assigned to whichever source image's region it falls inside (or the nearest one, for stray text landing in the gutter). This is reliable for well-separated tiles but isn't pixel-perfect for adversarial layouts.
- **Uses `DOCUMENT_TEXT_DETECTION`** internally (needed to get block-level bounding boxes for splitting), rather than `TEXT_DETECTION` used in normal mode — output quality is comparable or better for dense text.
- All other flags (`-o`, `-r`, `--combined`, `--overwrite`, `--delay`) work the same in tiling mode.

## Supported formats

`.png` `.jpg` `.jpeg` `.gif` `.bmp` `.tiff` `.tif` `.webp`

## Notes

- Cloud Vision charges per image after a free monthly quota — check current pricing before running on large batches.
- Each image is sent in full to the API (no local resizing/preprocessing).
