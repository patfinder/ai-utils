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

### Examples

```
# Basic: process all images in a folder
./ocr_extract.py ./scans

# Recurse into subfolders, write output elsewhere, also produce one combined file
./ocr_extract.py ./scans -o ./scans_text -r --combined ./scans_text/all.txt

# Re-run and overwrite previous results, throttling requests
./ocr_extract.py ./scans --overwrite --delay 0.2
```

## Supported formats

`.png` `.jpg` `.jpeg` `.gif` `.bmp` `.tiff` `.tif` `.webp`

## Notes

- Cloud Vision charges per image after a free monthly quota — check current pricing before running on large batches.
- Each image is sent in full to the API (no local resizing/preprocessing).
