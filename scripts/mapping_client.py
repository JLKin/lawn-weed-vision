"""Upload an image and a context JSON file without converting nanosecond strings."""

import argparse
import json
from pathlib import Path
import sys
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    mimes = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mimes.get(args.image.suffix.lower())
    if mime is None: parser.error("Use JPEG, PNG or WebP")
    try:
        context = args.context.read_text(encoding="utf-8-sig")
        json.loads(context)
        with args.image.open("rb") as image:
            response = httpx.post(args.url.rstrip("/") + "/v1/scene-mapping", timeout=180,
                data={"context": context}, files={"file": (args.image.name, image, mime)})
        print(response.text)
        return 0 if response.status_code == 200 else 1
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
