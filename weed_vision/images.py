"""Bounded decoding and format checks before any model inference."""

from io import BytesIO
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import APIError

MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


def decode_image(data: bytes, content_type: str | None, settings: Settings) -> Image.Image:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in MIME_FORMATS:
        raise APIError(415, "unsupported_image_type", "Use image/jpeg, image/png, or image/webp.")
    if not data:
        raise APIError(400, "empty_image", "The uploaded image is empty.")
    if len(data) > settings.max_upload_bytes:
        raise APIError(413, "image_too_large", "The image exceeds the configured byte limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format != MIME_FORMATS[mime]:
                    raise APIError(415, "image_type_mismatch", "Image contents do not match the MIME type.")
                if source.width * source.height > settings.max_image_pixels:
                    raise APIError(413, "image_too_large", "The image exceeds the configured pixel limit.")
                if getattr(source, "n_frames", 1) != 1:
                    raise APIError(400, "animated_image", "Upload a single-frame image.")
                source.verify()
            # verify() consumes the decoder. Open again, apply orientation, and detach pixels.
            with Image.open(BytesIO(data)) as source:
                oriented = ImageOps.exif_transpose(source)
                oriented.load()
                return oriented.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise APIError(413, "image_too_large", "The decoded image is too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise APIError(400, "invalid_image", "The upload is not a valid, complete image.") from exc
