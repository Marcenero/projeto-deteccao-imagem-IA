"""Decodificação e padronização para etapas posteriores."""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def inspect_image(data: bytes) -> tuple[int, int, str]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        return image.width, image.height, image.format


def preprocess_image(
    source: bytes | str | Path, size: tuple[int, int] = (256, 256)
) -> Image.Image:
    """Retorna pixels RGB com orientação EXIF e recorte central proporcional."""
    stream = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(stream) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(rgb, size, method=Image.Resampling.LANCZOS)
