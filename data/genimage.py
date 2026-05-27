"""Dataset utilities for the Tiny GenImage benchmark.

This module is intentionally method-agnostic: it only locates parquet shards
and yields decoded images plus labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image


GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}

BINARY_NAMES = {0: "real", 1: "fake"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    index: int
    image: Image.Image
    binary_label: int
    generator_label: int


def shard_paths(root: Path, split: str) -> list[Path]:
    paths = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No Tiny GenImage parquet shards found for split '{split}' under {root / 'data'}"
        )
    return paths


def _decode_image(value: object) -> Image.Image:
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw is not None:
            return Image.open(BytesIO(raw)).convert("RGB")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image parquet value: {type(value)!r}")


def iter_records(root: Path, split: str, batch_size: int = 64) -> Iterator[ImageRecord]:
    """Yield Tiny GenImage records from local parquet shards."""
    offset = 0
    for shard in shard_paths(root, split):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["image", "label", "generator"]
        ):
            rows = batch.to_pylist()
            for row in rows:
                yield ImageRecord(
                    split=split,
                    index=offset,
                    image=_decode_image(row["image"]),
                    binary_label=int(row["label"]),
                    generator_label=int(row["generator"]),
                )
                offset += 1
