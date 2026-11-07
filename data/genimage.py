"""Dataset utilities for the Tiny GenImage benchmark (method-agnostic).

Layout under ``data/downloads/tiny-genimage/`` (populated by scripts/download.sh):

* ``data/train-*.parquet`` - the published training split (28,000 images) with
  its ``label`` (real/fake) and ``generator`` (source) columns.
* ``test/test-*.parquet`` - the published validation split (7,000 images) used
  as the held-out test set. These shards carry only ``sample_id`` and the
  ``image`` (a lossless PNG of the decoded RGB pixels, so the container of the
  published file is not kept); their labels are not part of the workspace.

``data/splits/dev.csv`` lists a fixed, label-stratified 20% of the training
split (5,600 images) used as the visible development slice scored by
``scripts/evaluate_train.sh``. Its labels are ordinary training labels.

Every image is identified by a content-derived ``sample_id``
(``tr-``/``te-`` + the first 20 hex digits of the SHA-256 of its encoded
bytes). Prediction CSVs must use these ids.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data/downloads/tiny-genimage"
DEV_SPLIT_CSV = REPO_ROOT / "data/splits/dev.csv"

GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",  # declared by the dataset card; no image carries this label
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}
LABEL_SPACE = tuple(sorted(GENERATOR_NAMES))
BINARY_NAMES = {0: "real", 1: "fake"}

# Prediction CSV contract shared by every method (see briefing/evaluation.md).
PREDICTION_FIELDS = ["method", "split", "sample_id", "prediction"]


@dataclass(frozen=True)
class TrainRecord:
    sample_id: str
    image: Image.Image | None
    generator_label: int
    binary_label: int


@dataclass(frozen=True)
class TestRecord:
    sample_id: str
    image: Image.Image | None


def _image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"Unsupported image parquet value: {type(value)!r}")


def decode_image(raw: bytes) -> Image.Image:
    return Image.open(BytesIO(raw)).convert("RGB")


def content_id(prefix: str, raw: bytes) -> str:
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20]}"


def _shards(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No parquet shards matching '{pattern}' under {root}. "
            "Run scripts/download.sh first."
        )
    return paths


def train_shards(root: Path = DEFAULT_ROOT) -> list[Path]:
    return _shards(root, "data/train-*.parquet")


def test_shards(root: Path = DEFAULT_ROOT) -> list[Path]:
    return _shards(root, "test/test-*.parquet")


def iter_train(
    root: Path = DEFAULT_ROOT, decode: bool = True, batch_size: int = 64
) -> Iterator[TrainRecord]:
    """Yield every training image with its content id and labels.

    Byte-identical duplicates get a deterministic ``-dupN`` suffix in shard
    order, so ids are unique.
    """
    seen: dict[str, int] = {}
    for shard in train_shards(root):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["image", "label", "generator"]
        ):
            for row in batch.to_pylist():
                raw = _image_bytes(row["image"])
                sid = content_id("tr", raw)
                count = seen.get(sid, 0)
                seen[sid] = count + 1
                if count:
                    sid = f"{sid}-dup{count}"
                yield TrainRecord(
                    sample_id=sid,
                    image=decode_image(raw) if decode else None,
                    generator_label=int(row["generator"]),
                    binary_label=int(row["label"]),
                )


def iter_test(
    root: Path = DEFAULT_ROOT, decode: bool = True, batch_size: int = 64
) -> Iterator[TestRecord]:
    """Yield every held-out test image (no labels are available)."""
    for shard in test_shards(root):
        parquet = pq.ParquetFile(shard)
        columns = ["sample_id", "image"] if decode else ["sample_id"]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield TestRecord(
                    sample_id=str(row["sample_id"]),
                    image=decode_image(_image_bytes(row["image"])) if decode else None,
                )


def iter_train_labels(root: Path = DEFAULT_ROOT, batch_size: int = 4096) -> Iterator[int]:
    """Yield training generator labels without reading image payloads."""
    for shard in train_shards(root):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["generator"]):
            for value in batch.column(0).to_pylist():
                yield int(value)


def load_dev_split(path: Path = DEV_SPLIT_CSV) -> dict[str, int]:
    """Return {sample_id: generator_label} for the visible dev slice."""
    with Path(path).open(newline="") as f:
        return {row["sample_id"]: int(row["generator"]) for row in csv.DictReader(f)}
