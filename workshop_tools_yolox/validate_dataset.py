"""Validate the Workshop Tools dataset before starting a costly training run."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePath
import statistics
import sys

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class SplitStats:
    images: int
    boxes: int
    negative_images: int
    boxes_per_class: dict[str, int]
    bbox_area_ratio_min: float
    bbox_area_ratio_median: float
    bbox_area_ratio_max: float
    boxes_touching_edge: int
    suspicious_tiny_boxes: int
    full_frame_boxes: int


class ValidationError(ValueError):
    pass


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Tidak dapat membaca JSON {path}: {exc}") from exc


def _unique(items, key, description: str, split: str) -> dict:
    if not isinstance(items, list):
        raise ValidationError(f"{description} di {split} harus berupa list")
    output = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValidationError(f"Item {description} di {split} harus berupa object")
        value = item.get(key)
        if value in output:
            raise ValidationError(f"{description} duplikat di {split}: {value!r}")
        output[value] = item
    return output


def _safe_file_name(value, split: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"file_name kosong/tidak valid di {split}")
    path = PurePath(value)
    if path.name != value or value in {".", ".."} or "\\" in value:
        raise ValidationError(f"file_name harus berupa basename aman di {split}: {value!r}")
    return value


def _number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{description} harus berupa angka")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{description} harus finite")
    return result


def validate_dataset(
    root: Path, train_annotation: str = "instances_train.json"
) -> tuple[dict[str, SplitStats], list[str]]:
    root = root.expanduser().resolve()
    labels_path = root / "labels.json"
    labels = _read_json(labels_path)
    if not isinstance(labels, list) or not labels:
        raise ValidationError(f"labels.json harus berupa list non-kosong: {labels_path}")
    if not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValidationError("Semua label harus berupa string non-kosong")
    if len(labels) != len(set(labels)):
        raise ValidationError("Nama label harus unik")

    warnings: list[str] = []
    stats: dict[str, SplitStats] = {}
    expected_categories = None
    hashes: dict[str, tuple[str, str]] = {}
    original_names: dict[str, tuple[str, str]] = {}

    for split in SPLITS:
        split_dir = root / split
        annotation_name = train_annotation if split == "train" else f"instances_{split}.json"
        if Path(annotation_name).name != annotation_name:
            raise ValidationError("Nama anotasi train harus berupa basename")
        annotation_path = root / "annotations" / annotation_name
        if not split_dir.is_dir():
            raise ValidationError(f"Folder split tidak ditemukan: {split_dir}")
        document = _read_json(annotation_path)
        if not isinstance(document, dict):
            raise ValidationError(f"Root JSON harus object: {annotation_path}")

        categories = _unique(document.get("categories", []), "id", "Category ID", split)
        ordered_categories = [categories[key] for key in sorted(categories)]
        category_names = [category.get("name") for category in ordered_categories]
        category_ids = [category.get("id") for category in ordered_categories]
        if category_names != labels:
            raise ValidationError(
                f"Urutan categories di {split} tidak sama dengan labels.json: {category_names}"
            )
        if category_ids != list(range(1, len(labels) + 1)):
            raise ValidationError(
                f"Category ID di {split} harus rapat dari 1 agar mapping model eksplisit: "
                f"{category_ids}"
            )
        category_signature = tuple(zip(category_ids, category_names))
        if expected_categories is None:
            expected_categories = category_signature
        elif category_signature != expected_categories:
            raise ValidationError(f"Categories tidak konsisten di split {split}")

        images = _unique(document.get("images", []), "id", "Image ID", split)
        annotations = _unique(document.get("annotations", []), "id", "Annotation ID", split)
        if not images:
            raise ValidationError(f"Split {split} tidak memiliki gambar")

        image_names: set[str] = set()
        image_sizes: dict[object, tuple[int, int]] = {}
        for image_id, image_record in images.items():
            file_name = _safe_file_name(image_record.get("file_name"), split)
            if file_name in image_names:
                raise ValidationError(f"file_name duplikat di {split}: {file_name}")
            image_names.add(file_name)

            width = image_record.get("width")
            height = image_record.get("height")
            if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
                raise ValidationError(f"Dimensi JSON tidak valid: {split}/{file_name}")

            image_path = split_dir / file_name
            if not image_path.is_file():
                raise ValidationError(f"Gambar yang dirujuk tidak ditemukan: {image_path}")
            try:
                with Image.open(image_path) as picture:
                    picture.load()
                    actual_size = picture.size
            except (OSError, ValueError) as exc:
                raise ValidationError(f"Gambar rusak/tidak terbaca {image_path}: {exc}") from exc
            if actual_size != (width, height):
                raise ValidationError(
                    f"Dimensi berbeda untuk {image_path}: JSON={(width, height)}, file={actual_size}"
                )
            image_sizes[image_id] = (width, height)

            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            previous = hashes.get(digest)
            if previous and previous[0] != split:
                raise ValidationError(
                    f"Gambar byte-identik lintas split: {previous[0]}/{previous[1]} dan "
                    f"{split}/{file_name}"
                )
            hashes[digest] = (split, file_name)

            original = image_record.get("extra", {}).get("name")
            if isinstance(original, str) and original:
                previous_original = original_names.get(original)
                if previous_original and previous_original[0] != split:
                    raise ValidationError(
                        f"Nama sumber lintas split: {previous_original[0]}/{previous_original[1]} "
                        f"dan {split}/{file_name} (source={original})"
                    )
                original_names[original] = (split, file_name)

        class_counts: Counter[int] = Counter()
        labeled_images = set()
        area_ratios: list[float] = []
        edge_boxes = 0
        tiny_boxes = 0
        full_frame_boxes = 0
        for annotation_id, annotation in annotations.items():
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            if image_id not in images:
                raise ValidationError(
                    f"Anotasi {annotation_id} di {split} merujuk image_id yang tidak ada: {image_id}"
                )
            if category_id not in categories:
                raise ValidationError(
                    f"Anotasi {annotation_id} di {split} merujuk category_id yang tidak ada: "
                    f"{category_id}"
                )
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValidationError(f"BBox anotasi {annotation_id} di {split} harus [x,y,w,h]")
            x, y, width, height = [
                _number(value, f"bbox {split}/{annotation_id}") for value in bbox
            ]
            image_width, image_height = image_sizes[image_id]
            tolerance = 1e-3
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValidationError(f"BBox non-positif di {split}/{annotation_id}: {bbox}")
            if x + width > image_width + tolerance or y + height > image_height + tolerance:
                raise ValidationError(f"BBox keluar batas gambar di {split}/{annotation_id}: {bbox}")
            area = _number(annotation.get("area", width * height), f"area {split}/{annotation_id}")
            if area <= 0:
                raise ValidationError(f"Area harus positif di {split}/{annotation_id}")
            if annotation.get("iscrowd", 0) not in (0, False):
                warnings.append(f"iscrowd bukan 0 di {split}/{annotation_id}; loader YOLOX mengabaikannya")

            labeled_images.add(image_id)
            class_counts[category_id] += 1
            area_ratios.append((width * height) / (image_width * image_height))
            if min(width, height) < 3 or width * height < 16:
                tiny_boxes += 1
            if width * height >= 0.99 * image_width * image_height:
                full_frame_boxes += 1
            if (
                x <= tolerance
                or y <= tolerance
                or x + width >= image_width - tolerance
                or y + height >= image_height - tolerance
            ):
                edge_boxes += 1

        disk_images = {
            path.name
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        extra_files = sorted(disk_images - image_names)
        if extra_files:
            warnings.append(
                f"{split}: {len(extra_files)} file gambar tidak dirujuk JSON "
                f"(contoh: {extra_files[0]})"
            )
        missing_classes = [
            categories[category_id]["name"]
            for category_id in category_ids
            if class_counts[category_id] == 0
        ]
        if missing_classes:
            warnings.append(f"{split}: kelas tanpa bbox: {', '.join(missing_classes)}")

        if area_ratios:
            minimum = min(area_ratios)
            median = statistics.median(area_ratios)
            maximum = max(area_ratios)
        else:
            minimum = median = maximum = 0.0
            warnings.append(f"{split}: tidak memiliki bounding box")
        stats[split] = SplitStats(
            images=len(images),
            boxes=len(annotations),
            negative_images=len(set(images) - labeled_images),
            boxes_per_class={
                categories[category_id]["name"]: class_counts[category_id]
                for category_id in category_ids
            },
            bbox_area_ratio_min=round(minimum, 6),
            bbox_area_ratio_median=round(median, 6),
            bbox_area_ratio_max=round(maximum, 6),
            boxes_touching_edge=edge_boxes,
            suspicious_tiny_boxes=tiny_boxes,
            full_frame_boxes=full_frame_boxes,
        )

        if tiny_boxes:
            warnings.append(
                f"{split}: {tiny_boxes} bbox sangat kecil (<3 px pada salah satu sisi atau "
                "<16 px^2); periksa sebagai kemungkinan artefak label."
            )
        if full_frame_boxes:
            warnings.append(
                f"{split}: {full_frame_boxes} bbox mencakup >=99% gambar; periksa anotasinya."
            )

    if all(item.negative_images == 0 for item in stats.values()):
        warnings.append(
            "Tidak ada gambar negatif; false-positive pada latar tanpa tool belum terwakili."
        )
    if stats["test"].images < 30:
        warnings.append(
            f"Test hanya {stats['test'].images} gambar; metriknya ber-variance tinggi."
        )
    return stats, warnings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "workshop_tools",
        help="Root dataset (default: workshop_tools di sebelah script)",
    )
    parser.add_argument("--json", action="store_true", help="Cetak hasil sebagai JSON")
    parser.add_argument(
        "--train-ann",
        default="instances_train.json",
        help="Nama JSON train di folder annotations (default: instances_train.json)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        stats, warnings = validate_dataset(args.dataset, train_annotation=args.train_ann)
    except ValidationError as exc:
        print(f"GAGAL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset.expanduser().resolve()),
                    "valid": True,
                    "splits": {name: asdict(value) for name, value in stats.items()},
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("VALID: struktur COCO, gambar, kategori, dan bounding box konsisten.")
        for split, value in stats.items():
            counts = ", ".join(f"{name}={count}" for name, count in value.boxes_per_class.items())
            print(
                f"- {split}: {value.images} gambar, {value.boxes} bbox, "
                f"negatif={value.negative_images}; {counts}"
            )
        for warning in warnings:
            print(f"PERINGATAN: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
