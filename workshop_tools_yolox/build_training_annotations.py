"""Build a YOLOX training annotation file without degenerate box artefacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "workshop_tools" / "annotations" / "instances_train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "workshop_tools" / "annotations" / "instances_train.yolox.json",
    )
    parser.add_argument(
        "--min-side",
        type=float,
        default=3.0,
        help="Buang bbox dengan sisi lebih kecil dari nilai ini (default: 3 px)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=16.0,
        help="Buang bbox dengan area lebih kecil dari nilai ini (default: 16 px^2)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.min_side < 0 or args.min_area < 0:
        raise SystemExit("--min-side dan --min-area tidak boleh negatif")
    if source == output:
        raise SystemExit("Source dan output harus berbeda agar anotasi asli tidak tertimpa")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Gagal membaca {source}: {exc}") from exc

    removed = []
    kept = []
    for annotation in document.get("annotations", []):
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise SystemExit(f"BBox tidak valid pada annotation id={annotation.get('id')}")
        width, height = bbox[2], bbox[3]
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in (width, height)
        ):
            raise SystemExit(f"BBox tidak numerik pada annotation id={annotation.get('id')}")
        if min(width, height) < args.min_side or width * height < args.min_area:
            removed.append(annotation)
        else:
            kept.append(annotation)

    if not removed:
        raise SystemExit(
            "Tidak ada bbox yang memenuhi kriteria pembersihan; "
            "periksa threshold sebelum mengganti artefak yang ada."
        )
    document["annotations"] = kept
    document["workshop_cleaning"] = {
        "source": source.name,
        "min_side_px": args.min_side,
        "min_area_px2": args.min_area,
        "removed_annotation_ids": [annotation.get("id") for annotation in removed],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    removed_ids = ", ".join(str(annotation.get("id")) for annotation in removed)
    print(f"Sumber : {source}")
    print(f"Output : {output}")
    print(f"Bbox   : {len(kept)} dipertahankan, {len(removed)} dibuang (ID: {removed_ids})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
