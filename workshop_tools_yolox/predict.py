"""Run YOLOX inference with the Workshop Tools class names."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="File gambar atau folder gambar")
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=ROOT / "predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence inference")
    parser.add_argument("--nms", type=float, default=0.45, help="IoU threshold NMS")
    parser.add_argument("--size", type=int, default=640, help="Ukuran input; kelipatan 32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp16", action="store_true", help="FP16 (CUDA saja)")
    parser.add_argument("--fuse", action="store_true", help="Fuse Conv+BN untuk inference")
    parser.add_argument("--legacy", action="store_true", help="Preprocessing bobot YOLOX lama")
    parser.add_argument(
        "--no-json", action="store_true", help="Jangan simpan predictions.json"
    )
    return parser


def _images(input_path: Path) -> tuple[list[Path], Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Ekstensi gambar tidak didukung: {input_path}")
        return [input_path], input_path.parent
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"Tidak ada gambar yang didukung di {input_path}")
        return images, input_path
    raise ValueError(f"Input tidak ditemukan: {input_path}")


def _labels() -> list[str]:
    data_dir = Path(
        os.environ.get("WORKSHOP_DATA_DIR", str(ROOT / "workshop_tools"))
    ).expanduser().resolve()
    value = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"labels.json tidak valid di {data_dir}")
    return value


def _device(requested: str, torch):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA diminta tetapi torch.cuda.is_available() bernilai False")
    return torch.device(requested)


def _checkpoint_state(path: Path, torch):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Format checkpoint tidak dikenali: {path}")


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.nms <= 1.0:
        raise SystemExit("--conf dan --nms harus berada dalam rentang 0..1")
    if args.size <= 0 or args.size % 32:
        raise SystemExit("--size harus positif dan merupakan kelipatan 32")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    try:
        import cv2
        import torch
        from yolox.data.data_augment import ValTransform
        from yolox.exp import get_exp
        from yolox.utils import fuse_model, postprocess, vis
    except ImportError as exc:  # pragma: no cover - external YOLOX environment
        raise SystemExit(
            f"Dependensi inference belum terpasang ({exc}). "
            "Aktifkan environment PyTorch/YOLOX resmi terlebih dahulu."
        ) from exc

    try:
        image_paths, relative_root = _images(args.input)
        labels = _labels()
        device = _device(args.device, torch)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.fp16 and device.type != "cuda":
        raise SystemExit("--fp16 hanya dapat digunakan dengan --device cuda")

    exp = get_exp(str(ROOT / "workshop_yolox_s.py"), None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    exp.test_size = (args.size, args.size)
    if exp.num_classes != len(labels):
        raise SystemExit(
            f"Jumlah kelas experiment ({exp.num_classes}) tidak sama dengan labels.json "
            f"({len(labels)})"
        )

    model = exp.get_model()
    try:
        model.load_state_dict(_checkpoint_state(checkpoint_path, torch))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Gagal memuat checkpoint: {exc}") from exc
    model.eval().to(device)
    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model.half()

    transform = ValTransform(legacy=args.legacy)
    output_root = args.output.expanduser().resolve()
    image_paths = [
        path for path in image_paths if path != output_root and output_root not in path.parents
    ]
    if not image_paths:
        raise SystemExit("Semua input berada di dalam folder output; tidak ada yang diproses")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"PERINGATAN: gambar tidak terbaca, dilewati: {image_path}", file=sys.stderr)
            continue
        height, width = image.shape[:2]
        ratio = min(args.size / height, args.size / width)
        tensor, _ = transform(image, None, exp.test_size)
        tensor = torch.from_numpy(tensor).unsqueeze(0).float().to(device)
        if args.fp16:
            tensor = tensor.half()

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            detections = postprocess(
                model(tensor),
                exp.num_classes,
                args.conf,
                args.nms,
                class_agnostic=True,
            )[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

        if detections is None:
            boxes = torch.empty((0, 4))
            scores = torch.empty(0)
            class_ids = torch.empty(0, dtype=torch.long)
        else:
            detections = detections.detach().cpu()
            boxes = detections[:, :4] / ratio
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            scores = detections[:, 4] * detections[:, 5]
            class_ids = detections[:, 6].to(torch.long)

        rendered = vis(image.copy(), boxes, scores, class_ids, args.conf, labels)
        relative = image_path.relative_to(relative_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), rendered):
            raise SystemExit(f"Gagal menulis hasil: {destination}")

        image_detections = []
        for box, score, class_id in zip(boxes.tolist(), scores.tolist(), class_ids.tolist()):
            image_detections.append(
                {
                    "class_id": class_id,
                    "class_name": labels[class_id],
                    "score": round(float(score), 6),
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                }
            )
        records.append(
            {
                "image": str(image_path),
                "output": str(destination),
                "inference_ms": round(elapsed_ms, 2),
                "detections": image_detections,
            }
        )
        print(f"{relative}: {len(image_detections)} deteksi, {elapsed_ms:.1f} ms")

    if not args.no_json:
        (output_root / "predictions.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Selesai: {len(records)} gambar -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
