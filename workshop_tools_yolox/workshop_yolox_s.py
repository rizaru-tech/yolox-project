"""YOLOX-S experiment for the local Workshop Tools COCO dataset."""

import json
import os
from pathlib import Path

from yolox.exp import Exp as BaseExp


ROOT = Path(__file__).resolve().parent


def _dataset_root() -> Path:
    configured = os.environ.get("WORKSHOP_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else ROOT / "workshop_tools"


def _load_labels(data_dir: Path) -> list[str]:
    labels_file = data_dir / "labels.json"
    if not labels_file.is_file():
        raise FileNotFoundError(
            f"Label file tidak ditemukan: {labels_file}. "
            "Set WORKSHOP_DATA_DIR ke root dataset yang benar."
        )
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValueError(f"Format labels.json tidak valid: {labels_file}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Nama kelas harus unik: {labels_file}")
    return labels


class Exp(BaseExp):
    """Baseline fine-tuning YOLOX-S while preserving the held-out test split."""

    def __init__(self):
        super().__init__()

        data_dir = _dataset_root()
        labels = _load_labels(data_dir)

        # YOLOX-S architecture.
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = Path(__file__).stem

        # COCO annotations live in <data_dir>/annotations; images do not use
        # COCO's train2017/val2017 directory names.
        self.data_dir = str(data_dir)
        # Generated from instances_train.json by build_training_annotations.py;
        # only clearly degenerate sub-pixel label artefacts are removed.
        self.train_ann = "instances_train.yolox.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"
        self.num_classes = len(labels)

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = min(4, os.cpu_count() or 1)
        self.eval_interval = 1

        # Keep the official YOLOX baseline schedule and augmentation defaults:
        # 300 epochs, EMA, multi-scale, Mosaic, MixUp, and a 15-epoch no-aug
        # tail. Tune only after recording this baseline on validation data.
        self.max_epoch = 300
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        self.ema = True

        # COCO-style evaluation should retain low-score detections before AP is
        # calculated. A higher confidence threshold belongs to inference only.
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform

        if os.environ.get("WORKSHOP_EVAL_SPLIT", "val").lower() != "val":
            raise ValueError(
                "Set WORKSHOP_EVAL_SPLIT=val sebelum training agar test tetap held out."
            )
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        if kwargs.get("testdev", False):
            raise ValueError(
                "Jangan gunakan COCO testdev untuk dataset lokal; "
                "set WORKSHOP_EVAL_SPLIT=test."
            )
        split = os.environ.get("WORKSHOP_EVAL_SPLIT", "val").lower()
        if split not in {"val", "test"}:
            raise ValueError("WORKSHOP_EVAL_SPLIT harus 'val' atau 'test'.")
        return COCODataset(
            data_dir=self.data_dir,
            json_file=f"instances_{split}.json",
            name=split,
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )
