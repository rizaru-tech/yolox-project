"""YOLOX-S experiment using train/val/test directories, with no COCO year names."""
import json
import os
from pathlib import Path
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.data_dir = str(Path(os.environ.get('WORKSHOP_DATA_DIR',
            str(Path(__file__).resolve().parent/'workshop_tools'))).resolve())
        self.num_classes = len(json.loads((Path(self.data_dir)/'labels.json').read_text(encoding='utf-8')))
        self.depth, self.width = 0.33, 0.50
        self.exp_name = 'workshop_yolox_s'
        self.train_ann = 'instances_train.json'
        self.val_ann = 'instances_val.json'
        self.test_ann = 'instances_test.json'
        self.input_size = self.test_size = (640,640)
        self.multiscale_range = 0
        self.max_epoch = 30
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.eval_interval = 1
        self.data_num_workers = 0
        self.seed = 42
        # Source training data already contains offline augmentation.
        self.enable_mixup = False
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0

    def get_dataset(self, cache=False, cache_type='ram'):
        from yolox.data import COCODataset, TrainTransform
        if os.environ.get('WORKSHOP_EVAL_SPLIT','val') != 'val':
            raise ValueError('Set WORKSHOP_EVAL_SPLIT=val before training; test is held out.')
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann,
            name='train', img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache, cache_type=cache_type)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        if kwargs.get('testdev', False):
            raise ValueError('Do not use official COCO testdev. Set WORKSHOP_EVAL_SPLIT=test instead.')
        split = os.environ.get('WORKSHOP_EVAL_SPLIT','val')
        if split not in ('val','test'):
            raise ValueError('WORKSHOP_EVAL_SPLIT must be val or test')
        return COCODataset(data_dir=self.data_dir, json_file=f'instances_{split}.json',
            name=split, img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get('legacy',False)))
