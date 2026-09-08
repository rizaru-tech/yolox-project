"""Prepare a Roboflow COCO ZIP for YOLOX, preserving its original splits.
Usage: python prepare_workshop.py "Workshop tools 3.v2i.coco.zip" --output workshop_tools
Dependency: python -m pip install Pillow
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(zip_path, output):
    zip_path, output = Path(zip_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Folder output sudah ada: {output}. Pilih folder baru; tidak ada data ditimpa.')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='workshop_prepare_', dir=output.parent))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError('ZIP memiliki path duplikat.')
            documents = {}
            paths = {}
            for split, source in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
                candidates = [n for n in members if PurePosixPath(n).name == '_annotations.coco.json'
                              and PurePosixPath(n).parent.name == source]
                if len(candidates) != 1:
                    raise ValueError(f'Harus ada tepat satu {source}/_annotations.coco.json')
                paths[split] = PurePosixPath(candidates[0]).parent
                documents[split] = json.loads(archive.read(candidates[0]))

            # Keep categories that have actual boxes anywhere, rather than treating
            # an unused Roboflow parent category as a fourth detection class.
            all_categories = {}
            used_categories = set()
            for split, doc in documents.items():
                cats = {c['id']: c['name'] for c in doc['categories']}
                if len(cats) != len(doc['categories']):
                    raise ValueError(f'Category ID duplikat di {split}')
                for cid, name in cats.items():
                    if cid in all_categories and all_categories[cid] != name:
                        raise ValueError('Nama kategori tidak konsisten antar-split.')
                    all_categories[cid] = name
                for ann in doc['annotations']:
                    if ann['category_id'] not in cats:
                        raise ValueError(f'Kategori anotasi tidak ditemukan di {split}')
                    used_categories.add(ann['category_id'])
            old_ids = sorted(used_categories)
            if not old_ids:
                raise ValueError('Dataset tidak memiliki bounding box.')
            remap = {old: i+1 for i, old in enumerate(old_ids)}
            categories = [{'id': remap[old], 'name': all_categories[old], 'supercategory': 'tool'}
                          for old in old_ids]
            labels = [c['name'] for c in categories]
            (staging / 'annotations').mkdir()
            report = {'source_zip': zip_path.name, 'classes': labels,
                      'removed_unused_categories': [dict(id=k, name=v) for k,v in all_categories.items()
                                                    if k not in used_categories],
                      'split_policy': 'Preserved source train/valid/test; valid renamed val. No resplit.',
                      'splits': {}, 'cross_split_identical_images': [], 'warnings': []}
            seen_hashes = {}
            original_splits = {}
            for split, doc in documents.items():
                folder = staging / split
                folder.mkdir()
                images = {im['id']: im for im in doc['images']}
                if len(images) != len(doc['images']):
                    raise ValueError(f'Image ID duplikat di {split}')
                names = set()
                for im in doc['images']:
                    name = im['file_name']
                    if name != PurePosixPath(name).name or '\\' in name or name in ('.', '..'):
                        raise ValueError(f'Nama gambar harus berupa basename aman: {name}')
                    if name in names:
                        raise ValueError(f'Nama gambar duplikat di {split}: {name}')
                    names.add(name)
                    raw = archive.read(str(paths[split] / name))
                    with Image.open(io.BytesIO(raw)) as picture:
                        picture.load()
                        if picture.size != (im['width'], im['height']):
                            raise ValueError(f'Dimensi gambar tidak sesuai: {name}')
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen_hashes and seen_hashes[digest] != split:
                        raise ValueError(f'Gambar identik ada di split berbeda: {name}')
                    seen_hashes[digest] = split
                    original = im.get('extra', {}).get('name')
                    if original:
                        if original in original_splits and original_splits[original] != split:
                            raise ValueError(f'Nama gambar sumber ada di split berbeda: {original}')
                        original_splits[original] = split
                    (folder / name).write_bytes(raw)
                annotation_ids = set()
                counts = Counter()
                output_anns = []
                for ann in doc['annotations']:
                    if ann['id'] in annotation_ids:
                        raise ValueError(f'Annotation ID duplikat di {split}')
                    annotation_ids.add(ann['id'])
                    if ann['image_id'] not in images:
                        raise ValueError('Anotasi merujuk gambar yang tidak ada.')
                    im = images[ann['image_id']]
                    x,y,w,h = ann['bbox']
                    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0:
                        raise ValueError(f'Bounding box tidak valid: {split}/{ann["id"]}')
                    if x+w > im['width']+0.001 or y+h > im['height']+0.001:
                        raise ValueError('Bounding box di luar gambar.')
                    cleaned = dict(ann)
                    cleaned['category_id'] = remap[ann['category_id']]
                    cleaned.setdefault('iscrowd', 0)
                    cleaned.setdefault('area', w*h)
                    output_anns.append(cleaned)
                    counts[cleaned['category_id']] += 1
                cleaned_doc = dict(doc)
                cleaned_doc['categories'] = categories
                cleaned_doc['annotations'] = output_anns
                write_json(staging/'annotations'/f'instances_{split}.json', cleaned_doc)
                labeled_ids = {a['image_id'] for a in output_anns}
                report['splits'][split] = {'images': len(images), 'bounding_boxes':len(output_anns),
                    'boxes_per_class':{c['name']: counts[c['id']] for c in categories},
                    'negative_images':len(set(images)-labeled_ids)}
                print(f'{split}: {len(images)} gambar, {len(output_anns)} bounding box')
            report['warnings'] = ['Near-duplicate visual belum diperiksa.',
                                   f"Test berisi {report['splits']['test']['images']} gambar; nilai kecukupannya sesuai penggunaan."]
            write_json(staging/'labels.json', labels)
            write_json(staging/'class_mapping.json', [dict(model_index=i, category_id=i+1,
                       source_category_id=old, name=all_categories[old]) for i,old in enumerate(old_ids)])
            write_json(staging/'preparation_report.json', report)
            # Retain attribution and license documents supplied with the ZIP.
            for name in ('README.dataset.txt', 'README.roboflow.txt'):
                candidates=[n for n in members if PurePosixPath(n).name == name]
                if len(candidates)==1:
                    (staging/name).write_bytes(archive.read(candidates[0]))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f'SELESAI: {output}')
    print('Urutan label:', labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip_file', type=Path)
    parser.add_argument('--output', type=Path, default=Path('workshop_tools'))
    args = parser.parse_args()
    try:
        prepare(args.zip_file, args.output)
    except Exception as exc:
        parser.exit(1, f'GAGAL: {exc}\n')
