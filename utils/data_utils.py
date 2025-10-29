import os
from pathlib import Path
import random
import shutil
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from PIL import Image
MAX_PIXELS = 50_000_000

# To transform name to class
NAME2CLS = {
    "RandomResizedCrop": T.RandomResizedCrop,
    "RandomHorizontalFlip": T.RandomHorizontalFlip,
    "ColorJitter": T.ColorJitter,
    "ToTensor": T.ToTensor,
    "Normalize": T.Normalize,
    "CenterCrop": T.CenterCrop,
    "Resize": T.Resize,
}

def _build(seq_cfg):
    """
    Build a sequence of transforms from configuration.
    """
    ops = []
    for item in seq_cfg:
        # print(item.items())
        name, params = list(item.items())[0]
        # print(name)
        # print(params)
        op_cls = NAME2CLS[name]
        ops.append(op_cls(**params))
    return T.Compose(ops)
    
def _build_transforms(cfg):
    """
    cfg: Config file includes the transforms config for train, val, test
    """
    t_train = _build(cfg["train"])
    t_val = _build(cfg["val"])
    t_test = _build(cfg["test"])
    return {'train': t_train, 'val': t_val, 'test': t_test}

def _build_dataset(root, train_dir, val_dir, test_dir, transforms):
    """
    Build datasets for train, val, test
    """
    root = Path(root)
    train_dataset = ImageFolder(root / train_dir, transform=transforms["train"])
    val_dataset = ImageFolder(root / val_dir, transform=transforms["val"])
    test_dataset = ImageFolder(root / test_dir, transform=transforms["test"])
    return train_dataset, val_dataset, test_dataset

def _rename_images(class_dir, class_name, images):
    """
    Rename images in a class folder to have consistent naming
    """
    for idx, img_name in enumerate(images):
        src = class_dir / img_name
        ext = src.suffix.lower()
        new_name = class_dir / f"{class_name}_{idx}{ext}"
        src.rename(new_name)

def _should_skip(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
        return w * h > MAX_PIXELS
    except OSError:
        print(f"Skipping huge image: {path}")
        return True

def _process_dataset(cfg, skip_large=True):
    """
    Split dataset into train, val, test if not already processed
    """
    root = Path(cfg["root"])
    raw_dir = Path(cfg["root_raw"])
    train_dir = root / cfg["train_dir"]
    val_dir = root / cfg["val_dir"]
    test_dir = root / cfg["test_dir"]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # automatically detect classes folder in raw_dir
    classes = os.listdir(raw_dir)
    train_ratio, val_ratio, test_ratio = cfg["split_ratio"]
    print(f"Detected {len(classes)} classes: {classes}")
    print(f"Splitting dataset with ratio Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    for class_name in classes:
        class_path = raw_dir / class_name
        if not class_path.is_dir():
            continue # skip non-directory files
        images = os.listdir(class_path)
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        print(f"Class '{class_name}': Total={n_total}, Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        # create class directories in processed folders
        os.makedirs(train_dir / class_name, exist_ok=True)
        os.makedirs(val_dir / class_name, exist_ok=True)
        os.makedirs(test_dir / class_name, exist_ok=True)

        # copy images to respective folders
        for img_name in train_images:
            src = class_path / img_name
            if _should_skip(src):
                continue
            shutil.copy(class_path / img_name, train_dir / class_name / img_name)
        for img_name in val_images:
            src = class_path / img_name
            if _should_skip(src):
                continue
            shutil.copy(class_path / img_name, val_dir / class_name / img_name)
        for img_name in test_images:
            src = class_path / img_name
            if _should_skip(src):
                continue
            shutil.copy(class_path / img_name, test_dir / class_name / img_name)

        # Rename images in processed folders
        _rename_images(train_dir / class_name, class_name, os.listdir(train_dir / class_name))
        _rename_images(val_dir / class_name, class_name, os.listdir(val_dir / class_name))
        _rename_images(test_dir / class_name, class_name, os.listdir(test_dir / class_name))