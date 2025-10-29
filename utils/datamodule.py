from torch.utils.data import DataLoader
from utils.data_utils import _build_transforms, _build_dataset, _process_dataset

class DataModule:
    def __init__(self, dataset_cfg):
        self.dataset_cfg = dataset_cfg
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def setup(self, processed: bool = True, skip_large=True):
        transforms = _build_transforms(self.dataset_cfg)
        if not processed:
            _process_dataset(self.dataset_cfg, skip_large=skip_large)
        self.ds_train, self.ds_val, self.ds_test = _build_dataset(
            root=self.dataset_cfg["root"],
            train_dir=self.dataset_cfg["train_dir"],
            val_dir=self.dataset_cfg["val_dir"],
            test_dir=self.dataset_cfg["test_dir"],
            transforms=transforms,
        )

    def train_loader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.dataset_cfg["batch_size"],
            shuffle=self.dataset_cfg["shuffle_train"],
            num_workers=self.dataset_cfg["num_workers"],
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            persistent_workers=self.dataset_cfg.get("persistent_workers", False),
        )
    
    def val_loader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.dataset_cfg["batch_size"],
            shuffle=False,
            num_workers=self.dataset_cfg["num_workers"],
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            persistent_workers=self.dataset_cfg.get("persistent_workers", False),
        )
    
    def test_loader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.dataset_cfg["batch_size"],
            shuffle=False,
            num_workers=self.dataset_cfg["num_workers"],
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            persistent_workers=self.dataset_cfg.get("persistent_workers", False),
        )