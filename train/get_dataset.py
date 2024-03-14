from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, Sag3DDataset, DEFAULTDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'SAG3D':
        train_dataset = Sag3DDataset(
            root_dir=cfg.dataset.train_root_dir,
            target_size=cfg.dataset.target_size,
            num_frames=cfg.dataset.num_frames,
            axes=cfg.dataset.axes,
            num_samples_per_npz=cfg.dataset.num_samples_per_npz,
            output_mode=cfg.dataset.output_mode,
            output_with_info=cfg.dataset.output_with_info
        )
        val_dataset = Sag3DDataset(
            root_dir=cfg.dataset.val_root_dir,
            target_size=cfg.dataset.target_size,
            num_frames=cfg.dataset.num_frames,
            axes=cfg.dataset.axes,
            num_samples_per_npz=cfg.dataset.num_samples_per_npz,
            output_mode=cfg.dataset.output_mode,
            output_with_info=cfg.dataset.output_with_info
        )
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'MRNet':
        train_dataset = MRNetDataset(
            root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split='train')
        val_dataset = MRNetDataset(root_dir=cfg.dataset.root_dir,
                                   task=cfg.dataset.task, plane=cfg.dataset.plane, split='valid')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'BRATS':
        train_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        val_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'ADNI':
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DUKE':
        train_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'LIDC':
        train_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
