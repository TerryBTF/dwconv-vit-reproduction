import torch
import torch.distributed as dist
from timm.data import Mixup
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_loader(config):
    dataset_train = build_dataset(is_train=True, config=config)
    dataset_val = build_dataset(is_train=False, config=config)

    if dist.is_available() and dist.is_initialized():
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
    else:
        sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0 or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    dataset_name = config.DATA.DATASET.lower()

    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=config.DATA.DATA_PATH,
            train=is_train,
            transform=transform,
            download=False,
        )
    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=config.DATA.DATA_PATH,
            train=is_train,
            transform=transform,
            download=False,
        )

    raise NotImplementedError(f"Unsupported dataset: {config.DATA.DATASET}")


def build_transform(is_train, config):
    size = config.DATA.IMG_SIZE
    interpolation = transforms.InterpolationMode.BICUBIC

    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    if config.TEST.CROP:
        return transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
