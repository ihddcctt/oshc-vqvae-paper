from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split
from torchvision import datasets, transforms


def resolve_dataset_root(
    recorded_root: str | Path,
    override: str | Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Locate a moved dataset without modifying checkpoint/registry provenance."""
    effective = Path(override if override is not None else recorded_root)
    effective = effective.expanduser().resolve()
    return effective, {
        "recorded_root": str(recorded_root),
        "effective_root": str(effective),
        "override_supplied": override is not None,
    }


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "")


def _image_transform(*, horizontal_flip_probability: float = 0.0) -> Callable:
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    operations: list[Callable] = []
    if horizontal_flip_probability > 0.0:
        operations.append(
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability)
        )
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def build_dataset(
    name: str,
    root: str | Path,
    *,
    train: bool,
    download: bool,
    split: str | None = None,
    transform: Callable | None = None,
) -> Dataset:
    normalized = normalize_dataset_name(name)
    image_transform = transform or transforms.ToTensor()
    root_path = str(Path(root).expanduser())
    if normalized == "mnist":
        return datasets.MNIST(
            root_path, train=train, transform=image_transform, download=download
        )
    if normalized == "fashionmnist":
        return datasets.FashionMNIST(
            root_path, train=train, transform=image_transform, download=download
        )
    if normalized == "kmnist":
        return datasets.KMNIST(
            root_path, train=train, transform=image_transform, download=download
        )
    if normalized == "cifar10":
        return datasets.CIFAR10(
            root_path, train=train, transform=image_transform, download=download
        )
    if normalized == "stl10":
        selected_split = split or ("unlabeled" if train else "test")
        if selected_split not in {"train", "test", "unlabeled", "train+unlabeled"}:
            raise ValueError(f"unsupported STL-10 split: {selected_split}")
        return datasets.STL10(
            root_path,
            split=selected_split,
            transform=image_transform,
            download=download,
        )
    raise ValueError(f"unsupported dataset: {name}")


def dataset_channels(name: str) -> int:
    normalized = normalize_dataset_name(name)
    if normalized in {"mnist", "fashionmnist", "kmnist"}:
        return 1
    if normalized in {"cifar10", "stl10"}:
        return 3
    raise ValueError(f"unsupported dataset: {name}")


def build_dataloaders(
    dataset_config: dict[str, object],
    training_config: dict[str, object],
    *,
    split_seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    name = str(dataset_config["name"])
    root = str(dataset_config["root"])
    download = bool(dataset_config.get("download", False))
    normalized = normalize_dataset_name(name)
    if normalized == "stl10":
        train_split = str(dataset_config.get("train_split", "unlabeled"))
        validation_split = str(dataset_config.get("validation_split", "train"))
        test_split = str(dataset_config.get("test_split", "test"))
        if len({train_split, validation_split, test_split}) != 3:
            raise ValueError("STL-10 train, validation, and test splits must be distinct")
        if test_split != "test":
            raise ValueError("STL-10 final evaluation must use the official test split")
        flip_probability = float(
            dataset_config.get("train_horizontal_flip_probability", 0.0)
        )
        train_source = build_dataset(
            name,
            root,
            train=True,
            download=download,
            split=train_split,
            transform=_image_transform(
                horizontal_flip_probability=flip_probability
            ),
        )
        validation_source = build_dataset(
            name,
            root,
            train=True,
            download=download,
            split=validation_split,
            transform=_image_transform(),
        )
        test_dataset = build_dataset(
            name,
            root,
            train=False,
            download=download,
            split=test_split,
            transform=_image_transform(),
        )
        expected_validation_size = int(
            dataset_config.get("validation_size", len(validation_source))
        )
        if expected_validation_size != len(validation_source):
            raise ValueError(
                "STL-10 validation_size must equal the complete fixed validation "
                f"split ({len(validation_source)}), got {expected_validation_size}"
            )
        # Subset preserves explicit source-local IDs in run metadata and gives all
        # loaders the same `.indices` audit surface as the historical random split.
        train_dataset = Subset(train_source, range(len(train_source)))
        validation_dataset = Subset(
            validation_source, range(len(validation_source))
        )
    else:
        full_train_dataset = build_dataset(
            name, root, train=True, download=download
        )
        test_dataset = build_dataset(name, root, train=False, download=download)
        validation_size = int(dataset_config.get("validation_size", 5000))
        if not 1 <= validation_size < len(full_train_dataset):
            raise ValueError(
                f"validation_size must be in [1, {len(full_train_dataset) - 1}], "
                f"got {validation_size}"
            )
        train_size = len(full_train_dataset) - validation_size
        train_dataset, validation_dataset = random_split(
            full_train_dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(split_seed),
        )
    batch_size = int(training_config["batch_size"])
    workers = int(training_config.get("num_workers", 4))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(split_seed + 1),
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, validation_loader, test_loader
