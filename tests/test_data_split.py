from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from lightwave_seeded_vqvae import data


def test_train_validation_split_is_held_out_and_deterministic(monkeypatch) -> None:
    training_set = TensorDataset(torch.arange(20))
    test_set = TensorDataset(torch.arange(7))

    def fake_build_dataset(name: str, root: str, *, train: bool, download: bool):
        return training_set if train else test_set

    monkeypatch.setattr(data, "build_dataset", fake_build_dataset)
    dataset_config = {
        "name": "mnist",
        "root": ".",
        "download": False,
        "validation_size": 5,
    }
    training_config = {"batch_size": 4, "num_workers": 0}
    first = data.build_dataloaders(
        dataset_config, training_config, split_seed=20260907
    )
    second = data.build_dataloaders(
        dataset_config, training_config, split_seed=20260907
    )
    first_train, first_validation, first_test = first
    second_train, second_validation, second_test = second
    assert len(first_train.dataset) == 15
    assert len(first_validation.dataset) == 5
    assert len(first_test.dataset) == 7
    assert set(first_train.dataset.indices).isdisjoint(first_validation.dataset.indices)
    assert first_train.dataset.indices == second_train.dataset.indices
    assert first_validation.dataset.indices == second_validation.dataset.indices
    assert list(iter(first_train.batch_sampler)) == list(iter(second_train.batch_sampler))
    assert len(second_test.dataset) == 7


def test_stl10_uses_fixed_official_splits_and_no_test_augmentation(monkeypatch) -> None:
    sources = {
        "unlabeled": TensorDataset(torch.arange(100)),
        "train": TensorDataset(torch.arange(5)),
        "test": TensorDataset(torch.arange(8)),
    }
    calls: list[dict[str, object]] = []

    def fake_build_dataset(
        name: str,
        root: str,
        *,
        train: bool,
        download: bool,
        split: str | None = None,
        transform=None,
    ):
        calls.append(
            {
                "name": name,
                "train": train,
                "download": download,
                "split": split,
                "transform": transform,
            }
        )
        assert split is not None
        return sources[split]

    monkeypatch.setattr(data, "build_dataset", fake_build_dataset)
    dataset_config = {
        "name": "stl10",
        "root": ".",
        "download": False,
        "train_split": "unlabeled",
        "validation_split": "train",
        "test_split": "test",
        "validation_size": 5,
        "train_horizontal_flip_probability": 0.5,
    }
    training_config = {"batch_size": 4, "num_workers": 0}

    train, validation, test = data.build_dataloaders(
        dataset_config, training_config, split_seed=20260907
    )

    assert len(train.dataset) == 100
    assert len(validation.dataset) == 5
    assert len(test.dataset) == 8
    assert [call["split"] for call in calls] == ["unlabeled", "train", "test"]
    assert calls[0]["train"] is True
    assert calls[1]["train"] is True
    assert calls[2]["train"] is False
    assert type(calls[0]["transform"].transforms[0]).__name__ == (
        "RandomHorizontalFlip"
    )
    assert len(calls[1]["transform"].transforms) == 1
    assert len(calls[2]["transform"].transforms) == 1


def test_stl10_rejects_test_reuse_or_partial_official_validation(monkeypatch) -> None:
    source = TensorDataset(torch.arange(5))

    def fake_build_dataset(*args, split: str | None = None, **kwargs):
        return source

    monkeypatch.setattr(data, "build_dataset", fake_build_dataset)
    training_config = {"batch_size": 2, "num_workers": 0}

    with pytest.raises(ValueError, match="must be distinct"):
        data.build_dataloaders(
            {
                "name": "stl10",
                "root": ".",
                "train_split": "unlabeled",
                "validation_split": "train",
                "test_split": "train",
                "validation_size": 5,
            },
            training_config,
            split_seed=1,
        )

    with pytest.raises(ValueError, match="complete fixed validation"):
        data.build_dataloaders(
            {
                "name": "stl10",
                "root": ".",
                "train_split": "unlabeled",
                "validation_split": "train",
                "test_split": "test",
                "validation_size": 4,
            },
            training_config,
            split_seed=1,
        )
