from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightwave_seeded_vqvae.cli import train
from lightwave_seeded_vqvae.cli.train import (
    DistributedContext,
    _distributed_batch_size,
    _global_mean_metrics,
    _select_checkpoint,
    _selection_collapse_reasons,
    _torchrun_environment,
    _unwrap_model,
    _with_distributed_sampler,
    build_parser,
)


def _clear_torchrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)


def _context(*, rank: int = 0, world_size: int = 2) -> DistributedContext:
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        device=torch.device("cpu"),
        backend="nccl",
        owns_process_group=True,
    )


def test_parser_accepts_both_torchrun_local_rank_spellings() -> None:
    dashed = build_parser().parse_args(["--config", "config.yaml", "--local-rank", "2"])
    underscored = build_parser().parse_args(
        ["--config", "config.yaml", "--local_rank", "3"]
    )

    assert dashed.local_rank == 2
    assert underscored.local_rank == 3


def test_torchrun_environment_is_absent_or_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_torchrun_environment(monkeypatch)
    assert _torchrun_environment() is None

    monkeypatch.setenv("RANK", "1")
    with pytest.raises(RuntimeError, match="incomplete torchrun environment"):
        _torchrun_environment()

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert _torchrun_environment() == (1, 4, 1)


def test_initialize_torchrun_binds_local_cuda_and_uses_nccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    calls: dict[str, object] = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.setdefault("gpu", rank))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def fake_init_process_group(**kwargs: object) -> None:
        calls["init"] = kwargs

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init_process_group)

    context = train._initialize_distributed("cuda:0")

    assert calls["gpu"] == 2
    assert calls["init"]["backend"] == "nccl"  # type: ignore[index]
    assert calls["init"]["init_method"] == "env://"  # type: ignore[index]
    assert calls["init"]["rank"] == 2  # type: ignore[index]
    assert calls["init"]["world_size"] == 4  # type: ignore[index]
    assert context.device == torch.device("cuda", 2)
    assert context.owns_process_group


def test_distributed_sampler_shards_data_and_preserves_per_rank_batch() -> None:
    dataset = TensorDataset(torch.arange(24))
    original = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    rank_zero_loader, rank_zero_sampler = _with_distributed_sampler(
        original,
        _context(rank=0),
        seed=71,
        per_rank_batch_size=4,
    )
    rank_one_loader, rank_one_sampler = _with_distributed_sampler(
        original,
        _context(rank=1),
        seed=71,
        per_rank_batch_size=4,
    )
    assert rank_zero_sampler is not None
    assert rank_one_sampler is not None
    rank_zero_sampler.set_epoch(3)
    rank_one_sampler.set_epoch(3)
    rank_zero_indices = set(iter(rank_zero_sampler))
    rank_one_indices = set(iter(rank_one_sampler))

    assert rank_zero_loader.batch_size == 4
    assert rank_one_loader.batch_size == 4
    assert rank_zero_indices.isdisjoint(rank_one_indices)
    assert rank_zero_indices | rank_one_indices == set(range(24))


def test_per_rank_batch_is_default_semantics_for_stl_four_gpu() -> None:
    context = _context(rank=0, world_size=4)

    assert _distributed_batch_size(64, context, "per_rank") == 64
    assert 64 * context.world_size == 256
    assert _distributed_batch_size(256, context, "global") == 64
    with pytest.raises(ValueError, match="divisible"):
        _distributed_batch_size(255, context, "global")


def test_global_training_metrics_reduce_sum_and_batch_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"loss": 1.0, "mse": 2.0}, {"loss": 3.0, "mse": 4.0}]

    def fake_all_reduce(payload: torch.Tensor, *, op: object) -> None:
        assert op is torch.distributed.ReduceOp.SUM
        # Sorted keys are loss, mse; add the other rank's sums and batch count.
        payload.add_(torch.tensor([10.0, 2.0, 2.0], dtype=payload.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    metrics = _global_mean_metrics(rows, _context(rank=0))

    assert metrics == pytest.approx({"loss": 3.5, "mse": 2.0})


def test_correct_validation_selection_ignores_binding_gate() -> None:
    selected = _select_checkpoint(
        "correct_validation_mse",
        binding_evaluation_enabled=True,
        binding_training_enabled=False,
        has_binding_checkpoint=True,
        has_post_ramp_checkpoint=True,
    )
    historical = _select_checkpoint(
        "historical_binding",
        binding_evaluation_enabled=True,
        binding_training_enabled=False,
        has_binding_checkpoint=True,
        has_post_ramp_checkpoint=True,
    )

    assert selected == (
        "checkpoint_best_reconstruction.pt",
        "best_correct_state_validation_mse",
    )
    assert historical[0] == "checkpoint_best_binding.pt"

    collapse_metrics = {
        "mse": 0.03,
        "psnr_db": 10.0,
        "reconstruction_output_std": 0.0,
    }
    assert _selection_collapse_reasons(
        collapse_metrics,
        checkpoint_selection="correct_validation_mse",
        binding_evaluation_enabled=True,
        previous_validation_psnr=24.0,
        best_binding_mse=0.003,
        has_binding_checkpoint=True,
    ) == []
    assert _selection_collapse_reasons(
        collapse_metrics,
        checkpoint_selection="historical_binding",
        binding_evaluation_enabled=True,
        previous_validation_psnr=24.0,
        best_binding_mse=0.003,
        has_binding_checkpoint=True,
    ) == [
        "validation_mse_exceeded_guard",
        "validation_psnr_dropped_over_3db",
        "reconstruction_output_collapsed",
    ]


def test_checkpoint_state_unwraps_ddp_module_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = torch.nn.Linear(2, 1)

    class FakeDistributedDataParallel:
        def __init__(self, wrapped: torch.nn.Module) -> None:
            self.module = wrapped

    monkeypatch.setattr(train, "DistributedDataParallel", FakeDistributedDataParallel)
    wrapped = FakeDistributedDataParallel(module)

    assert _unwrap_model(wrapped) is module
    assert all(not key.startswith("module.") for key in _unwrap_model(wrapped).state_dict())


def test_main_destroys_owned_process_group_after_training_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(rank=0)
    destroyed: list[bool] = []
    notifications: list[tuple[bool, bool]] = []
    monkeypatch.setattr(train, "_initialize_distributed", lambda device: context)

    def fail_rank_zero_phase(args: object, distributed: DistributedContext) -> None:
        distributed.rank_zero_phase_active = True
        raise RuntimeError("boom")

    monkeypatch.setattr(train, "_run_training", fail_rank_zero_phase)
    monkeypatch.setattr(
        train,
        "_broadcast_rank_zero_phase",
        lambda success, stop, distributed: (
            notifications.append((success, stop)) or (success, stop)
        ),
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda: destroyed.append(True)
    )

    with pytest.raises(RuntimeError, match="boom"):
        train.main(["--config", "unused.yaml", "--device", "cuda:0"])

    assert notifications == [(False, True)]
    assert destroyed == [True]
