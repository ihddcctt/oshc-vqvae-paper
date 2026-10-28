from __future__ import annotations

import numpy as np
import pytest

from lightwave_seeded_vqvae.cli.plot_reconstruction_examples import (
    cross_pump_same_power_state,
    row_relative_mse,
    select_median_per_class,
    shifted_receiver_schedule,
)


def test_select_median_per_class_reduces_states_and_breaks_ties_by_index() -> None:
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    per_state_mse = np.asarray(
        [
            [0.30, 0.30],
            [0.10, 0.10],
            [0.20, 0.20],
            [3.00, 3.00],
            [1.00, 1.00],
            [2.00, 2.00],
        ]
    )
    assert select_median_per_class(targets, per_state_mse, class_count=2) == [2, 5]


def test_select_median_per_class_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="no sample for class 1"):
        select_median_per_class([0, 0], np.asarray([0.1, 0.2]), class_count=2)


def test_cross_pump_mapping_and_shifted_schedule() -> None:
    assert [cross_pump_same_power_state(index) for index in range(10)] == [
        5,
        6,
        7,
        8,
        9,
        0,
        1,
        2,
        3,
        4,
    ]
    np.testing.assert_array_equal(
        shifted_receiver_schedule(np.arange(10), 1),
        np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    )


def test_row_relative_mse_makes_each_diagonal_one() -> None:
    matrix = np.asarray([[2.0, 8.0], [9.0, 3.0]])
    relative = row_relative_mse(matrix)
    np.testing.assert_allclose(relative, np.asarray([[1.0, 4.0], [3.0, 1.0]]))


def test_row_relative_mse_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        row_relative_mse(np.asarray([[1.0, 0.0], [2.0, 1.0]]))
