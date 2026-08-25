import pandas as pd
import pytest

from forecasting.features.xgboost import (
    XGB_FEATURE_SETS,
    merge_external_features,
)
from forecasting.backtests.xgboost_permitted_events import (
    BASELINE_EXPERIMENT_NAME,
    BASELINE_NUMERIC_FEATURES,
    PERMIT_XGB_FEATURE_SETS,
    build_permit_feature_specs,
    compare_candidate_to_baseline,
)


def make_feature_panel():
    return pd.DataFrame(
        {
            "target_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ],
            "neighborhood": [
                "A",
                "B",
                "A",
            ],
            "calls": [1, 2, 3],
            "calls_lag_1": [0, 1, 2],
        }
    )


def make_external_panel():
    return pd.DataFrame(
        {
            "target_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ],
            "neighborhood": [
                "A",
                "B",
                "A",
            ],
            "se_permit_count": [
                5,
                0,
                1,
            ],
            "se_log_split_attendance_sum": [
                2.0,
                0.0,
                1.0,
            ],
        }
    )


def test_external_merge_preserves_row_count():
    feature_panel = make_feature_panel()
    external_panel = make_external_panel()

    merged = merge_external_features(
        feature_panel=feature_panel,
        external_panel=external_panel,
        feature_columns=[
            "se_permit_count",
        ],
    )

    assert len(merged) == len(feature_panel)
    assert list(merged.columns) == [
        *feature_panel.columns,
        "se_permit_count",
    ]

    expected_keys = (
        feature_panel.assign(
            target_date=pd.to_datetime(
                feature_panel["target_date"]
            ).dt.normalize()
        )[
            [
                "target_date",
                "neighborhood",
            ]
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        merged[
            [
                "target_date",
                "neighborhood",
            ]
        ].reset_index(drop=True),
        expected_keys,
        check_dtype=False,
    )


def test_duplicate_external_date_neighborhood_keys_raise():
    external_panel = pd.concat(
        [
            make_external_panel(),
            pd.DataFrame(
                {
                    "target_date": [
                        "2024-01-01"
                    ],
                    "neighborhood": [
                        "A"
                    ],
                    "se_permit_count": [9],
                    "se_log_split_attendance_sum": [
                        9.0
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="external_panel contains duplicate",
    ):
        merge_external_features(
            feature_panel=
                make_feature_panel(),
            external_panel=
                external_panel,
            feature_columns=[
                "se_permit_count",
            ],
        )


def test_missing_permit_feature_column_raises():
    external_panel = make_external_panel().drop(
        columns="se_permit_count"
    )

    with pytest.raises(
        ValueError,
        match="missing requested columns",
    ):
        merge_external_features(
            feature_panel=
                make_feature_panel(),
            external_panel=
                external_panel,
            feature_columns=[
                "se_permit_count",
            ],
        )


def test_missing_permit_coverage_raises():
    external_panel = make_external_panel().iloc[
        :-1
    ].copy()

    with pytest.raises(
        ValueError,
        match="introduced missing feature coverage",
    ):
        merge_external_features(
            feature_panel=
                make_feature_panel(),
            external_panel=
                external_panel,
            feature_columns=[
                "se_permit_count",
            ],
        )


def test_baseline_contains_no_permit_features():
    permit_features = {
        feature
        for features in
        PERMIT_XGB_FEATURE_SETS.values()
        for feature in features
    }

    assert (
        BASELINE_EXPERIMENT_NAME
        == "baseline"
    )
    assert BASELINE_NUMERIC_FEATURES == [
        *XGB_FEATURE_SETS[
            "lags_rolling_calendar"
        ]
    ]
    assert not (
        permit_features
        & set(
            BASELINE_NUMERIC_FEATURES
        )
    )


def test_candidate_feature_lists_extend_baseline_only_with_intended_permit_variables():
    specs = build_permit_feature_specs()

    assert specs[
        BASELINE_EXPERIMENT_NAME
    ] == BASELINE_NUMERIC_FEATURES

    for (
        candidate,
        permit_features,
    ) in PERMIT_XGB_FEATURE_SETS.items():
        assert specs[candidate] == [
            *BASELINE_NUMERIC_FEATURES,
            *permit_features,
        ]


def test_paired_comparison_requires_exact_one_to_one_fold_neighborhood_matches():
    baseline = pd.DataFrame(
        {
            "fold": [1, 1],
            "neighborhood": ["A", "B"],
            "mae": [1.0, 1.0],
            "rmse": [2.0, 2.0],
            "mase": [0.8, 0.8],
            "smape": [10.0, 10.0],
            "bias": [0.1, 0.1],
        }
    )

    candidate = pd.DataFrame(
        {
            "fold": [1],
            "neighborhood": ["A"],
            "mae": [0.9],
            "rmse": [1.9],
            "mase": [0.7],
            "smape": [9.0],
            "bias": [0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="exact same fold/neighborhood keys",
    ):
        compare_candidate_to_baseline(
            candidate_metrics=
                candidate,
            baseline_metrics=
                baseline,
            candidate_name=
                "permit_count",
        )


def test_delta_mase_sign_is_candidate_minus_baseline():
    baseline = pd.DataFrame(
        {
            "fold": [1],
            "neighborhood": ["A"],
            "mae": [1.0],
            "rmse": [2.0],
            "mase": [0.8],
            "smape": [10.0],
            "bias": [0.1],
        }
    )

    candidate = pd.DataFrame(
        {
            "fold": [1],
            "neighborhood": ["A"],
            "mae": [0.9],
            "rmse": [1.9],
            "mase": [0.6],
            "smape": [9.0],
            "bias": [0.0],
        }
    )

    comparison = compare_candidate_to_baseline(
        candidate_metrics=
            candidate,
        baseline_metrics=
            baseline,
        candidate_name=
            "permit_count",
    )

    row = comparison.iloc[0]

    assert row["delta_mae"] == pytest.approx(
        -0.1
    )
    assert row["delta_mase"] == pytest.approx(
        -0.2
    )
    assert row["improved_mase"]
