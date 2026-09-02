import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.backtests.xgboost import (
    DEFAULT_XGB_REGRESSOR_PARAMS,
    build_standard_backtest_folds,
    build_xgboost_pipeline,
)
from forecasting.features.xgboost import (
    build_xgboost_feature_panel,
    prepare_target_panel,
)
from forecasting.tuning.xgboost_regressor import (
    canonicalize_params,
    evaluate_parameter_configuration,
    generate_parameter_candidates,
    make_config_id,
    resolve_worker_count,
    run_tuning,
    validate_folds_exclude_final_test_period,
)


def make_target_panel(
    n_days: int = 800,
    neighborhoods: tuple[str, ...] = (
        "A",
        "B",
    ),
) -> pd.DataFrame:
    dates = pd.date_range(
        "2023-01-01",
        periods=n_days,
        freq="D",
    )

    rows = []

    for index, target_date in enumerate(
        dates
    ):
        for neighborhood_index, neighborhood in enumerate(
            neighborhoods
        ):
            rows.append(
                {
                    "target_date": target_date,
                    "neighborhood": neighborhood,
                    "calls": float(
                        20
                        + neighborhood_index * 3
                        + (index % 7)
                        + (index // 14)
                    ),
                }
            )

    return prepare_target_panel(
        pd.DataFrame(rows)
    )


def make_small_feature_panel():
    target_panel = make_target_panel(
        n_days=70
    )
    feature_panel = (
        build_xgboost_feature_panel(
            target_panel
        )
    )

    folds = pd.DataFrame(
        {
            "fold": [1],
            "train_start": [
                pd.Timestamp("2023-01-29")
            ],
            "train_end": [
                pd.Timestamp("2023-02-25")
            ],
            "val_start": [
                pd.Timestamp("2023-02-26")
            ],
            "val_end": [
                pd.Timestamp("2023-03-05")
            ],
        }
    )

    return target_panel, feature_panel, folds


def test_build_xgboost_pipeline_uses_default_params():
    pipeline = build_xgboost_pipeline(
        numeric_features=["calls_lag_1"]
    )

    assert (
        pipeline.named_steps["model"].get_params()
        ["n_estimators"]
        == DEFAULT_XGB_REGRESSOR_PARAMS[
            "n_estimators"
        ]
    )
    assert (
        pipeline.named_steps["model"].get_params()
        ["learning_rate"]
        == DEFAULT_XGB_REGRESSOR_PARAMS[
            "learning_rate"
        ]
    )
    assert (
        pipeline.named_steps["model"].get_params()
        ["gamma"]
        == DEFAULT_XGB_REGRESSOR_PARAMS[
            "gamma"
        ]
    )


def test_build_xgboost_pipeline_overrides_params_without_replacing_defaults():
    pipeline = build_xgboost_pipeline(
        numeric_features=["calls_lag_1"],
        model_params={"max_depth": 6},
    )

    params = pipeline.named_steps[
        "model"
    ].get_params()

    assert params["max_depth"] == 6
    assert (
        params["n_estimators"]
        == DEFAULT_XGB_REGRESSOR_PARAMS[
            "n_estimators"
        ]
    )
    assert (
        params["reg_lambda"]
        == DEFAULT_XGB_REGRESSOR_PARAMS[
            "reg_lambda"
        ]
    )


def test_default_xgboost_param_dict_is_not_mutated():
    original = dict(
        DEFAULT_XGB_REGRESSOR_PARAMS
    )

    build_xgboost_pipeline(
        numeric_features=["calls_lag_1"],
        model_params={"max_depth": 6},
    )

    assert (
        DEFAULT_XGB_REGRESSOR_PARAMS
        == original
    )


def test_config_id_is_stable_across_dictionary_order():
    left = {
        "max_depth": 6,
        "learning_rate": 0.05,
    }
    right = {
        "learning_rate": 0.05,
        "max_depth": 6,
    }

    assert make_config_id(
        left
    ) == make_config_id(right)


def test_candidate_generation_always_includes_baseline():
    candidates = generate_parameter_candidates(
        search_mode="random",
        n_iter=1,
        random_search_space={
            "max_depth": [6],
        },
    )

    baseline_ids = [
        candidate["config_id"]
        for candidate in candidates
        if candidate["model_params"]
        == DEFAULT_XGB_REGRESSOR_PARAMS
    ]

    assert len(baseline_ids) == 1


def test_automatic_worker_count_leaves_one_cpu_available():
    assert resolve_worker_count(
        detected_cpu_count=8
    ) == (7, 8)
    assert resolve_worker_count(
        detected_cpu_count=1
    ) == (1, 1)
    assert resolve_worker_count(
        detected_cpu_count=None,
        n_workers=1,
    )[0] == 1


def test_explicit_worker_count_overrides_default():
    assert resolve_worker_count(
        n_workers=3,
        detected_cpu_count=8,
    ) == (3, 8)


def test_invalid_worker_count_is_rejected():
    for n_workers in (0, -1):
        try:
            resolve_worker_count(n_workers=n_workers)
        except ValueError as exc:
            assert "at least 1" in str(exc)
        else:
            raise AssertionError("Expected a ValueError")


def test_tuning_candidates_keep_xgboost_single_threaded():
    params = canonicalize_params({"n_jobs": 8})
    pipeline = build_xgboost_pipeline(
        numeric_features=["calls_lag_1"],
        model_params=params,
    )

    assert params["n_jobs"] == 1
    assert pipeline.named_steps["model"].get_params()["n_jobs"] == 1


def test_evaluate_parameter_configuration_builds_fresh_pipeline_per_fold():
    target_panel, feature_panel, _ = (
        make_small_feature_panel()
    )

    folds = pd.DataFrame(
        {
            "fold": [1, 2],
            "train_start": [
                pd.Timestamp("2023-01-29"),
                pd.Timestamp("2023-01-29"),
            ],
            "train_end": [
                pd.Timestamp("2023-02-15"),
                pd.Timestamp("2023-02-20"),
            ],
            "val_start": [
                pd.Timestamp("2023-02-16"),
                pd.Timestamp("2023-02-21"),
            ],
            "val_end": [
                pd.Timestamp("2023-02-20"),
                pd.Timestamp("2023-02-25"),
            ],
        }
    )

    created_models = []

    class DummyModel:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, X, y):
            self.fit_calls += 1
            return self

        def predict(self, X):
            return np.zeros(len(X))

    def pipeline_builder(
        numeric_features,
        model_params,
    ):
        model = DummyModel()
        created_models.append(model)
        return model

    evaluate_parameter_configuration(
        feature_panel=feature_panel,
        target_panel=target_panel,
        folds=folds,
        feature_set_name="lags_rolling_calendar",
        model_params={},
        pipeline_builder=pipeline_builder,
    )

    assert len(created_models) == 2
    assert len({id(model) for model in created_models}) == 2
    assert all(
        model.fit_calls == 1
        for model in created_models
    )


def test_standard_tuning_folds_end_before_reserved_final_test_period():
    target_panel = make_target_panel()
    folds = build_standard_backtest_folds(
        target_panel
    )

    reserved_test_start = (
        validate_folds_exclude_final_test_period(
            folds=folds,
            target_panel=target_panel,
        )
    )

    assert (
        pd.to_datetime(
            folds["val_end"]
        ).max()
        < reserved_test_start
    )


def test_run_tuning_smoke_test_writes_expected_outputs():
    target_panel, feature_panel, folds = (
        make_small_feature_panel()
    )
    output_dir = (
        Path("tests")
        / "_tmp"
        / f"tuning_run_{uuid.uuid4().hex}"
    )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        result = run_tuning(
            target_panel=target_panel,
            feature_panel=feature_panel,
            folds=folds,
            feature_set_name="lags_rolling_calendar",
            search_mode="random",
            n_iter=0,
            output_dir=output_dir,
            fold_limit=1,
        )

        configuration_results_path = (
            output_dir
            / "configuration_results.csv"
        )
        fold_metrics_path = (
            output_dir
            / "fold_neighborhood_metrics.parquet"
        )
        best_params_path = (
            output_dir / "best_params.json"
        )

        assert configuration_results_path.exists()
        assert fold_metrics_path.exists()
        assert best_params_path.exists()
        assert (
            output_dir / "failures.csv"
        ).exists() is False

        configuration_results = pd.read_csv(
            configuration_results_path
        )
        fold_metrics = pd.read_parquet(
            fold_metrics_path
        )
        best_params = json.loads(
            best_params_path.read_text(
                encoding="utf-8"
            )
        )

        assert not result[
            "configuration_results"
        ].empty
        assert not configuration_results.empty
        assert not fold_metrics.empty
        assert "mean_fold_mase" in configuration_results.columns
        assert "config_id" in fold_metrics.columns
        assert (
            best_params
            == DEFAULT_XGB_REGRESSOR_PARAMS
        )

    finally:
        shutil.rmtree(
            output_dir,
            ignore_errors=True,
        )


def test_parallel_tuning_matches_serial_results_and_candidate_set():
    target_panel, feature_panel, folds = (
        make_small_feature_panel()
    )
    serial_dir = Path("tests") / "_tmp" / f"serial_{uuid.uuid4().hex}"
    parallel_dir = Path("tests") / "_tmp" / f"parallel_{uuid.uuid4().hex}"

    try:
        serial = run_tuning(
            target_panel=target_panel,
            feature_panel=feature_panel,
            folds=folds,
            feature_set_name="lags_rolling_calendar",
            search_mode="random",
            n_iter=1,
            output_dir=serial_dir,
            fold_limit=1,
            n_workers=1,
        )
        parallel = run_tuning(
            target_panel=target_panel,
            feature_panel=feature_panel,
            folds=folds,
            feature_set_name="lags_rolling_calendar",
            search_mode="random",
            n_iter=1,
            output_dir=parallel_dir,
            fold_limit=1,
            n_workers=2,
        )

        assert serial["candidates"] == parallel["candidates"]
        pd.testing.assert_frame_equal(
            serial["configuration_results"].drop(
                columns=["elapsed_seconds"]
            ),
            parallel["configuration_results"].drop(
                columns=["elapsed_seconds"]
            ),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            serial["fold_metrics"],
            parallel["fold_metrics"],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    finally:
        shutil.rmtree(serial_dir, ignore_errors=True)
        shutil.rmtree(parallel_dir, ignore_errors=True)


def test_parallel_checkpointing_is_parent_only_and_output_is_sorted(
    monkeypatch,
):
    target_panel, feature_panel, folds = (
        make_small_feature_panel()
    )
    output_dir = Path("tests") / "_tmp" / f"parallel_{uuid.uuid4().hex}"
    parent_pid = os.getpid()
    writer_pids = []

    from forecasting.tuning import xgboost_regressor

    original_writer = xgboost_regressor.write_checkpoint_artifacts

    def parent_only_writer(*args, **kwargs):
        writer_pids.append(os.getpid())
        assert os.getpid() == parent_pid
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(
        xgboost_regressor,
        "write_checkpoint_artifacts",
        parent_only_writer,
    )

    try:
        run_tuning(
            target_panel=target_panel,
            feature_panel=feature_panel,
            folds=folds,
            feature_set_name="lags_rolling_calendar",
            search_mode="random",
            n_iter=1,
            output_dir=output_dir,
            fold_limit=1,
            n_workers=2,
        )

        saved_metrics = pd.read_parquet(
            output_dir / "fold_neighborhood_metrics.parquet"
        )
        expected_metrics = saved_metrics.sort_values(
            ["config_id", "fold", "neighborhood"]
        ).reset_index(drop=True)

        assert writer_pids
        pd.testing.assert_frame_equal(
            saved_metrics,
            expected_metrics,
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
