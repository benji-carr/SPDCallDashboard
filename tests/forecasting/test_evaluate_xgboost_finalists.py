import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import pytest


def load_evaluator_module():
    module_path = (
        Path("scripts")
        / "forecasting"
        / "evaluate_xgboost_finalists.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_xgboost_finalists",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_data_paths_prefers_tuning_manifest():
    evaluator = load_evaluator_module()
    run_dir = Path("tests") / "_tmp" / f"manifest_{uuid.uuid4().hex}"
    target_path = run_dir / "target.parquet"
    feature_path = run_dir / "features.parquet"
    results_path = run_dir / "configuration_results.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    target_path.touch()
    feature_path.touch()
    results_path.touch()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_target_panel_path": str(target_path),
                "input_feature_panel_path": str(feature_path),
            }
        ),
        encoding="utf-8",
    )

    try:
        target, feature, manifest = evaluator.resolve_data_paths(
            tuning_results_path=results_path,
        )

        assert target == target_path
        assert feature == feature_path
        assert manifest == run_dir / "manifest.json"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_resolve_data_paths_reports_all_attempts_for_missing_target():
    evaluator = load_evaluator_module()
    run_dir = Path("tests") / "_tmp" / f"missing_{uuid.uuid4().hex}"
    results_path = run_dir / "configuration_results.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    results_path.touch()

    try:
        with pytest.raises(FileNotFoundError) as exc_info:
            evaluator.resolve_data_paths(
                tuning_results_path=results_path,
                target_panel_override="missing_target.parquet",
                feature_panel_override="missing_features.parquet",
            )

        message = str(exc_info.value)
        assert "Attempted manifest" in message
        assert "Attempted target-panel paths" in message
        assert "Attempted feature-panel paths" in message
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
