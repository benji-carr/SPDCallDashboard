import json
import subprocess
import sys


def test_production_imports_do_not_eagerly_load_sarima_stack():
    script = """
import importlib.abc
import json
import sys


class BlockPmdarima(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pmdarima" or fullname.startswith("pmdarima."):
            raise ModuleNotFoundError("blocked for regression test")
        return None


sys.meta_path.insert(0, BlockPmdarima())

import forecasting.production.xgboost
import forecasting.production.inference
import forecasting.production.orchestration

print(
    json.dumps(
        {
            "sarima_loaded": "forecasting.backtests.sarima" in sys.modules,
            "pmdarima_loaded": "pmdarima" in sys.modules,
        }
    )
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout.strip())

    assert payload == {
        "sarima_loaded": False,
        "pmdarima_loaded": False,
    }
