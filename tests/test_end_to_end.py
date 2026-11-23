"""Full modelling stage on synthetic data: panel -> ladder -> evaluation -> figures."""
import os
import subprocess
import sys
from pathlib import Path


def test_end_to_end(tmp_path: Path):
    env = {**os.environ, "MTR_ROOT": str(tmp_path)}
    root = Path(__file__).resolve().parents[1]
    code = ("from src import config; from tests import synthetic; synthetic.make(config.PROCESSED)")
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True)
    subprocess.run([sys.executable, "run_models.py"], cwd=root, env=env, check=True,
                   capture_output=True)
    assert (tmp_path / "reports/tables/summary.csv").exists()
    assert len(list((tmp_path / "reports/figures").glob("*.png"))) == 6
