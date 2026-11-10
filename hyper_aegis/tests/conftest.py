import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        tax_ledger_path=tmp_path / "tax_ledger.csv",
        initial_capital=400.0,
        exit_mode="tp_partial",
        trailing_atr_mult=2.0,
        default_stop_atr_mult=1.5,
        risk_basis="initial",
        dashboard_enabled=False,
    )


@pytest.fixture(autouse=True)
def _mkdirs(settings):
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
