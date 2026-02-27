import pandas as pd
import numpy as np

from propt.optimizer import equal_weight, min_vol, max_sharpe


def _fake_returns():
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, size=(len(idx), 4))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


def test_equal_weight_sums_to_1():
    rets = _fake_returns()
    res = equal_weight(rets)
    assert abs(res.weights.sum() - 1.0) < 1e-9


def test_min_vol_sums_to_1():
    rets = _fake_returns()
    res = min_vol(rets)
    assert abs(res.weights.sum() - 1.0) < 1e-6


def test_max_sharpe_sums_to_1():
    rets = _fake_returns()
    res = max_sharpe(rets)
    assert abs(res.weights.sum() - 1.0) < 1e-6
