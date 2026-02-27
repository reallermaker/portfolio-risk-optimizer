from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptResult:
    weights: pd.Series
    exp_return: float
    volatility: float
    sharpe: float


def _portfolio_stats(mu: np.ndarray, cov: np.ndarray, w: np.ndarray, rf: float) -> tuple[float, float, float]:
    exp_ret = float(mu @ w)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (exp_ret - rf) / vol if vol > 0 else np.nan
    return exp_ret, vol, float(sharpe)


def equal_weight(returns: pd.DataFrame, rf: float = 0.0) -> OptResult:
    n = returns.shape[1]
    w = np.ones(n) / n
    mu = returns.mean().to_numpy() * 252.0
    cov = returns.cov().to_numpy() * 252.0
    er, vol, sh = _portfolio_stats(mu, cov, w, rf)
    return OptResult(weights=pd.Series(w, index=returns.columns), exp_return=er, volatility=vol, sharpe=sh)


def min_vol(returns: pd.DataFrame, rf: float = 0.0) -> OptResult:
    n = returns.shape[1]
    mu = returns.mean().to_numpy() * 252.0
    cov = returns.cov().to_numpy() * 252.0

    def obj(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    er, vol, sh = _portfolio_stats(mu, cov, w, rf)
    return OptResult(weights=pd.Series(w, index=returns.columns), exp_return=er, volatility=vol, sharpe=sh)


def max_sharpe(returns: pd.DataFrame, rf: float = 0.0) -> OptResult:
    n = returns.shape[1]
    mu = returns.mean().to_numpy() * 252.0
    cov = returns.cov().to_numpy() * 252.0

    def neg_sharpe(w: np.ndarray) -> float:
        er, vol, sh = _portfolio_stats(mu, cov, w, rf)
        return -sh

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    er, vol, sh = _portfolio_stats(mu, cov, w, rf)
    return OptResult(weights=pd.Series(w, index=returns.columns), exp_return=er, volatility=vol, sharpe=sh)
