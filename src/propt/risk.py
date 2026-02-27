from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(returns.columns).to_numpy()
    pr = returns.to_numpy() @ w
    return pd.Series(pr, index=returns.index, name="portfolio_returns")


def var_cvar(returns: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Historical VaR/CVaR.
    Returns VaR and CVaR as positive numbers (risk).
    """
    x = returns.dropna().to_numpy()
    if x.size == 0:
        raise ValueError("Empty returns")

    q = np.quantile(x, alpha)
    var = -float(q)
    cvar = -float(x[x <= q].mean()) if (x <= q).any() else var
    return var, cvar
