from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_plots(port_rets: pd.Series, out_dir: str = "reports", title: str = "Portfolio") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    equity = (1.0 + port_rets).cumprod()

    plt.figure()
    equity.plot()
    plt.title(f"{title} - Equity")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    p1 = out / "equity.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()

    plt.figure()
    port_rets.hist(bins=50)
    plt.title(f"{title} - Returns Histogram")
    p2 = out / "returns_hist.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()

    return {"equity": str(p1), "hist": str(p2)}
