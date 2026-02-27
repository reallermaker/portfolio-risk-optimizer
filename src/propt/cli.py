from __future__ import annotations

import argparse

from .data import load_returns
from .optimizer import equal_weight, min_vol, max_sharpe
from .risk import portfolio_returns, var_cvar
from .report import save_plots


def main() -> None:
    p = argparse.ArgumentParser(prog="propt", description="Portfolio Optimizer CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--tickers", required=True, help="Comma separated tickers (e.g., SPY,QQQ,TLT,GLD)")
    run.add_argument("--start", required=True)
    run.add_argument("--end", default=None)
    run.add_argument("--mode", choices=["equal", "min_vol", "max_sharpe"], default="max_sharpe")
    run.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate, e.g. 0.02")
    run.add_argument("--alpha", type=float, default=0.05, help="VaR/CVaR alpha")
    run.add_argument("--plot", action="store_true")

    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    data = load_returns(tickers, start=args.start, end=args.end)

    if args.mode == "equal":
        opt = equal_weight(data.returns, rf=args.rf)
    elif args.mode == "min_vol":
        opt = min_vol(data.returns, rf=args.rf)
    else:
        opt = max_sharpe(data.returns, rf=args.rf)

    pr = portfolio_returns(data.returns, opt.weights)
    var, cvar = var_cvar(pr, alpha=args.alpha)

    print("=== Optimization Result ===")
    print(f"Mode: {args.mode}")
    print("\nWeights:")
    print((opt.weights * 100).round(2).astype(str) + "%")
    print(f"\nExp Return (ann): {opt.exp_return:.2%}")
    print(f"Volatility (ann): {opt.volatility:.2%}")
    print(f"Sharpe:           {opt.sharpe:.3f}")
    print(f"VaR({args.alpha:.0%}) daily:  {var:.2%}")
    print(f"CVaR({args.alpha:.0%}) daily: {cvar:.2%}")

    if args.plot:
        paths = save_plots(pr, title=f"Portfolio ({args.mode})")
        print("\nSaved plots:")
        for k, v in paths.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
