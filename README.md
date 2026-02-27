# Portfolio Risk Optimizer

A lightweight Python CLI for **portfolio optimization** and **risk reporting**.

- Optimization modes: **Equal Weight**, **Minimum Volatility**, **Maximum Sharpe**
- Risk metrics: **Historical VaR / CVaR**
- Optional plots: equity curve + returns histogram

> Educational project — not financial advice.

---

## Features

- Fetches historical **Adjusted Close** prices from Yahoo Finance (via `yfinance`)
- Converts prices to daily returns
- Optimizes long-only weights (0–100%) with fully-invested constraint (weights sum to 1)
- Prints annualized performance stats + daily VaR/CVaR
- Can save plots to `reports/`

---

## Quickstart

### 1) Setup (recommended: virtualenv)

```bash
git clone https://github.com/reallermaker/portfolio-risk-optimizer.git
cd portfolio-risk-optimizer

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
# (optional) if you have dev extras in pyproject:
# pip install -e ".[dev]"
