# portfolio-risk-optimizer

Portfolio optimization (Equal / MinVol / MaxSharpe) + VaR/CVaR risk metrics.

## Setup
```bash
cd portfolio-risk-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Run
```bash
python -m propt.cli run \
  --tickers SPY,QQQ,TLT,GLD \
  --start 2018-01-01 --end 2025-12-31 \
  --mode max_sharpe --rf 0.02 --plot
```

## Tests
```bash
pytest -q
```
