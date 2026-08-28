# Seattle Public Safety Dashboards

[View the live dashboard](https://spdcalldashboard.onrender.com)

This application is a collection of dashboards that aim to provide information on the most recent available data on violent, drug-related, and property-related crimes calls to SPD in the City of Seattle as well as the most recent records of crimes released by the City of Seattle. The data is from Seattle's Open Data Portal and can be found at [The Seattle Government Data Website](https://data.seattle.gov). Within this repository you can find the scripts for the dashboard, scripts for pulling data from the Socrata API that SPD uses, and the scripts for automatically updating the rolling snapshot of the last year of available data.  

## Running Forecasting Modules

Forecasting code now lives under the `forecasting` package. Prefer running executable modules with `python -m` from the repository root so imports resolve consistently from the package root.

Examples:

```bash
python -m forecasting.backtests.xgboost --feature-set lags_only
```

```bash
python -m forecasting.evaluation.full_ranking \
    --predictions data/backtest/xgboost/regression/lags_rolling_calendar/predictions.parquet \
    --output-dir data/backtest/xgboost/regression/full_ranking
```

TODO: 
- Deploy Sequential Gradient Boosting model to predict call volume and volume ranking by neighborhood.
- Add descriptive pages for each dashboard

