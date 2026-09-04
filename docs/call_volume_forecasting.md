# XGBoost Forecasting Architecture

## Production Training

The production XGBoost specification is source-controlled in
`forecasting/production/xgboost.py` and is frozen before the all-history
production refit. Production training uses every eligible feature-panel row,
including the former final holdout; it does not retune or use in-sample fit
metrics as model-quality evidence. The untouched holdout metrics remain the
quality reference recorded with each artifact.

Each run creates an immutable timestamped directory under
`artifacts/models/spd_neighborhood_xgboost/<version>/` containing the
authoritative sklearn `pipeline.joblib`, portable `booster.ubj`, feature
schema, metadata/lineage, training summary, and checksums. Training and daily
inference are intentionally separate stages; production training creates no
`latest` pointer and does not deploy a model.

## Production Inference And Monitoring

Inference requires an explicit immutable model artifact. It builds exactly one
next-day row per fitted neighborhood from observed history through the forecast
origin, then preserves the raw input features, unrounded regression outputs,
and deterministic predicted ranks in an immutable forecast ledger under
`data/forecasts/`. A checksum-validated `latest.parquet` convenience copy is
updated only after the immutable snapshot succeeds.

New training artifacts include `monitoring_baseline.json`: fitted entities,
numeric feature distributions and histogram references, plus fixed per-
neighborhood lag-7 seasonal-naive denominators for future MASE evaluation.
Snapshots record feature-bound observations and prediction distributions for
immediate data/prediction drift investigation. Performance and concept drift
still require actual target values after the forecast date and are intentionally
not inferred at forecast generation time.

## Production Monitoring

Production monitoring is a separate post-inference stage implemented in
`forecasting/production/monitoring.py` with a thin CLI in
`scripts/forecasting/update_xgboost_monitoring.py`. Monitoring requires an
explicit `--artifact-dir`, validates the artifact checksums first, and only
discovers forecast snapshots associated with that artifact run unless the
caller explicitly overrides that scope.

Monitoring does not retrain, retune, promote, or modify any immutable forecast
snapshot. It writes new immutable realized evaluations when actuals mature and
separate immutable monitoring reports for the current artifact state.

## Production Data Refresh

`forecasting/production/data_refresh.py` rebuilds the v1 target panel from
the Seattle Open Data SPD Calls Socrata dataset (`33kz-ixgy`). The target is
still the number of distinct `cad_event_number` values per Seattle-local day
and `dispatch_neighborhood`; repeated source rows do not increase a call
count. The bounded full refresh deliberately retrieves the rolling five-year
source horizon because the existing dashboard snapshot is only one year and
cannot safely supply the frozen v1 panel by itself. The resulting panel is a
complete daily artifact-neighborhood grid, so valid zero-call days are kept.

## Daily Orchestration

`scripts/forecasting/run_xgboost_daily_pipeline.py` is the scheduler-ready
entry point and requires `--artifact-dir`. It validates that artifact, refreshes
and validates the target panel, evaluates real matured forecasts, creates at
most one next-day forecast, then writes one final monitoring report. It never
trains, retunes, or promotes a model.

For Azure Container Apps Jobs that should emit an operational email after each
run, use `scripts/forecasting/run_xgboost_daily_pipeline_with_notifications.py`.
It wraps the same daily pipeline, preserves the true pipeline exit status, and
appends the read-only `show_xgboost_status` report text to success or failure
notifications when email is enabled.

### Production Email Notifications

Notifications use Python's standard-library SMTP client so the transport stays
isolated from forecasting logic and can be injected in tests. All notification
configuration comes from environment variables:

```text
SPD_XGBOOST_NOTIFICATION_ENABLED
SPD_XGBOOST_SMTP_HOST
SPD_XGBOOST_SMTP_PORT
SPD_XGBOOST_SMTP_USERNAME
SPD_XGBOOST_SMTP_PASSWORD
SPD_XGBOOST_EMAIL_FROM
SPD_XGBOOST_EMAIL_TO
```

Optional environment variables:

```text
SPD_XGBOOST_SMTP_STARTTLS
SPD_XGBOOST_SMTP_TIMEOUT_SECONDS
SPD_XGBOOST_EMAIL_SUBJECT_PREFIX
```

If notifications are enabled but required variables are missing, the wrapper
logs a clear notification error. A successful forecast run still exits
successfully even if email delivery fails; a failed forecast run still exits
non-zero even if notification delivery or status generation also fails.

## Data Freshness

The refresh summary records a canonical target-panel SHA-256 and source age.
`--max-source-age-days N` turns that age into an explicit fail-closed SLA;
without it the pipeline records a warning but does not invent a policy. A
failed source refresh stops inference. `--skip-source-refresh` is an explicit
local-input/reproducibility mode only and still validates that input.

## Complete-Day Policy

Daily completeness is evaluated in `America/Los_Angeles`. The current Seattle
calendar day is always excluded; the selected complete-through date is the
earlier of the newest source event date and Seattle yesterday. The panel must
end exactly at that date and match the fitted artifact neighborhood set.

## Missed Production Runs

When runs are missed, existing forecasts are evaluated if their actuals are
available, but dates whose actuals are already known are never backfilled into
the production ledger. They are recorded as operational missed dates and the
pipeline produces only the next unknown day.

## Failure And Retry Semantics

Completed operations runs are immutable under `data/operations/` and have a
deterministic logical identity based on artifact run, complete-through date,
and target-panel fingerprint. Retries of the same state reuse that record and
the immutable forecast snapshot. Failures write separate diagnostic records;
they are never labeled completed. A cross-platform exclusive file lock prevents
concurrent writers and is released in `finally`; stale locks require operator
inspection rather than automatic deletion.

### Data Quality

Data quality checks are kept separate from statistical drift. Monitoring fails
clearly on corrupted artifact or forecast checksums, incompatible model
identity, duplicate forecast keys, unexpected neighborhood sets, or incomplete
ledger directories.

For realized evaluation, each target date must contain one finite actual call
value per expected neighborhood. Legitimate zero-call neighborhoods are
accepted. Incomplete actual cross-sections are marked as not evaluable yet
rather than being silently scored.

### Feature Drift

Immediate feature-drift monitoring uses the immutable
`inference_features.parquet` rows preserved with each forecast and compares
them with the training reference distributions stored in
`monitoring_baseline.json`.

The primary statistical drift features are the target-history predictors:

```text
calls_lag_1
calls_lag_2
calls_lag_3
calls_lag_7
calls_lag_14
calls_lag_21
calls_lag_28
calls_rolling_mean_7
calls_rolling_mean_14
calls_rolling_mean_28
calls_rolling_std_7
calls_rolling_std_28
```

For these features, monitoring records inference counts, missing counts,
summary moments, tail quantiles, range excursions relative to training, and
PSI using the frozen training histogram bins from the artifact baseline.

The calendar covariates

```text
is_weekend
week_of_year_sin
week_of_year_cos
```

are treated differently. They are deterministic functions of `target_date`, so
daily cross-sectional drift alarms would be misleading. Monitoring validates
their correctness, records them, and excludes them from daily statistical drift
classification.

### Prediction Drift

Prediction drift is available immediately from the immutable `forecast.parquet`
 snapshot. For each forecast date, monitoring records the prediction
distribution, top-10 predicted volume, top-10 share of total predicted volume,
and rank-distribution sanity signals.

Monitoring also compares each neighborhood prediction with the frozen training
target statistics stored in `monitoring_baseline.json`. This produces
per-neighborhood z-reference behavior signals such as mean absolute prediction
z and the rate of neighborhoods with `|z| > 2` or `|z| > 3`. These are model
behavior signals, not proof of concept drift.

### Realized Forecast Evaluation

Realized evaluation only occurs after actual target values arrive. A production
forecast is eligible when:

```text
target_date <= latest observed target_date
```

and the actual target panel contains the complete expected neighborhood
cross-section for that date.

Forecasts still awaiting actuals are reported as immature rather than treated
as failures. If the observed target panel ends on `2026-08-12`, then a
forecast targeting `2026-08-13` remains unevaluated on September 2, 2026.

Each realized evaluation writes:

```text
actuals_used.parquet
realized_predictions.parquet
daily_metrics.json
checksums.json
```

The realized prediction table preserves the exact predicted and actual calls,
deterministic ranks, signed and absolute errors, squared errors, and scaled
absolute error based on the fixed lag-7 neighborhood denominators frozen in the
training artifact.

### Performance / Concept Drift

Performance and concept-drift monitoring uses only matured realized
evaluations. Daily realized metrics include MAE, RMSE, bias, mean absolute
bias, MASE, top-10 accuracy, mean correct top 10, top-10 actual volume
capture, and rank correlation.

Rolling monitoring windows of 7, 28, and 90 days summarize the latest current
evaluation for each forecast. Each rolling record explicitly states how many
realized dates are available and whether the requested window is complete.
Short partial histories are therefore not disguised as complete 28-day or
90-day evidence.

The production artifact carries frozen development and final-holdout reference
metrics. Monitoring reports descriptive comparisons such as rolling MASE minus
reference, rolling MASE ratio to reference, top-10 accuracy delta in percentage
points, and rank-correlation delta. These are operational signals only; they
do not automatically trigger retraining or model promotion.

### Revised Actuals

Forecast snapshots are immutable forever, but Seattle SPD actuals may be
revised later. Monitoring therefore fingerprints the exact actual cross-section
used for each target date with a deterministic SHA-256 hash over sorted
`target_date`, `neighborhood`, and `actual_calls`.

An evaluation is identified by:

```text
forecast_id + actuals_sha256
```

If actuals are revised later, the old evaluation remains auditable and a new
immutable evaluation directory is created for the revised fingerprint. Current
rolling performance uses the evaluation whose fingerprint matches the latest
current actual representation.

### Monitoring Windows

Feature drift, prediction drift, and realized performance all expose 7-day,
28-day, and 90-day rolling windows. These windows aggregate immutable
production snapshots or the latest realized evaluations rather than relying on
in-sample training predictions.

An optional mutable `latest/` convenience layer may mirror the newest validated
monitoring summary and key parquet outputs, but the immutable evaluation and
report ledgers remain the audit source of truth.

## Overview:

The XGBoost forecasting implementation provides a machine-learning alternative to the neighborhood-level SARIMA and SARIMAX forecasting models. The goal is to predict next-day Seattle Police Department call volume for each dispatch neighborhood using recent target history, calendar information, and optional external predictors.

The XGBoost model is implemented as a global model rather than fitting a separate regression model for every neighborhood. Each row in the modeling panel represents one target date and one neighborhood. Neighborhood identity is included as a categorical predictor, while lagged call counts, rolling statistics, calendar variables, and optional external variables are included as numeric predictors.

The forecasting task is defined as a sequential one-day-ahead prediction problem. For target date `t`, the model may only use information that would have been available by the end of date `t - 1`.

The overall forecasting workflow is:

```text
Neighborhood-level daily target data
        |
        ▼
Target-panel validation
        |
        ▼
Lag feature construction
        |
        ▼
Rolling feature construction
        |
        ▼
Calendar feature construction
        |
        ▼
Optional external feature merges
        |
        ▼
Date x neighborhood feature panel
        |
        ▼
Rolling-origin development folds
        |
        ▼
scikit-learn preprocessing pipeline
        |
        ▼
XGBRegressor
        |
        ▼
Sequential one-day-ahead validation
        |
        ▼
Neighborhood-level forecast metrics
````

The final 365 days of available target data are reserved as an untouched test period. Model development, feature selection, and hyperparameter experimentation are performed only on data occurring before this final test period.

---

## Target Panel:

The XGBoost forecasting workflow begins with the neighborhood-level daily target panel.

Each row represents the number of calls associated with one neighborhood on one calendar date. The required modeling keys are:

```text
target_date
neighborhood
```

The target column is:

```text
calls
```

The target panel must contain exactly one row for every date and neighborhood combination in the common modeling period.

### prepare_target_panel():

`prepare_target_panel()` validates and normalizes the raw neighborhood-level daily target data before feature construction begins.

The function requires the following columns:

* `target_date`
* `neighborhood`
* `calls`

`target_date` is converted to a normalized pandas datetime, neighborhood names are standardized as strings, and call counts are converted to numeric values.

Rows with missing or invalid neighborhood names are removed. Invalid dates, missing call counts, negative call counts, and duplicate date/neighborhood rows cause the function to fail rather than silently modifying the data.

The panel is returned sorted by neighborhood and target date so later lag and rolling operations operate on a consistent time order.

### validate_daily_panel():

`validate_daily_panel()` verifies that every neighborhood contains a complete daily time series across the common modeling date range.

A complete calendar date range is created using the minimum and maximum target dates in the panel. Each neighborhood is then required to contain exactly that same set of dates.

This validation is important because lag and rolling features assume that shifting a row by one position corresponds to shifting by exactly one calendar day. Missing dates could otherwise cause a feature such as `calls_lag_7` to represent something other than seven days before the target.

---

## XGBoost Feature Construction:

The XGBoost model converts the forecasting problem into supervised learning by representing past target history and known calendar information as predictors.

The base feature panel is constructed in `forecasting/features/xgboost.py`.

The current target-history lags are:

```text
1 day
2 days
3 days
7 days
14 days
21 days
28 days
```

The current rolling mean windows are:

```text
7 days
14 days
28 days
```

Rolling standard deviations are calculated for:

```text
7 days
28 days
```

The base calendar variables are:

```text
is_weekend
week_of_year_sin
week_of_year_cos
```

### add_lag_features():

`add_lag_features()` creates direct autoregressive predictors from previous call-volume observations.

For target date `t`, a lag feature is defined as:

```text
calls_lag_k = calls observed at t - k
```

For example:

```text
calls_lag_1  = yesterday's call count
calls_lag_7  = call count on the same day one week earlier
calls_lag_28 = call count four weeks earlier
```

Lag features are calculated independently within each neighborhood using `groupby()` followed by `shift()`.

Because `shift()` moves only earlier observations forward, the target value for date `t` can never appear inside a lag feature used to predict date `t`.

### add_rolling_features():

`add_rolling_features()` creates local summaries of recent call-volume behavior.

The call series is shifted by one day before any rolling calculation is performed:

```text
calls
   |
   ▼
shift(1)
   |
   ▼
rolling mean / rolling standard deviation
```

This ordering is an important leakage-prevention rule.

For example, the 7-day rolling mean used for target date `t` contains call counts from approximately:

```text
t - 1
through
t - 7
```

but never includes the target value from date `t`.

If the rolling window were calculated directly from the unshifted target series, the target day's own call count would be included in the feature and the model would have direct access to the value it is supposed to predict.

### add_calendar_features():

`add_calendar_features()` merges calendar variables onto the date/neighborhood feature panel.

Calendar variables do not depend on observed call volume and are therefore known before the forecast date occurs.

The current XGBoost calendar feature set includes:

```text
is_weekend
week_of_year_sin
week_of_year_cos
```

The sine and cosine terms provide a cyclical representation of position within the year without creating an artificial discontinuity between the end and beginning of the calendar year.

### build_xgboost_feature_panel():

`build_xgboost_feature_panel()` coordinates the complete construction of the base XGBoost supervised-learning dataset.

The function performs the following sequence:

```text
Prepare target panel
        |
        ▼
Validate complete daily coverage
        |
        ▼
Add lag features
        |
        ▼
Add rolling features
        |
        ▼
Add calendar features
        |
        ▼
Drop rows without sufficient target history
        |
        ▼
Sort by target date and neighborhood
```

Rows at the beginning of each neighborhood series naturally contain missing lag and rolling values because sufficient historical observations do not yet exist.

These rows are removed only after all target-history features have been created.

### validate_xgboost_feature_panel():

`validate_xgboost_feature_panel()` verifies the finished supervised-learning panel before it is passed into a backtest.

The function confirms that:

* required modeling keys exist,
* the target column exists,
* all required predictors exist,
* the panel is not empty,
* target-date/neighborhood keys are unique,
* predictor values are not missing,
* predictor values are finite.

The function fails early if the modeling panel violates any of these assumptions.

---

## XGBoost Feature Sets:

Predefined XGBoost feature sets allow experiments to add groups of predictors incrementally while keeping the rest of the model configuration unchanged.

The current base feature specifications include:

### `lags_only`:

Contains only lagged call-volume predictors.

```text
calls_lag_1
calls_lag_2
calls_lag_3
calls_lag_7
calls_lag_14
calls_lag_21
calls_lag_28
```

### `lags_rolling`:

Contains all lagged call-volume predictors plus rolling means and rolling standard deviations.

### `lags_rolling_calendar`:

Contains target-history predictors plus the calendar predictors:

```text
is_weekend
week_of_year_sin
week_of_year_cos
```

These feature sets allow the incremental forecasting value of rolling statistics and calendar information to be evaluated using the same model and validation folds.

---

## External Feature Merging:

Additional explanatory variables can be added to the existing XGBoost feature panel using `merge_external_features()`.

External feature panels must use the same modeling keys:

```text
target_date
neighborhood
```

The merge is intentionally strict. Both the existing feature panel and external feature panel are checked for missing keys and duplicate date/neighborhood combinations before the merge occurs.

The merge uses a one-to-one relationship and verifies that:

* the row count does not change,
* the original modeling keys remain unchanged,
* the requested external predictors exist,
* the external variables contain complete coverage over the experiment period.

This prevents a feature experiment from accidentally dropping observations, duplicating observations, or introducing missing predictors.

External variables are therefore added to the same base forecasting panel rather than creating a separate modeling pipeline.

---

## Rolling-Origin Backtest Design:

The XGBoost model is evaluated using rolling-origin time-series validation rather than random train/test splitting.

Ordinary shuffled cross-validation is inappropriate for this forecasting problem because observations occurring in the future could be used to train a model evaluated on earlier dates.

Instead, each validation fold follows chronological order:

```text
Past observations
        |
        ▼
Model training
        |
        ▼
Future observations
        |
        ▼
Validation
```

The final 365 days of the target panel are excluded from model development and remain untouched until final model selection is complete.

### build_standard_backtest_folds():

`build_standard_backtest_folds()` constructs the standard development folds.

The default configuration uses:

```text
Final test period:      365 days
Validation fold length:  90 days
Number of folds:           4
```

Each fold uses an expanding training window.

The first training window begins at the earliest available development date. Later folds retain that same training start date but extend the training end date forward.

The structure is approximately:

```text
Fold 1:
[---------- Training ----------][ Validation 1 ]

Fold 2:
[--------------- Training ---------------][ Validation 2 ]

Fold 3:
[-------------------- Training --------------------][ Validation 3 ]

Fold 4:
[------------------------- Training -------------------------][ Validation 4 ]

                                                               [ Final Test ]
```

This design evaluates the model at several historical forecast origins.

Later folds contain more training data because that is what would happen naturally if the forecasting system were operating over time and accumulating additional observations.

### validate_folds():

`validate_folds()` checks the fold table before model evaluation begins.

The function verifies that:

* all required fold columns exist,
* the fold table is not empty,
* all fold dates are valid,
* fold numbers are unique,
* the training period ends before validation begins,
* validation start dates do not occur after validation end dates.

Any overlap between training and validation causes the backtest to fail.

---

## Leakage Prevention:

Preventing information leakage is one of the main design requirements of the XGBoost forecasting pipeline.

Leakage occurs when information that would not have been available at the forecast origin influences model training or predictor construction.

The current implementation prevents several forms of leakage.

### Training and Validation Separation:

For every fold:

```text
train_end < val_start
```

The target values belonging to the validation period are therefore never included in the dataset passed to `model.fit()`.

The model only learns its tree structure and parameters from observations occurring before the validation period.

### Lag Feature Construction:

All lagged target variables use `shift()` within neighborhood.

For target date `t`:

```text
calls_lag_1 = calls[t - 1]
calls_lag_7 = calls[t - 7]
```

No future value can move backward into an earlier target row.

### Rolling Feature Construction:

Rolling variables shift the target by one observation before applying the rolling calculation.

This means:

```text
Correct:

calls
    |
    ▼
shift(1)
    |
    ▼
rolling(7)
```

rather than:

```text
Incorrect:

calls
    |
    ▼
rolling(7)
```

The incorrect version would include the target day's actual value in its own predictor.

### Pipeline Fitting:

Neighborhood encoding and XGBoost fitting are contained inside the same scikit-learn `Pipeline`.

The pipeline is fitted only on each fold's training observations.

As a result, preprocessing is not fitted against validation rows before prediction.

### Fold Independence:

`run_feature_set_backtest()` creates a new XGBoost pipeline for every validation fold.

Fitted model state from an earlier fold is therefore never reused by another fold.

Each fold represents a separate historical deployment experiment.

### Sequential Validation:

Within a validation fold, earlier actual observations are allowed to appear in lag and rolling features used for later validation dates.

For example:

```text
Forecast June 1 using data through May 31
        |
        ▼
June 1 actual becomes known
        |
        ▼
Forecast June 2 using data through June 1
```

This is not leakage because June 1 would genuinely be known before a June 2 forecast is generated.

Likewise, when forecasting June 8:

```text
calls_lag_7 = actual calls from June 1
```

is valid because June 1 has already occurred.

The validation process therefore simulates daily operation rather than producing the entire 90-day validation period from a single forecast origin.

---

## One-Day-Ahead Forecasting Convention:

The current XGBoost model is specifically evaluated as a one-day-ahead forecasting system.

For each prediction:

```text
forecast_origin = target_date - 1 day
```

The model answers the question:

> Given everything known through yesterday, how many calls should be expected in each neighborhood today?

The validation folds should therefore not be interpreted as 90-day-ahead forecasts.

A 90-day fold contains approximately 90 independent one-day-ahead forecast origins:

```text
Day 1:
information through t-1 -> predict t

Day 2:
information through t -> predict t+1

Day 3:
information through t+1 -> predict t+2

...
```

This matches the intended production use case where call-volume forecasts can be refreshed every day after the newest observed data become available.

---

## Fold Data:

### get_fold_data():

`get_fold_data()` converts one fold definition into the training and validation datasets used by the model.

The function first resolves the numeric predictors requested by the experiment.

It then converts the fold date boundaries into normalized timestamps and filters the feature panel into:

```text
training observations
validation observations
```

The function verifies that neither dataset is empty.

It also constructs the expected validation calendar and verifies that every date within the validation interval is represented in the panel.

The requested numeric predictors are checked against the available feature-panel columns.

The final model feature list is:

```text
neighborhood
+
selected numeric predictors
```

The function returns the training and validation frames along with the model features, numeric features, and fold date boundaries.

### resolve_numeric_features():

`resolve_numeric_features()` determines which numeric predictors should be included in a model run.

The features can be supplied either through:

* a predefined feature-set name,
* or an explicit list of numeric predictors.

This allows the standard backtest infrastructure to be reused for external-feature experiments without modifying the original `XGB_FEATURE_SETS` dictionary for every experiment.

---

## XGBoost Pipeline:

### build_xgboost_pipeline():

`build_xgboost_pipeline()` creates the scikit-learn preprocessing and regression pipeline used by each fold.

The pipeline contains two major stages:

```text
Input modeling rows
        |
        ▼
ColumnTransformer
        |
        ├── Neighborhood
        |       |
        |       ▼
        |   OneHotEncoder
        |
        └── Numeric predictors
                |
                ▼
            Passthrough
        |
        ▼
XGBRegressor
        |
        ▼
Predicted call volume
```

### Neighborhood Encoding:

Neighborhood is treated as a categorical predictor.

`OneHotEncoder` converts each neighborhood into binary indicator columns that can be consumed by the tree model.

The encoder uses:

```text
handle_unknown="ignore"
```

so an unexpected category does not cause the prediction pipeline to fail.

### Numeric Predictors:

Lag, rolling, calendar, and external numeric variables are passed directly through the `ColumnTransformer`.

No standardization is required because tree-based models do not depend on predictor scale in the same way as linear or distance-based methods.

### XGBRegressor:

The current XGBoost regressor uses a squared-error regression objective.

The configuration uses:

* 500 boosting trees,
* a low learning rate,
* shallow tree depth,
* minimum child-weight regularization,
* row subsampling,
* column subsampling,
* L2 regularization,
* fixed random seed.

The model currently runs with one XGBoost worker thread per fit so experiment-level parallelism can be controlled externally if needed.

---

## Global Neighborhood Model:

The current implementation fits one global XGBoost model rather than separate models for every neighborhood.

The training matrix contains rows similar to:

```text
Date        Neighborhood       lag_1   lag_7   rolling_28   calendar...
---------------------------------------------------------------------
2025-01-01  Ballard              ...     ...        ...        ...
2025-01-01  Downtown             ...     ...        ...        ...
2025-01-01  Capitol Hill         ...     ...        ...        ...
2025-01-02  Ballard              ...     ...        ...        ...
2025-01-02  Downtown             ...     ...        ...        ...
```

Neighborhood identity is one-hot encoded while historical and calendar features describe the state of each individual neighborhood series.

This allows XGBoost to learn relationships shared across neighborhoods while still allowing neighborhood identity to influence the forecast.

A global model also gives lower-volume neighborhoods access to patterns learned from the much larger pooled training dataset.

---

## Single-Fold Evaluation:

### run_sequential_fold():

`run_sequential_fold()` performs the complete evaluation of one rolling-origin validation fold.

The function first calls `get_fold_data()` to retrieve the fold-specific training and validation observations and determine which predictors are used by the model.

The fold is then executed in several stages.

### Model Fitting:

The training features are extracted using:

```text
X_train =
    neighborhood
    + numeric predictors
```

The target is:

```text
y_train = calls
```

The entire training panel is passed to:

```text
model.fit(X_train, y_train)
```

The fitted model therefore uses historical observations from all neighborhoods.

The XGBoost model is fitted once at the beginning of the validation fold.

It is not retrained after every validation date.

### Sequential Prediction:

The validation observations are sorted by target date.

For each date, all neighborhood rows corresponding to that target date are selected and passed into the fitted model.

The daily loop therefore behaves as:

```text
Select target date
        |
        ▼
Collect all neighborhood feature rows
        |
        ▼
Predict all neighborhoods
        |
        ▼
Store predictions
        |
        ▼
Advance to next target date
```

Although the fitted XGBoost model remains fixed during the fold, the lagged and rolling predictor values change from one forecast origin to the next.

Later validation dates may contain actual target observations from earlier validation dates because those observations would have become available by that point in time.

### Prediction Output:

For every neighborhood and target date, the function records:

```text
fold
feature_set
target_date
forecast_origin
neighborhood
actual
prediction
```

`forecast_origin` is always set to the previous day.

This makes the forecasting horizon explicit in the saved prediction data.

### Prediction Validation:

After every target date has been processed, the daily prediction frames are combined into one fold-level prediction table.

The function verifies that:

* at least one prediction was generated,
* prediction count equals validation row count,
* every target-date/neighborhood combination is unique,
* every model prediction is finite.

These checks prevent invalid model output from silently being passed into the metric calculations.

---

## Forecast Metrics:

After validation predictions are created, metrics are calculated separately for each neighborhood.

The common forecast evaluation function is shared with the SARIMA/SARIMAX backtesting implementation.

Current metrics include:

```text
MAE
RMSE
MASE
sMAPE
Bias
```

### MAE:

Mean Absolute Error measures the average absolute prediction error in call-count units.

Lower values are better.

### RMSE:

Root Mean Squared Error penalizes large forecast misses more strongly than MAE.

Lower values are better.

### MASE:

Mean Absolute Scaled Error compares model forecast error against a seasonal-naive benchmark.

The current seasonality is:

```text
7 days
```

A MASE below one indicates that the model performs better than the corresponding weekly seasonal-naive forecast on average.

```text
MASE < 1:
    better than seasonal naive

MASE = 1:
    approximately equal to seasonal naive

MASE > 1:
    worse than seasonal naive
```

### sMAPE:

Symmetric Mean Absolute Percentage Error expresses forecast error relative to the magnitude of the actual and predicted values.

Lower values are better.

### Bias:

Bias measures systematic overprediction or underprediction.

The shared forecast evaluator uses:

```text
prediction - actual
```

Therefore:

```text
positive bias:
    average overprediction

negative bias:
    average underprediction
```

---

## MASE Training History:

`run_sequential_fold()` intentionally retrieves the raw target-panel training history separately from the model-ready feature panel.

The feature panel loses early observations because lag and rolling features require historical context.

For example, a target row cannot have a valid 28-day lag until sufficient previous observations exist.

However, the MASE denominator should be calculated using the complete available target history rather than only the shortened model-ready feature panel.

The function therefore maintains two related training datasets:

```text
Feature-panel training rows
    -> used to fit XGBoost

Raw target-panel training rows
    -> used to calculate the MASE scale
```

This preserves the complete seasonal-naive training history used for forecast evaluation.

---

## Neighborhood-Level Evaluation:

Metrics are calculated independently for every neighborhood within every fold.

The resulting metric panel therefore contains rows conceptually similar to:

```text
Fold    Neighborhood      MAE    RMSE    MASE    sMAPE    Bias
----------------------------------------------------------------
1       Ballard           ...
1       Downtown          ...
1       Capitol Hill      ...
2       Ballard           ...
2       Downtown          ...
```

Neighborhood-level metrics make it possible to determine whether a model or feature set improves forecasting broadly or only for a small number of high-volume neighborhoods.

They are also useful when evaluating external predictors because feature effectiveness may vary geographically.

---

## Fold Diagnostics:

### summarize_fold_predictions():

`summarize_fold_predictions()` calculates general diagnostics for all predictions produced during one fold.

The current diagnostics include:

```text
number of predictions
minimum prediction
maximum prediction
negative prediction rate
mean residual
residual standard deviation
```

The residual used by this diagnostic function is:

```text
actual - prediction
```

This is the opposite sign convention from the shared forecast bias metric.

For this reason the residual diagnostic is explicitly named:

```text
residual_mean_actual_minus_pred
```

so the interpretation remains clear in saved results.

### Diagnostic Output:

`run_sequential_fold()` adds fold metadata to the diagnostics, including:

```text
fold
feature_set
train_start
train_end
val_start
val_end
n_training_rows
```

This makes each diagnostic record traceable to its exact historical experiment window.

---

## Multi-Fold Evaluation:

### run_feature_set_backtest():

`run_feature_set_backtest()` coordinates the complete rolling-origin backtest for one feature specification.

The function first resolves the numeric predictors associated with the feature set.

The fold table is sorted by fold number and processed sequentially.

For every fold:

```text
Create new XGBoost Pipeline
        |
        ▼
Pass fold definition to run_sequential_fold()
        |
        ▼
Collect predictions
        |
        ▼
Collect neighborhood metrics
        |
        ▼
Collect fold diagnostics
```

A completely new pipeline is created for each fold.

This ensures that:

* the neighborhood encoder is fitted only on the current fold,
* XGBoost parameters are learned only from the current fold,
* no fitted state carries forward from another validation experiment.

At the end of the function, all fold-level results are concatenated.

The returned dictionary contains:

```text
predictions
metrics
diagnostics
```

This standardized output format is reused by both base XGBoost experiments and external-feature experiments.

---

## Why run_sequential_fold() and run_feature_set_backtest() Are Separate:

The two functions intentionally operate at different levels of the backtesting process.

`run_sequential_fold()` answers:

> How should one historical validation fold be executed correctly?

Its responsibilities include:

* fitting the model,
* producing sequential predictions,
* validating prediction output,
* calculating neighborhood metrics,
* generating diagnostics.

`run_feature_set_backtest()` answers:

> How should the same forecasting procedure be repeated across every development fold?

Its responsibilities include:

* creating a fresh pipeline,
* calling the single-fold runner,
* collecting fold outputs,
* concatenating final experiment results.

Separating these responsibilities prevents single-fold forecasting logic from being duplicated across different experiments.

---

## Feature-Set Experiment Design:

The XGBoost backtesting infrastructure is designed so feature experiments can reuse the same validation procedure.

A feature experiment changes the available predictors while keeping the following fixed:

```text
forecasting horizon
fold dates
target observations
neighborhoods
XGBoost architecture
metric definitions
sequential validation process
```

This allows performance changes to be attributed more directly to the features being added or removed.

For example:

```text
Experiment 1:
lags only

Experiment 2:
lags + rolling statistics

Experiment 3:
lags + rolling statistics + calendar

Experiment 4:
baseline features + external event variables
```

All experiments can be evaluated using the exact same folds.

---

## External Feature Experiments:

External-feature experiments extend the same base XGBoost architecture rather than defining a separate forecasting system.

The permitted-event experiment is one example.

A baseline specification contains the existing lag, rolling, and calendar predictors.

Candidate models append one or more permitted-event variables to the baseline feature list.

Example candidate structures include:

```text
Baseline:
    lag + rolling + calendar features

Candidate:
    baseline
    + permitted event count

Candidate:
    baseline
    + estimated attendance

Candidate:
    baseline
    + multiple permitted-event variables
```

Each candidate is passed through `run_feature_set_backtest()` using the same folds.

This produces paired fold/neighborhood results that can be compared directly with the baseline model.

---

## Paired Candidate Comparison:

External-feature experiments compare candidate metrics to baseline metrics using the same:

```text
fold
neighborhood
```

keys.

This paired design ensures that the candidate and baseline are evaluated on identical observations.

For each metric, the experiment can calculate:

```text
candidate metric - baseline metric
```

For MASE:

```text
negative delta:
    candidate improved performance

positive delta:
    candidate reduced performance
```

The comparison can then summarize:

* mean change in MASE,
* median change in MASE,
* percentage of fold/neighborhood jobs improved,
* percentage of neighborhoods improved on average,
* changes in sMAPE and other metrics.

This is more informative than relying only on one global average because it shows how consistently an external feature improves forecasting.

---

## Backtest Output Files:

The XGBoost backtesting utilities save three primary result files for each experiment:

```text
predictions.parquet
metrics.parquet
diagnostics.parquet
```

### predictions.parquet:

Contains one row for every validation prediction.

Important fields include:

```text
fold
feature_set
target_date
forecast_origin
neighborhood
actual
prediction
```

### metrics.parquet:

Contains neighborhood-level forecast metrics for every fold.

Important fields include:

```text
fold
feature_set
neighborhood
n_model_train
n_mase_train
n_validation
mae
rmse
mase
smape
bias
```

### diagnostics.parquet:

Contains fold-level diagnostic information.

Important fields include:

```text
fold
feature_set
train_start
train_end
val_start
val_end
n_training_rows
prediction diagnostics
```

Keeping predictions, metrics, and diagnostics separate makes it possible to perform later analysis without rerunning the full XGBoost backtest.

---

## XGBoost Forecasting Architecture Summary:

The XGBoost forecasting layer follows several important design principles.

First, all forecasting experiments preserve chronological order. Future observations are never included in the model fitting process used to predict earlier dates.

Second, target-history features are constructed so every lag and rolling predictor represents information that would actually have been known before the target date.

Third, each fold receives a fresh scikit-learn pipeline so preprocessing and model state remain isolated between historical validation experiments.

Fourth, validation proceeds sequentially one day at a time. This allows newly observed actual call counts to become legitimate historical predictors for later forecasts while preserving the one-day-ahead forecasting contract.

Finally, feature experiments reuse the same folds, model architecture, and evaluation functions. This makes it possible to compare lag, calendar, event, weather, and other external predictors using a consistent backtesting framework.

---

## XGBoost Forecasting File Reference:

The following is a basic reference for where changes to different parts of the XGBoost forecasting system should be made:

```text
Change target-panel validation:
    forecasting/features/xgboost.py
        prepare_target_panel()
        validate_daily_panel()

Change lag definitions:
    forecasting/features/xgboost.py
        LAG_DAYS
        add_lag_features()

Change rolling feature definitions:
    forecasting/features/xgboost.py
        ROLLING_WINDOWS
        add_rolling_features()

Change XGBoost calendar features:
    forecasting/features/xgboost.py
        CALENDAR_XGB_FEATURES
        add_calendar_features()

Change predefined feature sets:
    forecasting/features/xgboost.py
        XGB_FEATURE_SETS

Change base feature-panel construction:
    forecasting/features/xgboost.py
        build_xgboost_feature_panel()

Change feature-panel validation:
    forecasting/features/xgboost.py
        validate_xgboost_feature_panel()

Change external feature merging:
    forecasting/features/xgboost.py
        merge_external_features()

Change standard development folds:
    forecasting/backtests/xgboost.py
        build_standard_backtest_folds()

Change fold validation:
    forecasting/backtests/xgboost.py
        validate_folds()

Change fold data selection:
    forecasting/backtests/xgboost.py
        get_fold_data()

Change feature resolution:
    forecasting/backtests/xgboost.py
        resolve_numeric_features()

Change neighborhood preprocessing:
    forecasting/backtests/xgboost.py
        build_xgboost_pipeline()

Change XGBRegressor hyperparameters:
    forecasting/backtests/xgboost.py
        build_xgboost_pipeline()

Change fold-level prediction diagnostics:
    forecasting/backtests/xgboost.py
        summarize_fold_predictions()

Change sequential one-day-ahead validation:
    forecasting/backtests/xgboost.py
        run_sequential_fold()

Change multi-fold backtest behavior:
    forecasting/backtests/xgboost.py
        run_feature_set_backtest()

Change aggregate XGBoost metric summaries:
    forecasting/backtests/xgboost.py
        summarize_metrics()

Change XGBoost backtest file output:
    forecasting/backtests/xgboost.py
        save_backtest_outputs()

Change shared forecast metric definitions:
    forecasting/backtests/sarima.py
        evaluate_forecast()

Change permitted-event XGBoost experiments:
    forecasting/backtests/xgboost_permitted_events.py
```

```

The feature definitions and leakage-safe `shift()`-before-rolling construction are implemented in the current `forecasting/features/xgboost.py`, while the fold execution, per-neighborhood metrics, and fresh-pipeline-per-fold logic are implemented in `forecasting/backtests/xgboost.py`.   

## Production Status Command

Use the read-only operator report for one explicit production artifact:

```powershell
python -m scripts.forecasting.show_xgboost_status --artifact-dir artifacts/models/spd_neighborhood_xgboost/v1/<run_id>
```

It only reads existing artifacts, forecasts, monitoring outputs, and operations records; it does not refresh data, generate forecasts, run monitoring, train, or update pointers. `INCOMPLETE` rolling metrics mean the existing monitoring window does not yet contain its required number of production dates.
```
