# Local input data

Place the three authorized hourly CSV files here when running the experiments:

```text
data/
  furnace_1.csv
  furnace_2.csv
  furnace_3.csv
```

These industrial data are intentionally **not included** in the public repository.
The first CSV column must contain hourly timestamps. The target-column names used
by the recorded experiment are defined in `src/data_preprocessing.py`.
Si values and their MAE are expressed in wt.%. Do not substitute a differently
constructed hourly target series and expect the published numerical results.

The code imposes an assumed result-release delay on the supplied hourly series.
Raw laboratory sampling and reporting timestamps are not part of these CSVs;
their upstream preparation must be documented separately by the data provider.
