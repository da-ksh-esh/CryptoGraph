# Experiment configs

One YAML file per experiment, run with:

```
python -m cryptograph.pipelines.train --config configs/experiment/<name>.yaml --seed <n>
```

An experiment names the data config (`configs/data/daily.yaml` or `hourly.yaml`), the dataset version,
the model config (`configs/model/*.yaml`), the target and loss, and the seeds to run.
Runs write to `runs/<run_id>/` (gitignored) and append one row to `runs/results.csv`.
