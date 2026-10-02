# Experiment configs

One YAML file per experiment, run with:

```
python -m cryptograph.pipelines.train --config configs/experiment/<name>.yaml --seed <n>
```

An experiment names the data config (`configs/data/daily.yaml` or `hourly.yaml`), the dataset version,
the model config (`configs/model/*.yaml`), the target and loss, and the seeds to run.
Runs write to `runs/<run_id>/` (gitignored) and append one row to `runs/results.csv`.

`smoke_lstm_daily.yaml` is the Phase C smoke test (tiny LSTM, 1 epoch): it proves the run-tracking flow and is
not a result. Only the LSTM task is wired up so far; other models exit with code 2 until the training phase.

Fields: `name`, `seed`, `device` (auto | cpu | cuda), `deterministic`, `data` (`version`, `resolution` null for
daily or "1h"/"2h"), `model` (`name`, `config`, `overrides`), `target`, `loss`, `training`
(epochs, batch_size, learning_rate, weight_decay, patience, optional max_train_batches / max_eval_samples).
