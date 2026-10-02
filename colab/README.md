# Running on Colab / Kaggle

The same code runs locally and on a cloud GPU. The notebook `runner.ipynb` only clones the repo, installs it,
unpacks the data and calls the training command; all logic stays in `src/cryptograph/`.

## Flow

1. **Locally:** build the data, then pack it with a hash manifest:
   ```
   python -m cryptograph.pipelines.package_data pack --version v2      # -> dist/cryptograph_data_v2.zip (~235 MB)
   ```
   Upload the zip to a private Google Drive folder (default `MyDrive/CryptoGraph/`). Data never goes into git.
2. **Colab:** open `colab/runner.ipynb`, set the GPU runtime, run the five cells:
   clone → `pip install -e .` → mount Drive + `unpack` (every SHA-256 verified; a mismatch stops the run) →
   `python -m cryptograph.pipelines.train --config configs/experiment/<name>.yaml --seed <n> --device cuda` →
   copy `runs/<run_id>/` and `results.csv` to Drive.
3. Each run folder holds `config.resolved.yaml`, `env.json` (library versions, GPU, git commit, dirty flag),
   `log.txt`, `metrics.json` and `checkpoint.pt`. Results from several machines are merged from the copied
   `results_*.csv` files.

## Access token (decision pending)

The repo is private-to-be, so cloning needs a **read-only** token stored as a secret named `GITHUB_TOKEN`
(never in the notebook, never printed; the runner removes it from the remote URL after cloning). Two options:

- **Organisation repo:** move the repo into a GitHub organisation; each member creates a fine-grained token
  with read access to that repo only.
- **Owner token:** the repo owner (Dakshesh) creates a fine-grained, read-only token for this repo and shares it
  privately with the team. Fine-grained tokens created by other accounts cannot reach a repo owned by a
  personal account.

Neither is assumed; the runner only reads the secret `GITHUB_TOKEN`. The branch must exist on the remote, and
nothing is pushed until the repo is confirmed private.

## Kaggle differences

| | Colab | Kaggle |
|---|---|---|
| Secret | `google.colab.userdata.get("GITHUB_TOKEN")` | `from kaggle_secrets import UserSecretsClient; UserSecretsClient().get_secret("GITHUB_TOKEN")` |
| Data | zip on Google Drive, `drive.mount(...)` | upload the zip as a **private** Kaggle dataset; it appears read-only under `/kaggle/input/<dataset>/` |
| Working dir | `/content/CryptoGraph` | `/kaggle/working/CryptoGraph` (only `/kaggle/working` is writable and kept as output) |
| Unpack | `unpack /content/drive/.../cryptograph_data_v2.zip` | `unpack /kaggle/input/<dataset>/cryptograph_data_v2.zip --dest /kaggle/working/CryptoGraph` |
| Saving runs | copy `runs/` to Drive (cell 5) | leave `runs/` in `/kaggle/working`; it is saved with the notebook version's output |
| Internet | on by default | enable *Internet* in the notebook settings for the clone and `pip install` |
| GPU | Runtime → GPU | Settings → Accelerator → GPU |
