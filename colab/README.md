# Headless Cloud Execution: Google Colab & Kaggle

This directory provides automated harnesses to run GPU-accelerated training jobs headlessly on Google Colab or Kaggle while keeping the code and experiment tracking identical to local execution.

---

## 1. Google Colab Instructions

1. **GitHub Access Token:**
   - In Colab, click the **Key icon (Secrets)** on the left sidebar.
   - Add a secret named `GITHUB_TOKEN` with read access to the repository.
   - *(Note: The token is passed directly to the clone command and scrubbed from remote git URLs to prevent credential leakage).*

2. **Mount Google Drive:**
   - Place your packaged dataset (`v1.zip` or `v2.zip`, generated via `python -m cryptograph.pipelines.package_data pack`) in your Google Drive under `CryptoGraph/data/`.
   - Mount Google Drive using Cell 3 in `runner.ipynb`.

3. **Execute Job:**
   - Run cells in `runner.ipynb`.
   - Upon completion, artifacts in `runs/<run_id>/` and `runs/results.csv` are automatically synced to your Google Drive.

---

## 2. Kaggle Differences

1. **Secrets:** Use `from kaggle_secrets import UserSecretsClient; token = UserSecretsClient().get_secret("GITHUB_TOKEN")`.
2. **Dataset:** Upload `v1.zip` / `v2.zip` directly as a private Kaggle Dataset instead of mounting Google Drive.
3. **Paths:** Use `/kaggle/working/CryptoGraph` as the execution root. Output runs can be saved to `/kaggle/working/runs/` and downloaded from the notebook output tab.
