"""Executes and verifies all research notebooks top-to-bottom."""

import sys
import time
import asyncio
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


NOTEBOOKS = [
    "notebooks/01_data_preprocessing_and_eda.ipynb",
    "notebooks/02_network_analysis.ipynb",
    "notebooks/03_models_vs_baselines.ipynb",
    "notebooks/04_tempme_explanations.ipynb",
    "notebooks/05_persistence_by_shock_type.ipynb",
    "notebooks/06_robustness_tgae.ipynb",
]


def execute_notebook(nb_path_str: str) -> dict:
    nb_path = Path(nb_path_str)
    print(f"\n==================================================")
    print(f"Executing: {nb_path.name}")
    print(f"==================================================")

    t0 = time.time()
    nb = nbformat.read(nb_path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent.resolve())}}
    )

    try:
        client.execute()
        elapsed = time.time() - t0
        print(f"SUCCESS: {nb_path.name} completed in {elapsed:.2f}s")
        # Save executed notebook with outputs
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return {
            "notebook": nb_path.name,
            "status": "PASSED",
            "elapsed_seconds": round(elapsed, 2),
            "error": None
        }
    except CellExecutionError as e:
        elapsed = time.time() - t0
        print(f"FAILED: {nb_path.name} in cell execution ({elapsed:.2f}s): {e}")
        return {
            "notebook": nb_path.name,
            "status": "FAILED",
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e)
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {nb_path.name} unexpected error ({elapsed:.2f}s): {e}")
        return {
            "notebook": nb_path.name,
            "status": "ERROR",
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e)
        }


def main():
    results = []
    for nb_str in NOTEBOOKS:
        res = execute_notebook(nb_str)
        results.append(res)

    print("\n\n" + "=" * 60)
    print("ALL NOTEBOOKS EXECUTION SUMMARY")
    print("=" * 60)
    for r in results:
        status_mark = "[PASS]" if r["status"] == "PASSED" else "[FAIL]"
        print(f"{status_mark} {r['notebook']:<40} {r['elapsed_seconds']:>6.2f}s  {r['status']}")
        if r["error"]:
            print(f"     Error: {r['error'][:150]}...")

    failures = [r for r in results if r["status"] != "PASSED"]
    if failures:
        print(f"\nTotal Failures: {len(failures)} / {len(NOTEBOOKS)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(NOTEBOOKS)} notebooks executed and verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
