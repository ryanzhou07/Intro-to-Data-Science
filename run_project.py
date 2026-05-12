from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(script_path: Path) -> None:
    """
    Run one pipeline script from the project root.

    Parameters:
    - script_path: Path to a Python script relative to the project root

    Returns:
    - None
    """
    print(f"\nRunning {script_path}...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / script_path)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """
    Run the full reproducible local project pipeline.
    """
    steps = [
        Path("scripts") / "repair_appid_price_matching.py",
        Path("scripts") / "merge_clean_data.py",
        Path("scripts") / "run_modeling.py",
    ]

    for step in steps:
        run_step(step)

    print("\nDone. Outputs were regenerated in Data/, figures/, and results_summary.md.")


if __name__ == "__main__":
    main()
