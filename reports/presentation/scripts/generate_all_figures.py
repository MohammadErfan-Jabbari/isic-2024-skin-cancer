"""Generate all figures/tables used in the Slidev deck.

This script is intentionally presentation-scoped:
- Reads inputs from the main project (DeepLearning/Kaggle/...) when needed.
- Writes ONLY to presentation/public/{figures,tables}/.

Run:
  cd .
  uv run python presentation/scripts/generate_all_figures.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRESENTATION_DIR = ROOT / "presentation"
PUBLIC_DIR = PRESENTATION_DIR / "public"
FIG_DIR = PUBLIC_DIR / "figures"
TABLE_DIR = PUBLIC_DIR / "tables"

KAGGLE_DIR = ROOT / "DeepLearning" / "Kaggle"
META_INVEST_DIR = KAGGLE_DIR / "metadata_investigation"
POST_FEATURE_DIR = KAGGLE_DIR / "post_feature_analysis"
LAST_RUN_DIR = KAGGLE_DIR / "last_run"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()

    # TODO: Add figure builders here.
    # Suggested first targets (already have machine-readable outputs):
    # - POST_FEATURE_DIR/results/vision_distribution_shift.csv  -> plot distribution shift summary
    # - POST_FEATURE_DIR/results/verify_vision_model_diversity_FULL.json -> write a small table/plot
    # - META_INVEST_DIR/results/phase6_final_summary.json -> key findings table

    # Placeholder “proof of life” output (so reruns show something happened)
    (TABLE_DIR / "_generated_ok.txt").write_text(
        "Presentation generation script ran successfully.\n"
        "Populate generate_all_figures.py with real plot/table generation.\n"
    )

    print(f"Wrote placeholder output to: {TABLE_DIR / '_generated_ok.txt'}")


if __name__ == "__main__":
    main()
