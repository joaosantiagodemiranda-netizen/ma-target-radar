"""Run the modelling stage (Steps 7-10) on data/processed/ and regenerate every
table and figure used in the manuscript.

    python run_models.py
"""
import time

from src.features import panel
from src.models import evaluate, figures, train

if __name__ == "__main__":
    t = time.time()
    print("=== Step 7: firm-year panel ===");          panel.build()
    print("\n=== Step 8: model ladder (expanding window) ==="); train.run()
    print("\n=== Step 9: evaluation ===");             evaluate.run()
    print("\n=== Step 10: figures ===");               figures.run()
    print(f"\nDone in {time.time() - t:.0f}s. Tables: reports/tables/  Figures: reports/figures/")
