from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = ROOT / "CICIOT23" / "train" / "train.csv"

# Use a sample to keep EDA fast; increase or remove .sample() for exhaustive profiling.
df = pd.read_csv(TRAIN_CSV).sample(100_000, random_state=42)

profile = ProfileReport(df, title="CIC-IoT-2023 EDA Report")
output_path = ROOT / "eda_report.html"
profile.to_file(str(output_path))
print(f"EDA report saved to: {output_path}")