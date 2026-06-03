"""
fix_notebook.py
===============
Fixes baseline_evaluation.ipynb in-place:
  1. Renames misleading variable names (X_test→X_val, y_binary_test→y_binary_val,
     y_family_test→y_family_val) in all code-cell source strings.
  2. Clears all cell outputs and resets execution_count so the notebook
     is clean and ready to re-run on the current data.
"""

import json
import re
from pathlib import Path

NOTEBOOK = Path("baseline_evaluation.ipynb")

RENAMES = [
    ("X_test",        "X_val"),
    ("y_binary_test", "y_binary_val"),
    ("y_family_test", "y_family_val"),
]

def fix_source_line(line: str) -> str:
    """Apply all variable renames to a single source-array string."""
    for old, new in RENAMES:
        # Use word-boundary-like replacement so we don't partial-match
        line = re.sub(r'\b' + re.escape(old) + r'\b', new, line)
    return line


def main():
    with NOTEBOOK.open("r", encoding="utf-8") as fh:
        nb = json.load(fh)

    cells_fixed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        # 1. Fix variable names in source lines
        new_source = [fix_source_line(line) for line in cell.get("source", [])]
        if new_source != cell.get("source", []):
            cell["source"] = new_source
            cells_fixed += 1

        # 2. Clear outputs and reset execution count
        cell["outputs"] = []
        cell["execution_count"] = None

    with NOTEBOOK.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")   # trailing newline for clean diffs

    print(f"Done. Variable renames applied to {cells_fixed} cell(s).")
    print("All cell outputs cleared — re-run the notebook to regenerate them.")


if __name__ == "__main__":
    main()
