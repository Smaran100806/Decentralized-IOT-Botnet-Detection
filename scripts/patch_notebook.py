import json
from pathlib import Path

nb_path = Path('baseline_evaluation.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell['source'])

    # ── Fix binary model comparison cell ─────────────────────────────────────
    if 'binary_results.append' in src and 'binary_df = pd.DataFrame' in src:
        # We replace the last 3 lines
        new_source = []
        for line in cell['source']:
            if line.startswith('binary_df = pd.DataFrame'):
                new_source.append('if binary_results:\n')
                new_source.append('    ' + line)
            elif line.startswith('display(binary_df'):
                new_source.append('    ' + line)
            elif line.startswith('print(f"Best binary model'):
                new_source.append('    ' + line)
                new_source.append('else:\n')
                new_source.append('    print("No binary models found — run train_binary_baseline.py first.")\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

    # ── Fix family model comparison cell ─────────────────────────────────────
    if 'family_results.append' in src and 'family_df = pd.DataFrame' in src:
        new_source = []
        for line in cell['source']:
            if line.startswith('family_df = pd.DataFrame'):
                new_source.append('if family_results:\n')
                new_source.append('    ' + line)
            elif line.startswith('display(family_df'):
                new_source.append('    ' + line)
            elif line.startswith('print(f"Best family model'):
                new_source.append('    ' + line)
                new_source.append('else:\n')
                new_source.append('    print("No family models found — run train_family_baseline.py first.")\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('Successfully patched baseline_evaluation.ipynb')
