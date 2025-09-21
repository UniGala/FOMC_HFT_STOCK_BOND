# -*- coding: utf-8 -*-
import json
from pathlib import Path

nb = json.loads(Path('PROJECT/Analysis5.ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        for line in cell.get('source', []):
            if "' & '.join(header)" in line:
                print(line)
                break
