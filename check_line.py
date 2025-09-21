# -*- coding: utf-8 -*-
import json
from pathlib import Path

nb = json.loads(Path('PROJECT/Analysis5.ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        for idx, line in enumerate(cell.get('source', [])):
            if "' & '.join(header)" in line:
                print(idx, repr(line))
                break
