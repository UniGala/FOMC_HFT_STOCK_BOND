# -*- coding: utf-8 -*-
import json
from pathlib import Path

nb_path = Path('PROJECT/Analysis5.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = cell.get('source', [])
        for idx, line in enumerate(src):
            if 'multicolumn{3}{c}{}' in line and '+ rf' in line:
                src[idx] = '    lines.append(r"\\multicolumn{3}{c}{} & " + rf"\\multicolumn{{{len(value_columns)}}}{{c}}{{{top_header}}} \\\\\\)")\n'
                break
        break
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
