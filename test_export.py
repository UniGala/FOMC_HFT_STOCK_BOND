# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd
import numpy as np

nb = json.loads(Path('PROJECT/Analysis5.ipynb').read_text(encoding='utf-8'))
code = None
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'def _short_panel_label' in src:
            code = src
            break
if code is None:
    raise SystemExit('target cell not found')
func_block = code.split("export_table(table1")[0]
ns = {}
exec(func_block, ns)
_prepare_table_layout = ns['_prepare_table_layout']
_build_latex_table = ns['_build_latex_table']
_infer_table_kind = ns['_infer_table_kind']

# Daily sample
index_daily = pd.MultiIndex.from_tuples([
    ('Panel A (Bond Futures)', 'Bond1', 'Avg Daily Volume'),
    ('Panel A (Bond Futures)', 'Bond1', '# Obs'),
    ('Panel B (E-mini Futures)', 'Emini', 'Avg Daily Volume'),
    ('Panel B (E-mini Futures)', 'Emini', '# Obs'),
], names=['Panel', 'Asset', 'Statistic'])

day_levels = ['FOMC_week_Day (-2)', 'FOMC_week_Day (-1)', 'FOMC_week_Day (0)', 'FOMC_week_Day (+1)', 'FOMC_week_Day (+2)']
metrics_daily = ['Mean', 'St. Dev', 'No. Obs']
columns_daily = pd.MultiIndex.from_product([day_levels, metrics_daily], names=['Horizon', 'Metric'])

data_daily = []
for stat in index_daily:
    row = []
    for horizon in day_levels:
        if stat[2] == 'Avg Daily Volume':
            row.extend([100.0, 15.0, 250])
        else:
            row.extend([np.nan, np.nan, 250])
    data_daily.append(row)

table_daily = pd.DataFrame(data_daily, index=index_daily, columns=columns_daily)
layout_daily = _prepare_table_layout(table_daily, 'daily')
latex_daily = _build_latex_table(layout_daily, 'Test Daily Table')
print(latex_daily)

# Intraday sample
index_intraday = pd.MultiIndex.from_tuples([
    ('Panel A (Bond Futures)', 'Bond1', 'Ann Window Volume'),
    ('Panel A (Bond Futures)', 'Bond1', 'Diff (Ann - Non)'),
    ('Panel A (Bond Futures)', 'Bond1', '# Obs'),
    ('Panel B (E-mini Futures)', 'Emini', 'Ann Window Volume'),
    ('Panel B (E-mini Futures)', 'Emini', 'Diff (Ann - Non)'),
    ('Panel B (E-mini Futures)', 'Emini', '# Obs'),
], names=['Panel', 'Asset', 'Statistic'])

windows = ['±15m', '±30m', '±1h', '±2h', '±12h']
metrics_intraday = ['Mean', 'No. Obs']
columns_intraday = pd.MultiIndex.from_product([windows, metrics_intraday], names=['Window', 'Metric'])

data_intraday = []
for stat in index_intraday:
    row = []
    for window in windows:
        if stat[2] == 'Ann Window Volume':
            row.extend([12.0, 200])
        elif stat[2] == 'Diff (Ann - Non)':
            row.extend([1.5, np.nan])
        else:
            row.extend([np.nan, 200])
    data_intraday.append(row)

table_intraday = pd.DataFrame(data_intraday, index=index_intraday, columns=columns_intraday)
layout_intraday = _prepare_table_layout(table_intraday, 'intraday')
latex_intraday = _build_latex_table(layout_intraday, 'Test Intraday Table')
print(latex_intraday)

print('kind daily', _infer_table_kind(table_daily, 'custom_name'))
print('kind intraday', _infer_table_kind(table_intraday, 'custom_intraday'))
