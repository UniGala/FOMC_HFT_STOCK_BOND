# -*- coding: utf-8 -*-
import json
from pathlib import Path

code = """
# Individual Tables and Latex Export
import numpy as np

def _short_panel_label(label):
    if isinstance(label, str):
        stripped = label.strip()
        if stripped.startswith('Panel A'):
            return 'Panel A'
        if stripped.startswith('Panel B'):
            return 'Panel B'
    return str(label)

def _safe_value(series, outer, metric):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[0]
    if isinstance(series.index, pd.MultiIndex):
        key = (outer, metric)
        if key in series.index:
            return series.loc[key]
    else:
        key = f"{outer} {metric}"
        if key in series.index:
            return series.loc[key]
    return np.nan

def _prepare_table_layout(table, kind):
    if not isinstance(table.columns, pd.MultiIndex):
        raise ValueError('Expected table with MultiIndex columns')
    if not isinstance(table.index, pd.MultiIndex):
        table = table.copy()
        table.index = pd.MultiIndex.from_frame(table.index.to_frame())
    stat_level = table.index.names[-1]
    rows = []
    panel_counts = {}
    asset_counts = {}
    if kind == 'daily':
        outer_labels = ['FOMC_week_Day (-2)', 'FOMC_week_Day (-1)', 'FOMC_week_Day (0)', 'FOMC_week_Day (+1)', 'FOMC_week_Day (+2)']
        value_columns = ['-2', '-1', '0', '1', '2']
        for (panel, asset), sub in table.groupby(level=table.index.names[:2], sort=False):
            panel_short = _short_panel_label(panel)
            try:
                avg_series = sub.xs('Avg Daily Volume', level=stat_level)
            except KeyError:
                continue
            obs_fallback = sub.xs('# Obs', level=stat_level) if '# Obs' in sub.index else None
            row_avg = {'Panel': panel_short, 'Asset': str(asset), 'Statistic': 'Avg. Daily Volume', 'RowType': 'avg'}
            row_std = {'Panel': panel_short, 'Asset': str(asset), 'Statistic': 'St. Dev', 'RowType': 'std'}
            row_obs = {'Panel': panel_short, 'Asset': str(asset), 'Statistic': '# Obs', 'RowType': 'obs'}
            for col_label, outer in zip(value_columns, outer_labels):
                row_avg[col_label] = _safe_value(avg_series, outer, 'Mean')
                row_std[col_label] = _safe_value(avg_series, outer, 'St. Dev')
                obs_val = _safe_value(avg_series, outer, 'No. Obs')
                if pd.isna(obs_val) and obs_fallback is not None:
                    obs_val = _safe_value(obs_fallback, outer, 'Mean')
                row_obs[col_label] = obs_val
            for row in (row_avg, row_std, row_obs):
                rows.append(row)
                panel_counts[panel_short] = panel_counts.get(panel_short, 0) + 1
                asset_key = (panel_short, str(asset))
                asset_counts[asset_key] = asset_counts.get(asset_key, 0) + 1
        top_header = 'FOMC Week Day'
    else:
        outer_labels = ['±15m', '±30m', '±1h', '±2h', '±12h']
        value_columns = outer_labels
        for (panel, asset), sub in table.groupby(level=table.index.names[:2], sort=False):
            panel_short = _short_panel_label(panel)
            try:
                avg_series = sub.xs('Ann Window Volume', level=stat_level)
            except KeyError:
                continue
            diff_series = sub.xs('Diff (Ann - Non)', level=stat_level) if 'Diff (Ann - Non)' in sub.index else None
            obs_fallback = sub.xs('# Obs', level=stat_level) if '# Obs' in sub.index else None
            row_avg = {'Panel': panel_short, 'Asset': str(asset), 'Statistic': 'Avg Window Volume', 'RowType': 'avg'}
            row_diff = {'Panel': panel_short, 'Asset': str(asset), 'Statistic': 'Diff', 'RowType': 'diff'}
            row_obs = {'Panel': panel_short, 'Asset': str(asset), 'Statistic': '# Obs', 'RowType': 'obs'}
            for outer in outer_labels:
                row_avg[outer] = _safe_value(avg_series, outer, 'Mean')
                diff_val = np.nan if diff_series is None else _safe_value(diff_series, outer, 'Mean')
                row_diff[outer] = diff_val
                obs_val = _safe_value(avg_series, outer, 'No. Obs')
                if pd.isna(obs_val) and obs_fallback is not None:
                    obs_val = _safe_value(obs_fallback, outer, 'Mean')
                row_obs[outer] = obs_val
            for row in (row_avg, row_diff, row_obs):
                rows.append(row)
                panel_counts[panel_short] = panel_counts.get(panel_short, 0) + 1
                asset_key = (panel_short, str(asset))
                asset_counts[asset_key] = asset_counts.get(asset_key, 0) + 1
        top_header = None
    columns = ['Panel', 'Asset', 'Statistic', 'RowType'] + list(value_columns)
    tidy_df = pd.DataFrame(rows)
    if tidy_df.empty:
        tidy_df = pd.DataFrame(columns=columns)
    else:
        tidy_df = tidy_df[columns]
    return {
        'data': tidy_df,
        'value_columns': list(value_columns),
        'panel_counts': panel_counts,
        'asset_counts': asset_counts,
        'top_header': top_header,
        'kind': kind,
    }

def _format_cell_value(value, *, bold=False, small=False, is_count=False):
    if pd.isna(value):
        text = ''
    else:
        if is_count:
            text = f"{int(round(float(value))):,}"
        else:
            text = f"{float(value):,.3f}"
        if bold and text:
            text = f"\\textbf{{{text}}}"
    if small:
        return f"{{\\footnotesize {text}}}" if text else "{\\footnotesize }"
    return text

def _build_latex_table(layout, caption):
    tidy_df = layout['data']
    value_columns = layout['value_columns']
    panel_counts = layout['panel_counts']
    asset_counts = layout['asset_counts']
    top_header = layout['top_header']
    align = 'lll' + 'r' * len(value_columns)
    lines = [
        r"\\begin{table}[!htbp]\\centering",
        f"\\caption{{{caption}}}",
        r"\\small",
        f"\\begin{{tabular}}{{{align}}}",
        r"\\toprule",
    ]
    if top_header:
        lines.append(r"\\multicolumn{3}{c}{} & " + rf"\\multicolumn{{{len(value_columns)}}}{{c}}{{{top_header}}} \\\")
    header = ['Panel', 'Asset', 'Statistic'] + value_columns
    lines.append(' & '.join(header) + r" \\")
    lines.append(r"\\midrule")
    used_panel = set()
    used_asset = set()
    for _, row in tidy_df.iterrows():
        panel = row['Panel']
        asset = row['Asset']
        row_type = row['RowType']
        if panel not in used_panel:
            panel_cell = f"\\multirow{{{panel_counts[panel]}}}{{*}}{{{panel}}}"
            used_panel.add(panel)
        else:
            panel_cell = ''
        asset_key = (panel, asset)
        if asset_key not in used_asset:
            asset_cell = f"\\multirow{{{asset_counts[asset_key]}}}{{*}}{{{asset}}}"
            used_asset.add(asset_key)
        else:
            asset_cell = ''
        if row_type == 'avg':
            stat_cell = f"\\textbf{{{row['Statistic']}}}"
        else:
            stat_cell = f"{{\\footnotesize {row['Statistic']}}}"
        small = row_type != 'avg'
        is_count = row_type == 'obs'
        cells = [panel_cell, asset_cell, stat_cell]
        for col in value_columns:
            cells.append(_format_cell_value(row.get(col), bold=row_type == 'avg', small=small, is_count=is_count))
        lines.append(' & '.join(cells) + r" \\")
    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append('')
    lines.append(r"\\end{table}")
    return '\\n'.join(lines)

def _infer_table_kind(table, basename):
    name = basename.lower()
    if 'daily' in name:
        return 'daily'
    if 'intraday' in name:
        return 'intraday'
    if isinstance(table.columns, pd.MultiIndex):
        top_labels = [str(level) for level in table.columns.get_level_values(0)]
    else:
        top_labels = [str(col) for col in table.columns]
    if any('Day' in label for label in top_labels):
        return 'daily'
    return 'intraday'

def export_table(table, subdir, basename, caption):
    out_dir = OUTPUT_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = _infer_table_kind(table, basename)
    layout = _prepare_table_layout(table, kind)
    tidy_df = layout['data']
    excel_df = tidy_df.drop(columns=['RowType']) if 'RowType' in tidy_df.columns else tidy_df.copy()
    excel_path = out_dir / f"{basename}.xlsx"
    excel_df.to_excel(excel_path, index=False)
    latex_content = _build_latex_table(layout, caption)
    latex_path = out_dir / f"{basename}.tex"
    latex_path.write_text(latex_content, encoding='utf-8')
    print('Saved', excel_path)
    print('Saved', latex_path)

export_table(table1, 'table1', 'Table1_daily_volume_total', 'Daily volume around FOMC (Total period)')
export_table(table2, 'table2', 'Table2_intraday_volume_total', 'Intraday volume around FOMC (Total period)')

periods = ['Pre-ZLB', 'ZLB', 'Post-ZLB']
for period in periods:
    export_table(
        table3_daily[period],
        'table3',
        f'Table3_{period}_daily_volume',
        f'Daily volume around FOMC ({period})'
    )
    export_table(
        table3_intraday[period],
        'table3',
        f'Table3_{period}_intraday_volume',
        f'Intraday volume around FOMC ({period})'
    )
""".strip('\n')

lines = code.split('\n')
lines[133] = r"    if top_header:"  # maintain consistent referencing? (line adjust optional)
with_newlines = [line + '\n' for line in lines]
nb_path = Path('PROJECT/Analysis5.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'def _short_panel_label' in src:
            cell['source'] = with_newlines
            break
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
