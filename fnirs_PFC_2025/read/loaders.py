import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INIT_ROWS = 3  # OxySoft junk at the top of every export

_META_KEYS = {
    "start of measurement": "Measurement Start",
    "record date/time": "Record Date/Time",
    "export date": "Export date",
    "subject public id": "Subject Public ID",
}


def read_txt_file(file_path):
    """Read an OxySoft OD export. Returns {'metadata', 'data'} or None if it can't be parsed."""
    with open(file_path) as f:
        rows = [line.split('\t') for line in f.read().splitlines()]
    if not rows:
        return None

    meta = {'Wavelengths': {}, 'Export file': file_path}
    for i, row in enumerate(rows[:50]):
        key = row[0].strip().lower()
        if len(row) >= 2:
            for prefix, name in _META_KEYS.items():
                if key.startswith(prefix):
                    meta[name] = row[1].strip()
            if key.startswith("export sample rate"):
                meta['Sample Rate (Hz)'] = _to_float(row[1], meta.get('Sample Rate (Hz)'))
        if "light source wavelengths" in key and not meta['Wavelengths']:
            meta['Wavelengths'] = _read_wavelengths(rows[i + 2:i + 20])
    if not meta['Wavelengths']:
        logger.error("no wavelengths in %s, all channels will be UNMAPPED", file_path)

    for row in rows:
        if len(row) > 1 and "Datafile sample rate:" in row[0]:
            meta['Sample Rate (Hz)'] = _to_float(row[1], meta.get('Sample Rate (Hz)'))

    start = next((i for i, r in enumerate(rows) if "(Sample number)" in '\t'.join(r)), None)
    ends = [i for i, r in enumerate(rows) if "(Event)" in '\t'.join(r)]
    if start is None or not ends:
        logger.error("no column header rows in %s", file_path)
        return None
    end = ends[-1]

    if not meta.get('Subject Public ID'):
        m = re.search(r'(OHSU[_-]?Turn[_-]?\d+|Turn[_-]?\d+|sub[-_]\w+)', file_path, flags=re.IGNORECASE)
        meta['Subject Public ID'] = m.group(1) if m else None
    if 'Record Date/Time' not in meta and 'Export date' in meta:
        meta['Record Date/Time'] = meta['Export date']

    labels = _column_labels(len(rows[start:end + 1]), meta['Wavelengths'], file_path)

    data = []
    for row in rows[end + 4:]:
        if len(row) == len(labels) + 1 and row[-1] == '':
            row = row[:-1]
        if len(row) == len(labels):
            data.append(row)
    if not data:
        logger.error("no data rows in %s", file_path)
        return None

    df = pd.DataFrame(data, columns=labels)
    for col in df.columns.drop('Event'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Event'] = df['Event'].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan})

    if len(df) > INIT_ROWS:
        df = df.iloc[INIT_ROWS:].reset_index(drop=True)
    df['Sample number'] = df['Sample number'].fillna(0).astype(int)

    # Sample number, CHn_WLxxx by channel/wavelength, ADC, Event
    def order(col):
        m = re.fullmatch(r'CH(\d+)_WL(\d+)', col)
        return (int(m[1]), int(m[2])) if m else (1000, 1000)

    signal = sorted((c for c in df.columns if c not in ('Sample number', 'ADC', 'Event')), key=order)
    df = df[['Sample number'] + signal + ['ADC', 'Event']]
    return {'metadata': meta, 'data': df}


def _to_float(text, default=None):
    try:
        return float(text)
    except ValueError:
        return default


def _read_wavelengths(rows):
    """{light source index: wavelength} from the rows under the 'Light source wavelengths' line."""
    out = {}
    for row in rows:
        if len(row) < 3 or not row[0].strip().isdigit():
            break
        try:
            out[int(row[1])] = int(row[2])
        except ValueError:
            break
    return out


def _column_labels(n_cols, wavelengths, file_path):
    """Column 1 = sample number, 2-17 = light sources 1-16, 18 = ADC, 19 = event.

    OctaMon: two light sources per channel, (1,2) -> CH0 ... (15,16) -> CH7.
    """
    labels = []
    for col in range(1, n_cols + 1):
        if col == 1:
            labels.append('Sample number')
        elif col == 18:
            labels.append('ADC')
        elif col == 19:
            labels.append('Event')
        elif 2 <= col <= 17:
            src = col - 1
            wl = wavelengths.get(src)
            labels.append(f"CH{(src - 1) // 2}_WL{wl}" if wl else f"UNMAPPED_{src}")
        else:
            labels.append(f"UNKNOWN_COL_{col}")

    dupes = [k for k, n in Counter(labels).items() if n > 1]
    if dupes:
        logger.error("duplicate column names in %s: %s", file_path, dupes)
    return labels
