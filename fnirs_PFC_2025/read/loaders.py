import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_txt_file(file_path: str) -> dict:

    try:
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
    except Exception as e:
        logger.error(f"Failed to read file: {file_path}. Error: {e}")
        return None

    if not lines:
        logger.error(f"File is empty: {file_path}")
        return None

    rows = [row.split('\t') for row in lines]

    # ---- metadata ----
    metadata = {}
    wavelengths_parsed = False

    try:
        for i in range(min(50, len(rows))):
            row = rows[i]
            if not row:
                continue

            k = row[0].strip() if row[0] else ""
            kl = k.lower()

            if len(row) >= 2:
                v = row[1].strip()

                if kl.startswith("start of measurement"):
                    metadata['Measurement Start'] = v
                elif kl.startswith("record date/time"):
                    metadata['Record Date/Time'] = v
                elif kl.startswith("export date"):
                    metadata['Export date'] = v
                elif kl.startswith("subject public id"):
                    metadata['Subject Public ID'] = v
                elif kl.startswith("export sample rate"):
                    try:
                        metadata['Sample Rate (Hz)'] = float(v)
                    except ValueError:
                        pass

            if "light source wavelengths" in kl and not wavelengths_parsed:
                metadata['Wavelengths'] = {}
                for j in range(i + 2, min(i + 20, len(rows))):
                    row_j = rows[j]
                    if not row_j or (len(row_j) == 1 and row_j[0].strip() == ''):
                        break
                    if len(row_j) >= 3:
                        try:
                            device_id = row_j[0].strip()
                            if not device_id.isdigit():
                                break
                            src_idx = int(row_j[1].strip())
                            wavelength = int(row_j[2].strip())
                            metadata['Wavelengths'][src_idx] = wavelength
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse wavelength row {j}: {row_j} - {e}")
                            break
                wavelengths_parsed = True
                logger.debug(f"Parsed {len(metadata.get('Wavelengths', {}))} wavelengths: "
                             f"{metadata.get('Wavelengths', {})}")

    except Exception as e:
        logger.warning(f"Metadata parsing error in {file_path}: {e}")

    if 'Wavelengths' not in metadata or not metadata['Wavelengths']:
        logger.error(f"No wavelengths found in metadata for {file_path}; all channels will be UNMAPPED")

    # sample rate + header row indices
    start_idx = end_idx = None
    sample_rate = metadata.get('Sample Rate (Hz)')

    for idx, row in enumerate(rows):
        if len(row) > 1 and "Datafile sample rate:" in row[0]:
            try:
                sample_rate = float(row[1])
                metadata['Sample Rate (Hz)'] = sample_rate
            except Exception:
                logger.warning(f"Could not parse sample rate in {file_path}")

    # Column header rows are bounded by the "(Sample number)" and "(Event)" markers.
    for idx, row in enumerate(rows):
        row_str = '\t'.join(str(cell) for cell in row)
        if "(Sample number)" in row_str and start_idx is None:
            start_idx = idx
        if "(Event)" in row_str:
            end_idx = idx

    if start_idx is None or end_idx is None:
        logger.error(f"Could not identify column header rows in {file_path} "
                     f"(start_idx={start_idx}, end_idx={end_idx})")
        return None

    if sample_rate is None:
        logger.warning(f"Could not determine sample rate for {file_path}")

    metadata['Export file'] = file_path

    if 'Subject Public ID' not in metadata or not metadata['Subject Public ID']:
        m = re.search(r'(OHSU[_-]?Turn[_-]?\d+|Turn[_-]?\d+|sub[-_]\w+)', file_path, flags=re.IGNORECASE)
        metadata['Subject Public ID'] = m.group(1) if m else None
        if metadata['Subject Public ID']:
            logger.warning(f"Inferred Subject ID from filename: {metadata['Subject Public ID']}")

    if 'Record Date/Time' not in metadata and 'Export date' in metadata:
        metadata['Record Date/Time'] = metadata['Export date']
        logger.warning(f"Using Export date as Record Date/Time for {file_path}")

    # ---- column labels ----
    # Column labels for the OD export span multiple header rows; the label
    # itself sits in column index 1 (or 0, for single-column rows).
    try:
        col_labels = []
        for idx in range(start_idx, end_idx + 1):
            row = rows[idx]
            if len(row) > 1:
                col_labels.append(row[1])
            elif len(row) == 1:
                col_labels.append(row[0])
            else:
                logger.warning(f"Unexpected row structure at {idx}: {row}")
                col_labels.append("")
    except Exception as e:
        logger.error(f"Failed to parse column labels in {file_path}: {e}")
        return None

    col_labels = _process_od_column_labels(col_labels, metadata, file_path)

    # ---- data rows ----
    data_rows = rows[end_idx + 4:]
    if not data_rows:
        logger.error(f"No data rows found after header in {file_path}")
        return None

    clean_rows = []
    for row in data_rows:
        if len(row) == len(col_labels) + 1 and row[-1] == '':
            row = row[:-1]
        if len(row) != len(col_labels):
            continue
        clean_rows.append(row)

    if not clean_rows:
        logger.error(f"No valid data rows in {file_path}")
        return None

    df = pd.DataFrame(clean_rows, columns=col_labels)

    # Convert every column except Event to numeric; a column that can't
    # convert (unexpected content) is logged and left as-is rather than
    # aborting the whole file.
    num_cols = [c for c in df.columns if c != 'Event']
    for col in num_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to convert column '{col}' to numeric: {e}")

    if 'Event' in df.columns:
        df['Event'] = df['Event'].astype(str).str.strip()
        df['Event'] = df['Event'].replace({'': np.nan, 'nan': np.nan})

    # Drop the first two rows (OxySoft initialization artifacts) if there's room.
    if len(df) > 3:
        df = df.iloc[3:].reset_index(drop=True)

    if 'Sample number' in df.columns:
        df['Sample number'] = pd.to_numeric(df['Sample number'], errors='coerce').fillna(0).astype(int)

    df = _reassign_channels_od(df, metadata, file_path)

    if df is None or df.empty:
        logger.error(f"DataFrame empty after channel reassignment in {file_path}")
        return None

    return {'metadata': metadata, 'data': df}


def _process_od_column_labels(col_labels, metadata, file_path):
    """
    Map raw column labels to:
      - 'Sample number'
      - 'ADC'
      - 'Event'
      - OD columns in the form 'CH{channel}_WL{wavelength}'

    Using the legend:

      Column 1     -> (Sample number)
      Columns 2-17 -> Light sources 1-16
      Column 18    -> ADC
      Column 19    -> (Event)
    """
    wavelength_map = metadata.get('Wavelengths', {})

    # Channel pairing based on OctaMon 2x3 channel + 2x1 SSC.
    # channel_number: (light_source_a, light_source_b)
    channel_pairing = {
        0: (1, 2),
        1: (3, 4),
        2: (5, 6),
        3: (7, 8),
        4: (9, 10),
        5: (11, 12),
        6: (13, 14),
        7: (15, 16),
    }

    new_labels = []

    for i, label in enumerate(col_labels):
        column_number = i + 1  # legend is 1-based

        if column_number == 1:
            new_labels.append('Sample number')
        elif column_number == 18:
            new_labels.append('ADC')
        elif column_number == 19:
            new_labels.append('Event')
        elif 2 <= column_number <= 17:
            light_source_idx = column_number - 1

            channel_found = None
            wavelength = None
            for ch_num, (ls_a, ls_b) in channel_pairing.items():
                if light_source_idx in (ls_a, ls_b):
                    channel_found = ch_num
                    wavelength = wavelength_map.get(light_source_idx)
                    if wavelength is None:
                        logger.warning(
                            f"No wavelength found for light source {light_source_idx} in metadata for {file_path}")
                        new_labels.append(f"UNMAPPED_{light_source_idx}")
                        break
                    new_labels.append(f"CH{channel_found}_WL{wavelength}")
                    break

            if channel_found is None and wavelength is not None:
                logger.warning(
                    f"Could not map light source {light_source_idx} to a channel "
                    f"in {file_path}; keeping as UNMAPPED_{light_source_idx}"
                )
                new_labels.append(f"UNMAPPED_{light_source_idx}")
        else:
            logger.warning(f"Unexpected column number {column_number} in {file_path}")
            new_labels.append(f"UNKNOWN_COL_{column_number}")

    if len(new_labels) != len(set(new_labels)):
        duplicates = [item for item, count in Counter(new_labels).items() if count > 1]
        logger.error(f"Duplicate column names: {duplicates}")

    return new_labels


def _reassign_channels_od(df: pd.DataFrame, metadata: dict, file_path: str) -> pd.DataFrame:
    """Order columns as Sample number, then CH*_WL* sorted by channel/wavelength, then ADC, then Event."""
    cols = list(df.columns)

    standard_cols = ['Sample number', 'ADC', 'Event']
    data_cols = [col for col in cols if col not in standard_cols]

    def sort_key(col):
        if col.startswith('CH') and 'WL' in col:
            try:
                ch_num = int(col.split('CH')[1].split('_')[0])
                wl_num = int(col.split('WL')[1])
                return (ch_num, wl_num)
            except (ValueError, IndexError):
                return (999, 999)
        return (1000, 1000)

    sorted_data_cols = sorted(data_cols, key=sort_key)

    new_cols = []
    if 'Sample number' in cols:
        new_cols.append('Sample number')
    new_cols.extend(sorted_data_cols)
    if 'ADC' in cols:
        new_cols.append('ADC')
    if 'Event' in cols:
        new_cols.append('Event')

    if len(new_cols) == len(cols):
        df = df[new_cols]
    else:
        logger.warning(f"Column count mismatch in OD reassignment for {file_path}")

    return df
