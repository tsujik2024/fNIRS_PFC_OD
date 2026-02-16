import logging
import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust logging level as needed

def read_txt_file(file_path: str) -> dict:
    """
    Reads an OxySoft .txt export and returns:
      - 'metadata': parsed header info (+ 'Sample Rate (Hz)')
      - 'data': cleaned DataFrame with renamed channels 'CH{i} HbO' / 'CH{i} HbR'
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
    except Exception as e:
        logger.error(f" Failed to read file: {file_path}. Error: {e}")
        return None

    if not lines:
        logger.error(f" File is empty: {file_path}")
        return None

    rows = [row.split('\t') for row in lines]

    # ---- metadata ----
    metadata = {}
    try:
        for i in range(min(7, len(rows))):
            row = rows[i]
            if not row or len(row) < 2:
                continue
            k = row[0].strip()
            v = row[1].strip()
            kl = k.lower()
            if kl.startswith("start of measurement"):
                metadata['Measurement Start'] = v
            elif kl.startswith("record date/time"):
                metadata['Record Date/Time'] = v
            elif kl.startswith("export date"):
                metadata['Export date'] = v
            elif kl.startswith("subject public id"):
                metadata['Subject Public ID'] = v
    except Exception as e:
        logger.warning(f" Metadata parsing error in {file_path}: {e}")

    # sample rate + header row indices
    start_idx = end_idx = None
    sample_rate = None
    for idx, row in enumerate(rows):
        if "Datafile sample rate:" in row:
            try:
                sample_rate = int(float(row[1]))
            except Exception:
                logger.warning(f" Could not parse sample rate in {file_path}")
        if "(Sample number)" in row:
            start_idx = idx
        if "(Event)" in row:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        logger.error(f" Could not identify column header rows in {file_path}")
        return None
    metadata['Sample Rate (Hz)'] = sample_rate
    metadata['Export file'] = file_path

    # Subject fallback
    if 'Subject Public ID' not in metadata or not metadata['Subject Public ID']:
        m = re.search(r'(OHSU[_-]?Turn[_-]?\d+|sub[-_]\w+)', file_path, flags=re.IGNORECASE)
        metadata['Subject Public ID'] = m.group(1) if m else None
        if metadata['Subject Public ID']:
            logger.warning(f" Inferred Subject ID from filename: {metadata['Subject Public ID']}")

    # Date fallback
    if 'Record Date/Time' not in metadata and 'Export date' in metadata:
        metadata['Record Date/Time'] = metadata['Export date']
        logger.warning(f" Using Export date as Record Date/Time for {file_path}")

    # ---- column labels ----
    try:
        col_labels = [r[1] for r in rows[start_idx:end_idx+1]]
    except Exception as e:
        logger.error(f" Failed to parse column labels in {file_path}: {e}")
        return None

    for i, label in enumerate(col_labels):
        # Signals → strip "(...)"
        if any(tok in label for tok in ("O2Hb", "HHb", "HbO", "HbR")):
            col_labels[i] = label.split('(')[0].strip()
        # Sample/Event → pull token in parentheses
        elif "(Sample number)" in label or "(Event)" in label:
            parts = label.split('(')
            if len(parts) == 2:
                col_labels[i] = parts[1].split(')')[0]
            else:
                logger.warning(f" Unexpected label format: {label}")

    # ---- data rows ----
    data_rows = rows[end_idx + 4:]
    if not data_rows:
        logger.error(f" No data rows found after header in {file_path}")
        return None

    clean_rows = []
    for row in data_rows:
        if len(row) == len(col_labels) + 1 and row[-1] == '':
            row = row[:-1]
        if len(row) != len(col_labels):
            continue
        clean_rows.append(row)

    if not clean_rows:
        logger.error(f" No valid data rows in {file_path}")
        return None

    df = pd.DataFrame(clean_rows, columns=col_labels)

    # Convert numeric where possible (preserve Event)
    num_cols = [c for c in df.columns if c != 'Event']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Clean Event once
    if 'Event' in df.columns:
        df['Event'] = df['Event'].astype(str).str.strip()
        df['Event'] = df['Event'].replace({'': np.nan, 'nan': np.nan})

    # Drop first two rows (OxySoft artifacts) only if there’s room
    if len(df) > 2:
        df = df.iloc[2:].reset_index(drop=True)

    # Ensure Sample number is int if present
    if 'Sample number' in df.columns:
        df['Sample number'] = pd.to_numeric(df['Sample number'], errors='coerce').fillna(0).astype(int)

    # Final rename to CH{i} HbO/HbR pairs
    df = _reassign_channels(df, file_path)
    if df is None or df.empty:
        logger.error(f" DataFrame empty after channel reassignment in {file_path}")
        return None

    return {'metadata': metadata, 'data': df}


def _read_metadata(rows: list, file_path: str) -> dict:
    """
    Parses up to the first 7 lines of the OxySoft .txt header,
    extracting into metadata:
      - 'Measurement Start'    (from "Start of measurement:")
      - 'Record Date/Time'     (from "Record Date/Time:")
      - 'Export date'          (from "Export date:")
      - 'Subject Public ID'    (falls back to filename if missing)
    """
    metadata = {}
    for i in range(min(7, len(rows))):
        row = rows[i]
        if not row or len(row) < 2:
            continue
        key = row[0].strip()
        val = row[1].strip()

        kl = key.lower()
        # 1️⃣ Prefer “Start of measurement”
        if kl.startswith("start of measurement"):
            metadata['Measurement Start'] = val
            logger.info(f"Using 'Start of measurement' for date: {val}")
        # 2️⃣ Then capture Record Date/Time
        elif kl.startswith("record date/time"):
            metadata['Record Date/Time'] = val
        # 3️⃣ Then Export date
        elif kl.startswith("export date"):
            metadata['Export date'] = val
        # also capture subject if it appears directly
        elif kl.startswith("subject public id"):
            metadata['Subject Public ID'] = val

    # ── Ensure we have a Record Date/Time field (if only Export date existed)
    if 'Record Date/Time' not in metadata and 'Export date' in metadata:
        metadata['Record Date/Time'] = metadata['Export date']
        logger.warning(
            "No 'Record Date/Time' header—falling back to 'Export date'"
        )

    # ── Fallback: infer subject ID from filename if still missing
    if 'Subject Public ID' not in metadata:
        m = re.search(r'(OHSU_Turn_\d+)', file_path)
        if m:
            metadata['Subject Public ID'] = m.group(1)
            logger.warning(
                f"Inferred Subject Public ID from filename: {m.group(1)}"
            )
        else:
            metadata['Subject Public ID'] = None
            logger.error(f"Could not infer Subject Public ID in {file_path}")

    metadata['Export file'] = file_path
    return metadata

def _read_data(rows: list, file_path: str) -> pd.DataFrame:
    """
    Internal helper function to parse the data portion of the Oxysoft .txt file.
    Returns a DataFrame with columns for O2Hb, HHb, Sample number, Event, etc.
    """
    rows_copy = rows.copy()
    start = None
    end = None
    sample_rate = None

    for idx, row in enumerate(rows_copy):
        if "Datafile sample rate:" in row:
            try:
                sample_rate = int(float(row[1]))
            except (ValueError, IndexError):
                logger.error(f"Could not parse sample rate from row {idx}. Row content: {row}")
                sample_rate = None
        elif "(Sample number)" in row:
            start = idx
        elif "(Event)" in row:
            end = idx
            break

    if start is None or end is None or sample_rate is None:
        msg = (
            f"Could not find required markers in '{file_path}'.\n"
            f"start={start}, end={end}, sample_rate={sample_rate}"
        )
        logger.error(msg)
        raise ValueError(msg)

    # The column labels are in rows from 'start' to 'end'
    col_label_rows = rows_copy[start: end + 1]
    try:
        col_labels = [r[1] for r in col_label_rows]
    except IndexError as e:
        logger.error(
            f"Column label rows do not have the expected structure in file '{file_path}'. Row content: {col_label_rows}"
        )
        raise

    # Clean up column labels: For signals, remove the trailing part (e.g., "O2Hb(1)" -> "O2Hb")
    for idx, label in enumerate(col_labels):
        if "O2Hb" in label or "HHb" in label:
            new_label = label.split('(')[0].strip()
            col_labels[idx] = new_label
        elif "(Sample number)" in label or "(Event)" in label:
            parts = label.split('(')
            if len(parts) == 2:
                new_label = parts[1].split(')')[0]
                col_labels[idx] = new_label
            else:
                logger.warning(f"Unexpected format for column label '{label}'. Leaving it as is.")
        else:
            logger.warning(f"Unexpected value in column labels: '{label}'.")

    # Data section starts after end + 4 lines
    data_rows = rows_copy[end + 4:]
    if not data_rows:
        msg = (
            f"No data rows found after line {end + 4} in file '{file_path}'. "
            "Check if the file is truncated or incorrectly formatted."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Remove the last line if it is empty
    if len(data_rows[-1]) == 1 and data_rows[-1][0] == '':
        data_rows = data_rows[:-1]

    # Clean each row to have the expected number of columns
    clean_data_rows = []
    for idx, row in enumerate(data_rows):
        if len(row) == len(col_labels) + 1:
            if row[-1] == '':
                row.pop()
            else:
                logger.warning(
                    f"Row {idx} has {len(row)} items (1 too many), but the extra item is not empty. Row content: {row}"
                )
                row.pop()
        elif len(row) != len(col_labels):
            logger.error(
                f"Row {idx} has {len(row)} columns, expected {len(col_labels)}. Row content: {row}"
            )
            continue
        clean_data_rows.append(row)

    df = pd.DataFrame(data=clean_data_rows, columns=col_labels)
    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    # Replace empty strings with NaN in the 'Event' column if it exists
    if 'Event' in df.columns:
        mask_empty = (df['Event'] == '')
        if mask_empty.any():
            df.loc[mask_empty, 'Event'] = np.nan
            logger.debug(f"Replaced empty strings with NaN in 'Event' column for file '{file_path}'.")

        df['Event'] = df['Event'].astype(str)
        df['Event'] = df['Event'].str.strip()# Force everything to string
        df['Event'] = df['Event'].replace('nan', np.nan)  # Reset any 'nan' strings back to real NaN
    else:
        logger.warning(f"No 'Event' column found in file '{file_path}'. This may be normal or unexpected.")

    # Drop only the first two rows
    if len(df) > 2:
        df.drop(df.index[[0, 1]], inplace=True)
        logger.debug(
            f"Dropped the first 2 rows of data in '{file_path}' to remove initialization artifacts."
        )
    else:
        logger.warning(
            f"Dataframe is too short (only {len(df)} rows). Not dropping the first two rows."
        )

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    if 'Event' in df.columns:
        mask_empty = (df['Event'] == '')
        if mask_empty.any():
            df.loc[mask_empty, 'Event'] = np.nan
            logger.debug(f"Replaced empty strings with NaN in 'Event' column for file '{file_path}'.")
    else:
        logger.warning(f"No 'Event' column found in file '{file_path}'. This may be normal or unexpected.")

    return df


def label_turning_epochs(fnirs_df, subject_id, metadata, turning_csv_path, sample_rate, task_type='ST'):
    """
    Labels rows as 'Turning' or 'Walking' by:
    1. First trying to match fNIRS file's subject + header dates to Mobility Lab CSV
    2. If that fails, matching subject + Visit Type (with Baseline=Pre)
    """
    df_turn = pd.read_csv(turning_csv_path)

    # Parse the Mobility Lab timestamp into date and visit type
    df_turn['ML Date'] = pd.to_datetime(
        df_turn['Record Date/Time'],
        errors='coerce',
        utc=True
    ).dt.tz_localize(None)  # Remove timezone for comparison

    # Clean Visit Type column
    df_turn['Visit Type'] = df_turn['Visit Type'].str.strip().str.lower()

    # Build a list of possible dates from metadata (timezone-naive)
    possible_dates = []
    for key in ('Measurement Start', 'Record Date/Time', 'Export date'):
        dt = metadata.get(key)
        if dt:
            # Try parsing without timezone first
            p = pd.to_datetime(dt, errors='coerce')
            if pd.isna(p):
                p = pd.to_datetime(dt, errors='coerce', utc=True)

            if not pd.isna(p):
                # Make timezone-naive for comparison
                p = p.tz_localize(None)
                possible_dates.append(p.date())

    # Remove duplicates while preserving order
    possible_dates = list(dict.fromkeys(possible_dates))

    # Try to get visit type from file path (Baseline or Pre)
    visit_type = None
    file_path = metadata.get('Export file', '')
    if 'baseline' in file_path.lower():
        visit_type = 'baseline'
    elif 'pre' in file_path.lower():
        visit_type = 'pre'
    elif 'post' in file_path.lower():
        visit_type = 'post'

    # Debug logging
    logger.info(f"[{task_type}] Subject {subject_id}: fNIRS dates = {possible_dates}")
    logger.info(f"[{task_type}] Subject {subject_id}: ML dates = {df_turn['ML Date'].dt.date.unique().tolist()}")
    logger.info(f"[{task_type}] Subject {subject_id}: Inferred visit type = {visit_type}")

    # First try: match on Subject & Date
    sub_df = df_turn[
        (df_turn['Subject Public ID'] == subject_id) &
        (df_turn['ML Date'].dt.date.isin(possible_dates))
        ]

    # Second try: if no date match, try matching by visit type (with Baseline=Pre)
    if sub_df.empty and visit_type:
        # Map our inferred visit type to possible ML visit types
        ml_visit_types = []
        if visit_type == 'baseline':
            ml_visit_types = ['baseline', 'pre']  # accept either
        elif visit_type == 'pre':
            ml_visit_types = ['pre', 'baseline']  # accept either
        else:
            ml_visit_types = [visit_type]

        sub_df = df_turn[
            (df_turn['Subject Public ID'] == subject_id) &
            (df_turn['Visit Type'].str.lower().isin(ml_visit_types))
            ]
        if not sub_df.empty:
            logger.info(f"[{task_type}] Matched {subject_id} by visit type {visit_type}")

    if sub_df.empty:
        logger.warning(
            f"[{task_type}] No turning metadata for {subject_id} on dates {possible_dates} or visit type {visit_type}")
        fnirs_df['TaskPhase'] = 'Walking'
        return fnirs_df

    logger.info(f"[{task_type}] Found turning metadata for {subject_id}")

    # extract onsets & durations
    on = sub_df[sub_df['Measure'] == 'Turns - Turn (s)']
    du = sub_df[sub_df['Measure'] == 'Turns - Duration (s)']

    # only numeric, non-nan columns
    turn_cols = [
        c for c in on.columns
        if c.isdigit() and not pd.isna(on.iloc[0][c]) and not pd.isna(du.iloc[0][c])
    ]

    fnirs_df['TaskPhase'] = 'Walking'
    for col in turn_cols:
        try:
            o_sec = float(on.iloc[0][col])
            d_sec = float(du.iloc[0][col])
            start = int(o_sec * sample_rate)
            end = int((o_sec + d_sec) * sample_rate)
            if 0 <= start < len(fnirs_df):
                end = min(end, len(fnirs_df) - 1)
                fnirs_df.loc[start:end, 'TaskPhase'] = 'Turning'
            else:
                logger.warning(f"[{task_type}] Turn {col} out of bounds: {start}-{end}")
        except Exception as e:
            logger.warning(f"[{task_type}] Error on turn {col}: {e}")

    return fnirs_df
def _reassign_channels(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Reassign column names into pairs: CH0 HbO, CH0 HbR, CH1 HbO, CH1 HbR, etc.
    If the file has an odd number of data columns (after removing Sample number/Event),
    logs a warning and renames the available columns.
    """
    cols = list(df.columns)

    sample_col = None
    event_col = None
    if "Sample number" in cols:
        sample_col = "Sample number"
        cols.remove("Sample number")
    if "Event" in cols:
        event_col = "Event"
        cols.remove("Event")

    if len(cols) == 0:
        logger.warning(f"No data columns to rename in '{file_path}'.")
        return df  # nothing to do

    if len(cols) % 2 != 0:
        logger.warning(
            f"Data columns in '{file_path}' are not an even number ({len(cols)}). "
            "Cannot reliably split into HbO/HbR pairs. We'll do the best we can."
        )

    num_channels = len(cols) // 2  # integer division
    new_data_cols = []
    # Use zero-based indexing: CH0, CH1, etc.
    for i in range(num_channels):
        new_data_cols.append(f"CH{i} HbO")
        new_data_cols.append(f"CH{i} HbR")

    extra_cols = cols[2 * num_channels:]  # any leftover columns

    new_cols_order = []
    if sample_col is not None:
        new_cols_order.append(sample_col)
    new_cols_order.extend(new_data_cols)
    new_cols_order.extend(extra_cols)
    if event_col is not None:
        new_cols_order.append(event_col)

    if len(new_cols_order) != len(df.columns):
        logger.warning(
            f"New column order has {len(new_cols_order)} columns but df has {len(df.columns)}. "
            f"Skipping renaming for '{file_path}'."
        )
        return df

    # Force assign the new column names
    df.columns = new_cols_order
    logger.debug(f"Reassigned columns to: {new_cols_order}")
    return df
