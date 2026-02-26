import logging
import pandas as pd
import numpy as np
import re
import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust logging level as needed


def read_txt_file(file_path: str) -> dict:
    """
    Reads an OxySoft .txt export with OD data and returns:
      - 'metadata': parsed header info (+ 'Sample Rate (Hz)', wavelengths)
      - 'data': cleaned DataFrame with renamed channels in format 'CH{i}_WL{wavelength}'
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
    wavelengths_parsed = False

    try:
        for i in range(min(50, len(rows))):
            row = rows[i]
            if not row:
                continue

            k = row[0].strip() if row[0] else ""
            kl = k.lower()

            # Handle rows with key-value pairs
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
                    except:
                        pass

            # Handle wavelength section (single column header)
            if "light source wavelengths" in kl and not wavelengths_parsed:
                logger.info(f"Found wavelength header at row {i}: {row}")
                metadata['Wavelengths'] = {}

                # Skip the header row (device | index | wavelength)
                # Start parsing from the next row
                for j in range(i + 2, min(i + 20, len(rows))):  # i+2 to skip both header rows
                    row_j = rows[j]

                    # Stop if we hit an empty row or new section
                    if not row_j or (len(row_j) == 1 and row_j[0].strip() == ''):
                        logger.info(f"Stopped wavelength parsing at row {j} (empty row)")
                        break

                    # Parse wavelength data: device | index | wavelength | nm
                    if len(row_j) >= 3:
                        try:
                            # row_j[0] = device id (always '1' in your case)
                            # row_j[1] = light source index (1-16)
                            # row_j[2] = wavelength (e.g., '846')
                            # row_j[3] = 'nm' (optional)

                            device_id = row_j[0].strip()
                            if not device_id.isdigit():
                                # Not a data row, stop parsing
                                break

                            src_idx = int(row_j[1].strip())
                            wavelength = int(row_j[2].strip())
                            metadata['Wavelengths'][src_idx] = wavelength
                            logger.debug(f"Parsed: Device {device_id}, LS{src_idx} = {wavelength}nm")

                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse wavelength row {j}: {row_j} - {e}")
                            break

                wavelengths_parsed = True
                logger.info(f"Total wavelengths parsed: {len(metadata.get('Wavelengths', {}))}")
                logger.info(f"Wavelength map: {metadata.get('Wavelengths', {})}")

    except Exception as e:
        logger.warning(f" Metadata parsing error in {file_path}: {e}")

    # Verify wavelengths were found
    if 'Wavelengths' not in metadata or not metadata['Wavelengths']:
        logger.error(f"No wavelengths found in metadata for {file_path}")
        logger.error("This will cause all channels to be UNMAPPED")
    else:
        logger.info(f"Successfully parsed {len(metadata['Wavelengths'])} wavelengths")

    # sample rate + header row indices
    start_idx = end_idx = None
    sample_rate = metadata.get('Sample Rate (Hz)')

    # First pass: find sample rate
    for idx, row in enumerate(rows):
        if len(row) > 1 and "Datafile sample rate:" in row[0]:
            try:
                sample_rate = float(row[1])
                metadata['Sample Rate (Hz)'] = sample_rate
            except Exception:
                logger.warning(f" Could not parse sample rate in {file_path}")

    # Second pass: find column header rows
    # Look for rows containing (Sample number) and (Event) markers
    for idx, row in enumerate(rows):
        # Convert all cells to strings and check for markers
        row_str = '\t'.join([str(cell) for cell in row])

        if "(Sample number)" in row_str and start_idx is None:
            start_idx = idx
            logger.debug(f"Found (Sample number) marker at row {idx}: {row}")

        if "(Event)" in row_str:
            end_idx = idx
            logger.debug(f"Found (Event) marker at row {idx}: {row}")

    if start_idx is None or end_idx is None:
        logger.error(f" Could not identify column header rows in {file_path}")
        logger.error(f" start_idx={start_idx}, end_idx={end_idx}")
        # Debug output
        logger.debug("Searching for markers in file. Showing rows 40-70:")
        for i in range(40, min(70, len(rows))):
            row_str = '\t'.join([str(cell) for cell in rows[i]])
            if "(Sample number)" in row_str or "(Event)" in row_str:
                logger.debug(f"*** Row {i}: {rows[i]}")
            else:
                logger.debug(f"Row {i}: {rows[i]}")
        return None

    if sample_rate is None:
        logger.warning(f" Could not determine sample rate for {file_path}")

    metadata['Export file'] = file_path

    # Subject fallback
    if 'Subject Public ID' not in metadata or not metadata['Subject Public ID']:
        m = re.search(r'(OHSU[_-]?Turn[_-]?\d+|Turn[_-]?\d+|sub[-_]\w+)', file_path, flags=re.IGNORECASE)
        metadata['Subject Public ID'] = m.group(1) if m else None
        if metadata['Subject Public ID']:
            logger.warning(f" Inferred Subject ID from filename: {metadata['Subject Public ID']}")

    # Date fallback
    if 'Record Date/Time' not in metadata and 'Export date' in metadata:
        metadata['Record Date/Time'] = metadata['Export date']
        logger.warning(f" Using Export date as Record Date/Time for {file_path}")

    # ---- column labels ----
    try:
        # In OD exports, the column structure spans multiple rows
        # We need to extract the labels from the appropriate column
        # Based on the Legend structure, column labels are in column index 1
        col_labels = []
        for idx in range(start_idx, end_idx + 1):
            row = rows[idx]
            # The column label is typically in index 1
            if len(row) > 1:
                col_labels.append(row[1])
            elif len(row) == 1:
                col_labels.append(row[0])
            else:
                logger.warning(f"Unexpected row structure at {idx}: {row}")
                col_labels.append("")

        logger.debug(f"Extracted {len(col_labels)} raw column labels: {col_labels}")

    except Exception as e:
        logger.error(f" Failed to parse column labels in {file_path}: {e}")
        return None

    # Process OD column labels
    col_labels = _process_od_column_labels(col_labels, metadata, file_path)

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

    logger.debug(f"Attempting to convert {len(num_cols)} columns to numeric")

    # Convert each column individually with extensive error handling
    for col in num_cols:
        try:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            # Check if the column data is accessible
            col_data = df[col]
            logger.debug(
                f"Processing column '{col}': type = {type(col_data)}, shape = {getattr(col_data, 'shape', 'N/A')}")

            # Ensure we have a Series or convertible object
            if not isinstance(col_data, (pd.Series, list, tuple, np.ndarray)):
                logger.warning(f"Column '{col}' is not a Series (type: {type(col_data)}), converting to list")
                col_data = col_data.tolist() if hasattr(col_data, 'tolist') else list(col_data)

            # Convert to numeric
            df[col] = pd.to_numeric(col_data, errors='coerce')
            logger.debug(f"Successfully converted column '{col}' to numeric")

        except Exception as e:
            logger.error(f"Failed to convert column '{col}' to numeric: {e}")
            logger.error(
                f"Column type: {type(df[col])}, Column content sample: {df[col].iloc[:3] if hasattr(df[col], 'iloc') else 'N/A'}")
            # Keep the column as string if conversion fails

    # Clean Event once
    if 'Event' in df.columns:
        df['Event'] = df['Event'].astype(str).str.strip()
        df['Event'] = df['Event'].replace({'': np.nan, 'nan': np.nan})

    # Drop first two rows (OxySoft artifacts) only if there's room
    if len(df) > 3:
        df = df.iloc[3:].reset_index(drop=True)

    # Ensure Sample number is int if present
    if 'Sample number' in df.columns:
        df['Sample number'] = pd.to_numeric(df['Sample number'], errors='coerce').fillna(0).astype(int)

    # Final reorganization of OD columns
    df = _reassign_channels_od(df, metadata, file_path)

    if df is None or df.empty:
        logger.error(f" DataFrame empty after channel reassignment in {file_path}")
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

      Column 1  -> (Sample number)
      Columns 2–17 -> Light sources 1–16
      Column 18 -> ADC
      Column 19 -> (Event)
    """
    wavelength_map = metadata.get('Wavelengths', {})

    logger.info(f"Wavelength map from metadata: {wavelength_map}")

    # Channel pairing based on OctaMon 2x3 channel + 2x1 SSC
    # channel_number: (light_source_~850nm, light_source_~760nm)
    channel_pairing = {
        0: (1, 2),  # CH0: LS1 (846 nm) + LS2 (757 nm)
        1: (3, 4),  # CH1: LS3 (848 nm) + LS4 (759 nm)
        2: (5, 6),  # CH2: LS5 (848 nm) + LS6 (756 nm)
        3: (7, 8),  # CH3: LS7 (848 nm) + LS8 (758 nm)
        4: (9, 10),  # CH4: LS9 (847 nm) + LS10 (757 nm)
        5: (11, 12),  # CH5: LS11 (848 nm) + LS12 (758 nm)
        6: (13, 14),  # CH6: LS13 (846 nm) + LS14 (759 nm)
        7: (15, 16),  # CH7: LS15 (846 nm) + LS16 (759 nm)
    }

    new_labels = []

    for i, label in enumerate(col_labels):
        # i is 0-based index in col_labels
        # Column numbers in legend are 1-based
        column_number = i + 1  # Convert to 1-based

        if column_number == 1:
            # Column 1: (Sample number)
            new_labels.append('Sample number')
        elif column_number == 18:
            # Column 18: ADC
            new_labels.append('ADC')
        elif column_number == 19:
            # Column 19: (Event)
            new_labels.append('Event')
        elif 2 <= column_number <= 17:
            # Columns 2–17 map to light sources 1–16
            light_source_idx = column_number - 1  # Column 2 -> LS1, Column 3 -> LS2, ..., Column 17 -> LS16

            channel_found = None
            wavelength = None

            for ch_num, (ls_850, ls_760) in channel_pairing.items():
                if light_source_idx == ls_850 or light_source_idx == ls_760:
                    channel_found = ch_num
                    # Prefer actual wavelength from metadata
                    wavelength = wavelength_map.get(light_source_idx)
                    if wavelength is None:
                        logger.warning(
                            f"No wavelength found for light source {light_source_idx} in metadata for {file_path}")
                        new_labels.append(f"UNMAPPED_{light_source_idx}")
                        break
                    new_label = f"CH{channel_found}_WL{wavelength}"
                    new_labels.append(new_label)
                    logger.debug(
                        f"Column {column_number} -> Light source {light_source_idx} ({wavelength}nm) → {new_label}"
                    )
                    break

            if channel_found is None and wavelength is not None:
                # Wavelength was found but no channel pairing matched
                logger.warning(
                    f"Could not map light source {light_source_idx} to a channel "
                    f"in {file_path}; keeping as UNMAPPED_{light_source_idx}"
                )
                new_labels.append(f"UNMAPPED_{light_source_idx}")
        else:
            logger.warning(f"Unexpected column number {column_number} in {file_path}")
            new_labels.append(f"UNKNOWN_COL_{column_number}")

    # Check for duplicates
    if len(new_labels) != len(set(new_labels)):
        from collections import Counter
        duplicates = [item for item, count in Counter(new_labels).items() if count > 1]
        logger.error(f"DUPLICATE COLUMN NAMES: {duplicates}")

    logger.info(f"Final column labels: {new_labels}")
    return new_labels

def _reassign_channels_od(df: pd.DataFrame, metadata: dict, file_path: str) -> pd.DataFrame:
    """
    For OD data, ensure proper channel naming and organization.
    """
    cols = list(df.columns)

    # Keep standard columns
    standard_cols = ['Sample number', 'ADC', 'Event']
    data_cols = [col for col in cols if col not in standard_cols]

    # Sort data columns by channel and wavelength for consistency
    def sort_key(col):
        if col.startswith('CH') and 'WL' in col:
            try:
                ch_num = int(col.split('CH')[1].split('_')[0])
                wl_num = int(col.split('WL')[1])
                return (ch_num, wl_num)
            except:
                return (999, 999)  # Put problematic columns at end
        return (1000, 1000)

    sorted_data_cols = sorted(data_cols, key=sort_key)

    # Reconstruct column order
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
        logger.debug(f"Reorganized OD columns for {file_path}")
    else:
        logger.warning(f"Column count mismatch in OD reassignment for {file_path}")

    return df


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
        # Prefer "Start of measurement"
        if kl.startswith("start of measurement"):
            metadata['Measurement Start'] = val
            logger.info(f"Using 'Start of measurement' for date: {val}")
        # Then capture Record Date/Time
        elif kl.startswith("record date/time"):
            metadata['Record Date/Time'] = val
        #  Then Export date
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
    Returns a DataFrame with columns for OD data, Sample number, Event, etc.
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

    # Clean up column labels for OD data
    for idx, label in enumerate(col_labels):
        # For OD data with wavelength information
        if any(wl in label for wl in ["730", "735", "740", "745", "750", "755", "760",
                                      "765", "770", "805", "810", "815", "820", "825",
                                      "830", "835", "840", "845", "850", "855", "860", "865"]):
            # Keep wavelength information but clean up formatting
            col_labels[idx] = re.sub(r'\([^)]*\)', '', label).strip()
        elif "(Sample number)" in label or "(Event)" in label:
            parts = label.split('(')
            if len(parts) == 2:
                col_labels[idx] = parts[1].split(')')[0]
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
        df['Event'] = df['Event'].str.strip()  # Force everything to string
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
