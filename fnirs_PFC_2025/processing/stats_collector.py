import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class StatsCollector:
    """
    FIXED StatsCollector - properly extracts timepoint from folder structure (Pre/Post folders)
    and correctly handles both single-batch and dual-batch output directory structures.
    Includes fallback broad search when strict path reconstruction fails.
    """

    def __init__(self, fs: float = 50.0):
        self.fs = fs
        logger.info("FIXED StatsCollector initialized - will properly extract timepoint from folder structure")

    def run_statistics(self,
                       processed_files: List[str],
                       input_base_dir: str,
                       output_base_dir: str,
                       file_type: str = "RAW") -> pd.DataFrame:
        """
        Run stats pipeline on processed files and return combined DataFrame.

        Args:
            file_type: "RAW" for raw concentrations or "ZSCORE" for Z-scored data
        """
        all_stats = []

        # Track what we've already processed to prevent ANY duplicates
        processed_stat_signatures = set()

        # Deduplicate input files by full path
        unique_processed_files = sorted(set(processed_files))

        if len(unique_processed_files) < len(processed_files):
            duplicates_removed = len(processed_files) - len(unique_processed_files)
            logger.warning(f" Removed {duplicates_removed} duplicate input file paths")

        logger.info(f" Processing {file_type} statistics for {len(unique_processed_files)} UNIQUE input files")
        logger.info(f"   Will look for processed versions in: {output_base_dir}")

        # Pre-build an index of all processed CSVs in the output tree for fallback search
        self._output_csv_index = self._build_output_csv_index(output_base_dir)
        logger.info(f"   Indexed {len(self._output_csv_index)} processed CSV files in output tree")

        files_found = 0
        files_not_found = 0
        not_found_list = []

        for file_path in unique_processed_files:
            # FIXED: Extract metadata properly from file path
            subject, timepoint = self._extract_metadata_from_path(file_path)

            logger.info(f" Processing input file: {os.path.basename(file_path)}")
            logger.info(f"   Subject: {subject}, Timepoint: {timepoint}")

            # Find matching processed files of the specified type
            all_processed_dfs = self._load_all_processed_files(
                file_path, input_base_dir, output_base_dir, file_type=file_type
            )

            if not all_processed_dfs:
                logger.warning(
                    f" Skipping stats for {os.path.basename(file_path)} - no {file_type} processed files found")
                files_not_found += 1
                not_found_list.append(os.path.basename(file_path))
                continue

            files_found += 1
            logger.info(f"   Found {len(all_processed_dfs)} {file_type} processed file version(s)")

            # Calculate statistics for EACH processed file version
            for idx, (processed_df, batch_type) in enumerate(all_processed_dfs, 1):
                logger.info(f"   Processing version {idx}/{len(all_processed_dfs)}: {batch_type}")

                # Calculate stats using the timepoint from the PATH (not from the file)
                stats = self._calculate_file_statistics(
                    processed_df, subject, timepoint, batch_type
                )

                if stats is not None:
                    # Create a comprehensive signature to detect duplicates
                    stat_signature = (
                        stats['Subject'],
                        stats['Timepoint'],
                        stats['Condition'],
                        stats['TaskType'],
                        stats['SQI_Filtering_Applied'],
                        round(stats['Overall grand oxy Mean'], 10),
                        round(stats['Overall grand deoxy Mean'], 10)
                    )

                    if stat_signature in processed_stat_signatures:
                        logger.warning(f"       DUPLICATE DETECTED - SKIPPING")
                        logger.warning(f"         Signature: {stat_signature}")
                        continue

                    processed_stat_signatures.add(stat_signature)
                    all_stats.append(stats)
                    logger.info(
                        f"       Stats added: {stats['Timepoint']} - {stats['Condition']} (SQI: {stats['SQI_Filtering_Applied']})")
                else:
                    logger.warning(f"       Stats calculation returned None")

        # Log summary of found vs not found
        logger.info(f" Stats collection summary: {files_found} files found, {files_not_found} files NOT found")
        if not_found_list:
            logger.warning(f" Files with no matching processed CSV ({len(not_found_list)}):")
            for fname in not_found_list[:20]:
                logger.warning(f"   - {fname}")
            if len(not_found_list) > 20:
                logger.warning(f"   ... and {len(not_found_list) - 20} more")

        if all_stats:
            logger.info(f" Successfully calculated {file_type} statistics for {len(all_stats)} UNIQUE file versions")
            logger.info(f"   From {len(unique_processed_files)} input files")
        else:
            logger.warning(f" No {file_type} statistics were calculated!")

        # Clean up index
        self._output_csv_index = None

        return self._create_stats_dataframe(all_stats)

    def _build_output_csv_index(self, output_base_dir: str) -> Dict[str, List[str]]:
        """
        Build an index of all processed CSV files in the output directory tree.
        Maps base_name (without _OD, without extension) to list of full paths.
        This enables fast fallback lookups when strict path reconstruction fails.
        """
        index = {}
        for root, _, files in os.walk(output_base_dir):
            for filename in files:
                if filename.endswith('.csv') and 'FULLY_PROCESSED' in filename:
                    full_path = os.path.join(root, filename)
                    # Extract the original base name from the processed filename
                    # e.g., "Walking_DT_OD.txt_FULLY_PROCESSED_RAW_with_SQI_filtering.csv"
                    # We want to index by everything before "_FULLY_PROCESSED"
                    # NOTE: file_processor may leave .txt in the output filename,
                    # so we normalize by stripping it
                    parts = filename.split('_FULLY_PROCESSED')
                    if parts:
                        base_key = parts[0].replace('.txt', '').replace('.TXT', '')
                        if base_key not in index:
                            index[base_key] = []
                        index[base_key].append(full_path)
        return index

    def _extract_metadata_from_path(self, file_path: str) -> Tuple[str, str]:
        """
        Extract subject and timepoint from file path structure.

        Handles multiple project conventions:
          1. Timepoint as subfolder:   .../OHSU_Turn_001/Pre/file.txt
          2. Timepoint in folder name: .../Long_058_V1/file.txt  (split on last _Vn)
          3. Timepoint in folder name: .../AUT_042_Pre/file.txt  (split on last _Pre/_Post)
        """
        path_parts = file_path.split(os.sep)

        subject = "Unknown"
        timepoint = "Unknown"

        # ── Strategy 1: Timepoint is its own subfolder (Pre/Post/Baseline) ──
        for i, part in enumerate(path_parts):
            part_lower = part.lower()
            if part_lower in ('pre', 'baseline', 'post', 'post-intervention'):
                timepoint = "Pre" if part_lower in ('pre', 'baseline') else "Post"
                if i > 0:
                    subject = path_parts[i - 1]
                logger.debug(f"Strategy 1 (subfolder): Subject={subject}, Timepoint={timepoint}")
                return subject, timepoint

        # ── Strategy 2: Timepoint embedded in a folder name ──
        # Look for patterns like Long_058_V1, AUT_042_Pre, Subject001_Post
        # Search from deepest directory upward (closest to the file first)
        dir_path = os.path.dirname(file_path)
        dir_parts = dir_path.split(os.sep)

        import re
        # Patterns to match timepoint suffixes at the end of a folder name
        timepoint_patterns = [
            # _V1, _V2, _V3 etc. — visit numbers
            (r'^(.+?)_(V\d+)$', None),
            # _Pre, _Post, _Baseline (case-insensitive)
            (r'^(.+?)_(Pre|Post|Baseline|Post-Intervention)$', {
                'pre': 'Pre', 'baseline': 'Pre',
                'post': 'Post', 'post-intervention': 'Post'
            }),
            # _T1, _T2, _T3 etc. — timepoint numbers
            (r'^(.+?)_(T\d+)$', None),
        ]

        for part in reversed(dir_parts):
            for pattern, mapping in timepoint_patterns:
                match = re.match(pattern, part, re.IGNORECASE)
                if match:
                    subject = match.group(1)
                    raw_timepoint = match.group(2)

                    if mapping:
                        timepoint = mapping.get(raw_timepoint.lower(), raw_timepoint)
                    else:
                        timepoint = raw_timepoint  # Keep as-is (V1, V2, T1, etc.)

                    logger.debug(f"Strategy 2 (embedded): Subject={subject}, Timepoint={timepoint}")
                    return subject, timepoint

        # ── Strategy 3: Fallback — use parent folder as subject, no timepoint ──
        if len(dir_parts) >= 1:
            subject = dir_parts[-1]
            logger.warning(f"No timepoint detected; using folder as subject: {subject}")

        logger.debug(f"Fallback: Subject={subject}, Timepoint={timepoint}")
        return subject, timepoint

    def _load_all_processed_files(self,
                                  file_path: str,
                                  input_base_dir: str,
                                  output_base_dir: str,
                                  file_type: str = "RAW") -> List[Tuple[pd.DataFrame, str]]:
        """
        Load processed file versions using the SAME path logic as file_processor._create_output_dir.

        FIXED: Includes fallback broad search when strict path reconstruction fails.

        Args:
            file_path: Original input file path
            input_base_dir: Base directory of input files
            output_base_dir: Base directory of output files (may be a batch-specific dir)
            file_type: "RAW" for raw concentrations or "ZSCORE" for Z-scored data
        """
        try:
            file_basename = os.path.basename(file_path)
            # Build candidate base names: try WITH _OD first (exact), then WITHOUT
            base_name_raw = file_basename.replace('.txt', '').replace('.TXT', '')
            base_name_no_od = base_name_raw.replace('_OD', '')

            # Use both as candidates for matching
            base_name_candidates = [base_name_raw]
            if base_name_no_od != base_name_raw:
                base_name_candidates.append(base_name_no_od)

            subject, timepoint = self._extract_metadata_from_path(file_path)

            logger.info(f" Searching for {file_type} versions of: {base_name_candidates}")
            logger.info(f"   Subject: {subject}, Timepoint: {timepoint}")

            # Determine search directories based on what actually exists
            batch_dirs = self._determine_batch_dirs(output_base_dir)

            # Determine file pattern
            if file_type == "RAW":
                search_pattern = "FULLY_PROCESSED_RAW"
            elif file_type == "ZSCORE":
                search_pattern = "FULLY_PROCESSED_ZSCORE"
            else:
                logger.error(f"Unknown file_type: {file_type}")
                return []

            # Reconstruct the output directory using the SAME logic
            # as file_processor._create_output_dir
            relative_path = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)
            logger.info(f"   Relative path from input base: {relative_path}")

            found_files_by_batch = {}

            # === ATTEMPT 1: Strict path reconstruction ===
            for batch_dir in batch_dirs:
                if not os.path.exists(batch_dir):
                    continue

                batch_type = self._determine_batch_type(batch_dir)

                if batch_type in found_files_by_batch:
                    continue

                expected_dir = os.path.join(batch_dir, relative_path)

                if not os.path.exists(expected_dir):
                    logger.debug(f"       Expected directory does not exist: {expected_dir}")
                    continue

                result = self._search_directory_for_file(
                    expected_dir, base_name_candidates, search_pattern, batch_type
                )
                if result:
                    actual_batch_type, df, full_path = result
                    found_files_by_batch[actual_batch_type] = (df, full_path)

            # === ATTEMPT 2: Fallback broad search using pre-built index ===
            if not found_files_by_batch and hasattr(self, '_output_csv_index') and self._output_csv_index:
                logger.info(f"   Strict path lookup failed, trying fallback index search for: {base_name_candidates}")

                matching_paths = []
                for candidate in base_name_candidates:
                    # Exact key match first
                    if candidate in self._output_csv_index:
                        matching_paths.extend(self._output_csv_index[candidate])

                if matching_paths:
                    logger.info(f"   Fallback found {len(matching_paths)} candidate(s)")

                    for full_path in matching_paths:
                        filename = os.path.basename(full_path)

                        # Verify it matches our search pattern
                        if search_pattern not in filename:
                            continue

                        # Verify precise match: the part before _FULLY_PROCESSED must
                        # exactly match one of our candidates (after normalizing .txt)
                        csv_base = filename.split('_FULLY_PROCESSED')[0]
                        csv_base = csv_base.replace('.txt', '').replace('.TXT', '')
                        if csv_base not in base_name_candidates:
                            continue

                        batch_type = self._determine_batch_type_from_path(full_path)

                        if batch_type in found_files_by_batch:
                            continue

                        try:
                            df = pd.read_csv(full_path)
                            required_columns = {'grand oxy', 'grand deoxy', 'Time (s)'}
                            if not required_columns.issubset(df.columns):
                                continue
                            if df.empty:
                                continue

                            # Detect actual SQI status from file content
                            actual_batch_type = batch_type
                            if 'SQI_Filtering_Applied' in df.columns:
                                sqi_applied = df['SQI_Filtering_Applied'].iloc[0]
                                actual_batch_type = "SQI_Filtered" if sqi_applied else "All_Channels"

                            found_files_by_batch[actual_batch_type] = (df, full_path)
                            logger.info(f"   FALLBACK FOUND: {filename} as {actual_batch_type}")

                        except Exception as e:
                            logger.error(f"   Fallback error reading {full_path}: {e}")
                            continue
                else:
                    logger.warning(f"   Fallback search also found no matches for: {base_name_candidates}")

            # Convert to list format
            found_files = [(df, batch_type) for batch_type, (df, path) in found_files_by_batch.items()]

            logger.info(f" Total loaded {file_type} file versions: {len(found_files)}")

            if not found_files:
                logger.error(f" No processed {file_type} files found for {base_name_candidates}")
                logger.error(f"   Searched relative path: {relative_path}")
                logger.error(f"   Search pattern: {search_pattern}")
                logger.error(f"   In dirs: {[os.path.basename(d) for d in batch_dirs]}")

            return found_files

        except Exception as e:
            logger.error(f" Error loading processed files for {file_path}: {str(e)}", exc_info=True)
            return []

    def _determine_batch_dirs(self, output_base_dir: str) -> List[str]:
        """Determine which batch directories to search."""
        if "batch_with_SQI_filtering" in output_base_dir or "batch_no_SQI_filtering" in output_base_dir:
            return [output_base_dir]

        batch_with_dir = os.path.join(output_base_dir, "batch_with_SQI_filtering")
        batch_no_dir = os.path.join(output_base_dir, "batch_no_SQI_filtering")

        if os.path.exists(batch_with_dir) or os.path.exists(batch_no_dir):
            return [batch_with_dir, batch_no_dir]

        return [output_base_dir]

    def _determine_batch_type(self, batch_dir: str) -> str:
        """Determine batch type from directory path."""
        batch_dir_basename = os.path.basename(batch_dir)
        if "with_SQI" in batch_dir_basename:
            return "SQI_Filtered"
        elif "no_SQI" in batch_dir_basename or "without_SQI" in batch_dir_basename:
            return "All_Channels"
        return "All_Channels"

    def _determine_batch_type_from_path(self, file_path: str) -> str:
        """Determine batch type from any part of a file path."""
        if "batch_with_SQI_filtering" in file_path or "with_SQI_filtering" in file_path:
            return "SQI_Filtered"
        elif "batch_no_SQI_filtering" in file_path or "without_SQI_filtering" in file_path:
            return "All_Channels"
        return "All_Channels"

    def _search_directory_for_file(self,
                                   search_dir: str,
                                   base_name_candidates: List[str],
                                   search_pattern: str,
                                   batch_type: str) -> Optional[Tuple[str, pd.DataFrame, str]]:
        """
        Search a directory tree for a matching processed CSV file.
        Uses PRECISE matching: the portion before _FULLY_PROCESSED must exactly
        match one of the candidate base names to avoid cross-matching
        (e.g., Walking_DT matching Walking_DT-AC).

        Args:
            search_dir: Directory to search
            base_name_candidates: List of possible base names (with and without _OD)
            search_pattern: Pattern to look for (e.g., "FULLY_PROCESSED_RAW")
            batch_type: Default batch type from directory name

        Returns:
            Tuple of (actual_batch_type, dataframe, full_path) or None
        """
        for root, dirs, files in os.walk(search_dir):
            for filename in files:
                if not (search_pattern in filename and filename.endswith(".csv")):
                    continue

                # PRECISE match: extract the base name from the CSV filename
                # e.g., "Walking_DT_OD.txt_FULLY_PROCESSED_RAW_without_SQI_filtering.csv"
                #    -> csv_base = "Walking_DT_OD" (after stripping .txt)
                csv_base = filename.split('_FULLY_PROCESSED')[0]
                csv_base = csv_base.replace('.txt', '').replace('.TXT', '')

                # Check if csv_base exactly matches one of our candidates
                if csv_base not in base_name_candidates:
                    continue

                full_path = os.path.join(root, filename)
                logger.info(f"       FOUND (precise match): {filename}")

                try:
                    df = pd.read_csv(full_path)

                    required_columns = {'grand oxy', 'grand deoxy', 'Time (s)'}
                    if not required_columns.issubset(df.columns):
                        missing = required_columns - set(df.columns)
                        logger.warning(f"          Missing columns {missing}")
                        continue

                    if df.empty:
                        logger.warning(f"          File is empty (0 rows): {filename}")
                        continue

                    # Detect SQI status from file content
                    actual_batch_type = batch_type
                    if 'SQI_Filtering_Applied' in df.columns:
                        sqi_applied = df['SQI_Filtering_Applied'].iloc[0]
                        if sqi_applied:
                            actual_batch_type = "SQI_Filtered"
                        else:
                            actual_batch_type = "All_Channels"

                    logger.info(f"          LOADED: {actual_batch_type} ({len(df)} rows)")
                    return (actual_batch_type, df, full_path)

                except Exception as e:
                    logger.error(f"          Error reading: {str(e)}")
                    continue

        return None

    def _calculate_file_statistics(self,
                                   processed_df: pd.DataFrame,
                                   subject: str,
                                   timepoint: str,
                                   batch_type: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        FIXED: Calculate statistics using timepoint from PATH, other metadata from FILE.

        Args:
            processed_df: The processed dataframe
            subject: Subject ID from path
            timepoint: Timepoint (Pre/Post) from path - THIS IS CRITICAL
            batch_type: SQI_Filtered or All_Channels
        """
        try:
            required_cols = {'grand oxy', 'grand deoxy', 'Time (s)'}
            if not required_cols.issubset(processed_df.columns):
                missing = required_cols - set(processed_df.columns)
                logger.warning(f"Missing required columns {missing} for {subject}")
                return None

            if len(processed_df) == 0:
                logger.warning(f"Empty DataFrame for {subject}")
                return None

            total_samples = len(processed_df)

            # Get condition and task type from the processed file
            actual_condition = processed_df['Condition'].iloc[0] if 'Condition' in processed_df.columns else "Unknown"
            actual_task_type = processed_df['TaskType'].iloc[0] if 'TaskType' in processed_df.columns else "Unknown"

            # Determine SQI filtering status from batch type
            sqi_filtering_applied = (batch_type == "SQI_Filtered")

            # CRITICAL: Use the timepoint from the PATH, not from the file
            logger.info(f" Calculating statistics:")
            logger.info(f"   Subject: {subject}")
            logger.info(f"   Timepoint (from path): {timepoint}")
            logger.info(f"   Condition: {actual_condition}")
            logger.info(f"   Task Type: {actual_task_type}")
            logger.info(f"   SQI Filtering: {sqi_filtering_applied}")

            # Calculate statistics on RAW concentration values
            oxy_overall = processed_df['grand oxy'].mean(skipna=True)
            deoxy_overall = processed_df['grand deoxy'].mean(skipna=True)

            # Calculate first/second half means
            oxy_first_half = processed_df.iloc[:total_samples // 2]['grand oxy'].mean(skipna=True)
            oxy_second_half = processed_df.iloc[total_samples // 2:]['grand oxy'].mean(skipna=True)

            deoxy_first_half = processed_df.iloc[:total_samples // 2]['grand deoxy'].mean(skipna=True)
            deoxy_second_half = processed_df.iloc[total_samples // 2:]['grand deoxy'].mean(skipna=True)

            # Initialize stats dictionary
            stats = {
                'Subject': subject,
                'Timepoint': timepoint,  # CRITICAL: Using timepoint from PATH
                'Condition': actual_condition,
                'TaskType': actual_task_type,
                'SQI_Filtering_Applied': sqi_filtering_applied,

                # HbO concentration statistics
                'Overall grand oxy Mean': oxy_overall,
                'First Half grand oxy Mean': oxy_first_half,
                'Second Half grand oxy Mean': oxy_second_half,

                # HbR concentration statistics
                'Overall grand deoxy Mean': deoxy_overall,
                'First Half grand deoxy Mean': deoxy_first_half,
                'Second Half grand deoxy Mean': deoxy_second_half,
            }

            logger.info(f" Stats calculated: {timepoint}, HbO={oxy_overall:.6f}, HbR={deoxy_overall:.6f}")
            return stats

        except Exception as e:
            logger.error(f"Error calculating stats for {subject}: {str(e)}", exc_info=True)
            return None

    def create_summary_sheets(self,
                              combined_stats_df: Optional[pd.DataFrame],
                              output_folder: str,
                              suffix: str = "") -> None:
        """Create summary sheets for different task types."""
        try:
            # FIXED: Guard against both None and empty DataFrame
            if combined_stats_df is None or combined_stats_df.empty:
                logger.warning(f"No statistics data to create summary sheets (suffix={suffix})")
                return

            if 'Condition' not in combined_stats_df.columns:
                logger.warning("No 'Condition' column in stats DataFrame — cannot create summaries")
                return

            unique_conditions = combined_stats_df['Condition'].unique()
            logger.info(f" Creating summaries for conditions: {list(unique_conditions)} (suffix={suffix})")

            for condition in unique_conditions:
                summary_df = self._filter_and_format_summary(combined_stats_df, condition)

                if not summary_df.empty:
                    safe_condition = condition.replace('_', '-').replace(' ', '-')
                    summary_filename = f'summary_{safe_condition}{suffix}.csv'
                    summary_path = os.path.join(output_folder, summary_filename)

                    summary_df.to_csv(summary_path, index=False)
                    logger.info(f" Saved summary ({len(summary_df)} rows): {summary_filename}")
                else:
                    logger.warning(f" No data for condition: {condition}")

            logger.info(" Summary sheets creation completed.")

        except Exception as e:
            logger.error(f"Error creating summary sheets: {str(e)}", exc_info=True)
            raise

    def _create_stats_dataframe(self, all_stats: List[Dict]) -> Optional[pd.DataFrame]:
        """Convert list of stats dictionaries to DataFrame. Returns None if no stats."""
        if not all_stats:
            logger.warning("No statistics collected - returning None")
            return None

        df = pd.DataFrame(all_stats)
        logger.info(f" Created stats DataFrame with {len(df)} rows and {len(df.columns)} columns")

        if 'Timepoint' in df.columns:
            timepoint_counts = df['Timepoint'].value_counts()
            logger.info(f" Timepoint distribution: {timepoint_counts.to_dict()}")

        if 'SQI_Filtering_Applied' in df.columns:
            sqi_counts = df['SQI_Filtering_Applied'].value_counts()
            logger.info(f" SQI Filtering distribution: {sqi_counts.to_dict()}")

        return df

    def _filter_and_format_summary(self,
                                   df: pd.DataFrame,
                                   condition: str) -> pd.DataFrame:
        """Filter and format summary for a specific condition."""
        filtered = df[df['Condition'] == condition].copy()

        if filtered.empty:
            return filtered

        # Use ONLY the columns that we actually calculate
        base_columns = [
            'Subject', 'Timepoint', 'TaskType', 'SQI_Filtering_Applied',
            'Overall grand oxy Mean', 'First Half grand oxy Mean', 'Second Half grand oxy Mean',
            'Overall grand deoxy Mean', 'First Half grand deoxy Mean', 'Second Half grand deoxy Mean'
        ]

        # Include columns that exist in the data
        columns_to_include = [col for col in base_columns if col in filtered.columns]

        return filtered[columns_to_include]

    def calculate_subject_y_limits(self,
                                   processed_files: List[str],
                                   output_base_dir: str,
                                   input_base_dir: str) -> Dict[str, Dict[str, float]]:
        """Calculate consistent y-axis limits per subject across all processed files (both SQI variants)."""
        logger.info("Calculating subject y-limits from processed files (including both SQI variants)")

        subject_data = {}

        for file_path in processed_files:
            try:
                # Extract metadata from path
                subject, timepoint = self._extract_metadata_from_path(file_path)

                # Load the processed files (will try both SQI variants)
                all_processed_dfs = self._load_all_processed_files(
                    file_path, input_base_dir, output_base_dir, file_type="RAW"
                )

                if not all_processed_dfs:
                    continue

                # Process each variant
                for processed_df, batch_type in all_processed_dfs:
                    # Extract signal values
                    if 'grand oxy' in processed_df.columns and 'grand deoxy' in processed_df.columns:
                        oxy_values = processed_df['grand oxy'].dropna()
                        deoxy_values = processed_df['grand deoxy'].dropna()

                        if subject not in subject_data:
                            subject_data[subject] = {'oxy': [], 'deoxy': []}

                        subject_data[subject]['oxy'].extend(oxy_values.tolist())
                        subject_data[subject]['deoxy'].extend(deoxy_values.tolist())

            except Exception as e:
                logger.warning(f" Error processing {file_path} for y-limits: {e}")
                continue

        # Calculate limits per subject
        subject_limits = {}
        for subject, data in subject_data.items():
            if data['oxy'] and data['deoxy']:
                all_values = data['oxy'] + data['deoxy']
                min_val = np.percentile(all_values, 1)  # 1st percentile
                max_val = np.percentile(all_values, 99)  # 99th percentile

                # Add some padding
                range_val = max_val - min_val
                padding = range_val * 0.1

                subject_limits[subject] = {
                    'raw_min': min_val - padding,
                    'raw_max': max_val + padding
                }

                logger.debug(f"Subject {subject}: y-limits [{min_val - padding:.3f}, {max_val + padding:.3f}]")

        logger.info(f" Calculated y-limits for {len(subject_limits)} subjects")
        return subject_limits
