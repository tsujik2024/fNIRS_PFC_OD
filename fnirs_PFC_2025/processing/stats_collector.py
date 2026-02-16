import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from fnirs_PFC_2025.read.loaders import read_txt_file

logger = logging.getLogger(__name__)


class StatsCollector:
    """
    Enhanced StatsCollector for multi-task fNIRS data with robust CV filtering support.
    """

    def __init__(self, fs: float = 50.0):
        self.fs = fs
        logger.info("Enhanced StatsCollector initialized for multi-task processing with robust CV filtering support")

    def run_statistics(self,
                       processed_files: List[str],
                       input_base_dir: str,
                       output_base_dir: str) -> pd.DataFrame:
        """Run stats pipeline on all processed files and return combined DataFrame."""
        all_stats = []

        logger.info(f"🔍 Processing statistics for {len(processed_files)} files")

        for file_path in processed_files:
            # Enhanced metadata extraction
            subject, timepoint, condition, task_type = self._extract_enhanced_metadata(file_path)

            # Load the corresponding processed file (handles both CV filtering variants)
            processed_df = self._load_processed_file_enhanced(
                file_path, input_base_dir, output_base_dir, task_type
            )

            if processed_df is None:
                logger.warning(f" Skipping stats for {os.path.basename(file_path)} - no processed file found")
                continue

            # Calculate statistics
            stats = self._calculate_file_statistics(processed_df, subject, timepoint, condition, task_type)
            if stats is not None:
                all_stats.append(stats)
                logger.info(f" Stats calculated for {subject} {timepoint} {condition}")

        if all_stats:
            logger.info(f" Successfully calculated statistics for {len(all_stats)} files")
        else:
            logger.warning(" No statistics were calculated!")

        return self._create_stats_dataframe(all_stats)

    def _extract_enhanced_metadata(self, file_path: str) -> tuple:
        """Enhanced metadata extraction for multi-task support with improved CV filtering detection."""
        path_parts = file_path.split(os.sep)
        file_basename = os.path.basename(file_path)

        # Extract subject
        subject = "Unknown"
        for part in path_parts:
            if "OHSU_Turn" in part or any(x in part for x in ["Subject", "subj", "sub-"]):
                subject = part
                break

        # Extract timepoint
        timepoint = "Unknown"
        for part in path_parts:
            if part.lower() in ["baseline", "pre", "post"] or part in ["Baseline", "Pre", "Post"]:
                timepoint = part
                break

        # Enhanced task type detection with improved CV filtering status
        task_type = "Unknown"
        condition = "Unknown"
        cv_filtering_applied = False

        filename_upper = file_basename.upper()

        # IMPROVED: More comprehensive CV filtering detection
        path_str = "/".join(path_parts).upper()

        # Look for CV filtering indicators (comprehensive check)
        cv_filtering_indicators = [
            "WITH_CV_FILTERING", "_WITH_CV_FILTERING",
            "CV_FILTERING", "CVFILTERING",
            "_CV_FILTERED", "CV_FILTERED",
            "WITH_CV", "_WITH_CV"
        ]

        no_cv_indicators = [
            "WITHOUT_CV_FILTERING", "_WITHOUT_CV_FILTERING",
            "NO_CV_FILTERING", "NOCVFILTERING",
            "_ALL_CHANNELS", "ALL_CHANNELS",
            "WITHOUT_CV", "_WITHOUT_CV"
        ]

        # Check path for CV filtering status (more reliable than filename alone)
        if any(indicator in path_str for indicator in cv_filtering_indicators):
            cv_filtering_applied = True
            logger.debug(f"CV filtering detected in path: {path_str}")
        elif any(indicator in path_str for indicator in no_cv_indicators):
            cv_filtering_applied = False
            logger.debug(f"No CV filtering detected in path: {path_str}")
        else:
            # Fallback: check filename
            if any(indicator in filename_upper for indicator in cv_filtering_indicators):
                cv_filtering_applied = True
                logger.debug(f"CV filtering detected in filename: {filename_upper}")
            elif any(indicator in filename_upper for indicator in no_cv_indicators):
                cv_filtering_applied = False
                logger.debug(f"No CV filtering detected in filename: {filename_upper}")
            else:
                # Final fallback: assume no CV filtering if unclear
                cv_filtering_applied = False
                logger.debug(f"CV filtering status unclear, defaulting to False for: {file_path}")

        # Check for specific task types
        if "FTURN" in filename_upper or "F_TURN" in filename_upper:
            task_type = "fTurn"
            base_condition = "EventTask_fTurn"
        elif "LSHAPE" in filename_upper or "L_SHAPE" in filename_upper:
            task_type = "LShape"
            base_condition = "EventTask_LShape"
        elif "FIGURE8" in filename_upper or "FIG8" in filename_upper:
            task_type = "Figure8"
            base_condition = "EventTask_Figure8"
        elif "OBSTACLE" in filename_upper:
            task_type = "Obstacle"
            base_condition = "EventTask_Obstacle"
        elif "NAVIGATION" in filename_upper or "NAV" in filename_upper:
            task_type = "Navigation"
            base_condition = "EventTask_Navigation"
        elif "DT" in filename_upper:
            task_type = "DT"
            base_condition = "LongWalk_DT"
        elif "ST" in filename_upper:
            task_type = "ST"
            base_condition = "LongWalk_ST"
        elif "WALK" in filename_upper:
            task_type = "LongWalk"
            base_condition = "LongWalk_LongWalk"
        else:
            base_condition = "Unknown_Unknown"

        # Append CV filtering status to condition
        if cv_filtering_applied:
            condition = f"{base_condition}_CV_Filtered"
        else:
            condition = f"{base_condition}_All_Channels"

        logger.debug(
            f"Extracted metadata: {subject}, {timepoint}, {condition}, {task_type}, CV_filtered={cv_filtering_applied}")
        return subject, timepoint, condition, task_type

    def _load_processed_file_enhanced(self,
                                      file_path: str,
                                      input_base_dir: str,
                                      output_base_dir: str,
                                      task_type: str) -> Optional[pd.DataFrame]:
        """Enhanced file loading that handles both CV filtering variants."""
        try:
            # Get the relative path structure from input to preserve folder hierarchy
            relative_path = os.path.relpath(os.path.dirname(file_path), input_base_dir)
            file_basename = os.path.basename(file_path)

            # Try both CV filtering variants
            cv_variants = [
                ("_with_CV_filtering", "_with_CV_filtering"),
                ("_without_CV_filtering", "_without_CV_filtering"),
                ("", "")  # Fallback for files without CV filtering suffix
            ]

            for folder_suffix, file_suffix in cv_variants:
                # Construct expected output path
                expected_processed_file = os.path.join(
                    output_base_dir,
                    relative_path,
                    f"{task_type}{folder_suffix}",
                    f"{file_basename}_FULLY_PROCESSED{file_suffix}.csv"
                )

                logger.debug(f"🔍 Looking for processed file: {expected_processed_file}")

                if os.path.exists(expected_processed_file):
                    logger.debug(f" Found: {expected_processed_file}")
                    df = pd.read_csv(expected_processed_file)

                    # Verify required columns
                    required_columns = {'grand oxy', 'grand deoxy', 'Time (s)'}
                    if not required_columns.issubset(df.columns):
                        missing = required_columns - set(df.columns)
                        logger.warning(f" Missing columns {missing} in {expected_processed_file}")
                        continue

                    return df

                # Fallback: search for the file in the task type directory
                task_dir = os.path.join(output_base_dir, relative_path, f"{task_type}{folder_suffix}")
                if os.path.exists(task_dir):
                    logger.debug(f"🔍 Searching in directory: {task_dir}")
                    for filename in os.listdir(task_dir):
                        if filename.endswith("_FULLY_PROCESSED.csv") and file_basename.replace('.txt', '') in filename:
                            fallback_file = os.path.join(task_dir, filename)
                            logger.info(f" Found fallback file: {fallback_file}")
                            return pd.read_csv(fallback_file)

            logger.warning(f" No processed file found for: {file_basename}")
            return None

        except Exception as e:
            logger.error(f" Error loading processed file for {file_path}: {str(e)}", exc_info=True)
            return None

    def _calculate_file_statistics(self,
                                   processed_df: pd.DataFrame,
                                   subject: str,
                                   timepoint: str,
                                   condition: str,
                                   task_type: str) -> Optional[Dict[str, Union[str, float]]]:
        """Calculate statistics for a single processed file with ROBUST CV filtering detection."""
        try:
            required_cols = {'grand oxy', 'grand deoxy'}
            if not required_cols.issubset(processed_df.columns):
                logger.warning(f"Missing required columns {required_cols - set(processed_df.columns)} for {subject}")
                return None

            if len(processed_df) == 0:
                logger.warning(f"Empty DataFrame for {subject}")
                return None

            total_samples = len(processed_df)

            # ROBUST CV FILTERING STATUS DETECTION (Multiple methods)
            cv_filtering_applied = False

            # Method 1: Read directly from CSV column (most reliable)
            if 'CV_Filtering_Applied' in processed_df.columns:
                cv_value = processed_df['CV_Filtering_Applied'].iloc[0]
                if isinstance(cv_value, (bool, np.bool_)):
                    cv_filtering_applied = bool(cv_value)
                elif isinstance(cv_value, str):
                    cv_filtering_applied = cv_value.lower() in ['true', '1', 'yes', 'on']
                else:
                    cv_filtering_applied = bool(cv_value)
                logger.debug(f"CV filtering status from CSV column: {cv_filtering_applied}")

            # Method 2: Infer from condition name as fallback
            elif "CV_Filtered" in condition:
                cv_filtering_applied = True
                logger.debug(f"CV filtering status inferred from condition '{condition}': True")

            elif "All_Channels" in condition:
                cv_filtering_applied = False
                logger.debug(f"CV filtering status inferred from condition '{condition}': False")

            # Method 3: Check if DataFrame has metadata columns that might indicate CV status
            elif 'Condition' in processed_df.columns:
                file_condition = str(processed_df['Condition'].iloc[0])
                if "CV_Filtered" in file_condition:
                    cv_filtering_applied = True
                elif "All_Channels" in file_condition:
                    cv_filtering_applied = False
                logger.debug(f"CV filtering status from file condition '{file_condition}': {cv_filtering_applied}")

            else:
                # Final fallback: default to False with warning
                cv_filtering_applied = False
                logger.warning(f" Could not determine CV filtering status for {subject}, defaulting to False")

            # Log the final determination
            logger.info(f"📊 Final CV filtering status for {subject}: {cv_filtering_applied}")

            # Basic HbO/HbR stats (all task types)
            oxy_overall = processed_df['grand oxy'].mean(skipna=True)
            deoxy_overall = processed_df['grand deoxy'].mean(skipna=True)

            oxy_first_half = processed_df.iloc[:total_samples // 2]['grand oxy'].mean(skipna=True)
            oxy_second_half = processed_df.iloc[total_samples // 2:]['grand oxy'].mean(skipna=True)

            deoxy_first_half = processed_df.iloc[:total_samples // 2]['grand deoxy'].mean(skipna=True)
            deoxy_second_half = processed_df.iloc[total_samples // 2:]['grand deoxy'].mean(skipna=True)

            # Initialize stats dictionary with CORRECTED CV filtering status
            stats = {
                'Subject': subject,
                'Timepoint': timepoint,
                'Condition': condition,
                'TaskType': task_type,
                'CV_Filtering_Applied': cv_filtering_applied,  # This should now be correct!
                # HbO
                'Overall grand oxy Mean': oxy_overall,
                'First Half grand oxy Mean': oxy_first_half,
                'Second Half grand oxy Mean': oxy_second_half,
                # HbR
                'Overall grand deoxy Mean': deoxy_overall,
                'First Half grand deoxy Mean': deoxy_first_half,
                'Second Half grand deoxy Mean': deoxy_second_half,
            }

            # Task-specific analysis
            if task_type in ['DT', 'ST', 'LongWalk']:
                # Long walk tasks may have TaskPhase column for walking/turning analysis
                if 'TaskPhase' in processed_df.columns:
                    walking = processed_df[processed_df['TaskPhase'] == 'Walking']
                    turning = processed_df[processed_df['TaskPhase'] == 'Turning']

                    if not walking.empty:
                        stats['Walking grand oxy Mean'] = walking['grand oxy'].mean(skipna=True)
                        stats['Walking grand deoxy Mean'] = walking['grand deoxy'].mean(skipna=True)

                    if not turning.empty:
                        stats['Turning grand oxy Mean'] = turning['grand oxy'].mean(skipna=True)
                        stats['Turning grand deoxy Mean'] = turning['grand deoxy'].mean(skipna=True)

                    # Calculate differences
                    if pd.notna(stats.get('Walking grand oxy Mean')) and pd.notna(stats.get('Turning grand oxy Mean')):
                        stats['Δ HbO Turning - Walking'] = (
                                stats['Turning grand oxy Mean'] - stats['Walking grand oxy Mean']
                        )

                    if pd.notna(stats.get('Walking grand deoxy Mean')) and pd.notna(
                            stats.get('Turning grand deoxy Mean')):
                        stats['Δ HbR Turning - Walking'] = (
                                stats['Turning grand deoxy Mean'] - stats['Walking grand deoxy Mean']
                        )

            elif task_type in ['fTurn', 'LShape', 'Figure8', 'Obstacle', 'Navigation']:
                # Event-dependent tasks - different analysis approach
                # For now, just use overall means as "task execution" means
                stats['Task Execution HbO Mean'] = oxy_overall
                stats['Task Execution HbR Mean'] = deoxy_overall

                # Could add event-based analysis here in the future
                logger.debug(f"Event-dependent task {task_type} - using overall means for task execution")

            return stats

        except Exception as e:
            logger.error(f"Error calculating stats for {subject}: {str(e)}", exc_info=True)
            return None

    def create_summary_sheets(self,
                              combined_stats_df: pd.DataFrame,
                              output_folder: str,
                              suffix: str = "") -> None:
        """Create summary sheets for different task types with optional suffix for CV filtering variants."""
        try:
            # Get unique conditions in the data
            unique_conditions = combined_stats_df['Condition'].unique()
            logger.info(f"📊 Creating summaries for conditions: {list(unique_conditions)}")

            # Create summaries for each condition
            for condition in unique_conditions:
                summary_df = self._filter_and_format_summary(combined_stats_df, condition)

                if not summary_df.empty:
                    # Create safe filename with suffix
                    safe_condition = condition.replace('_', '-').replace(' ', '-')
                    summary_filename = f'summary_{safe_condition}{suffix}.csv'
                    summary_path = os.path.join(output_folder, summary_filename)

                    summary_df.to_csv(summary_path, index=False)
                    logger.info(f" Saved summary: {summary_filename}")
                else:
                    logger.warning(f" No data for condition: {condition}")

            # Create CV filtering comparison summaries if both variants exist
            self._create_cv_comparison_summaries(combined_stats_df, output_folder, suffix)

            # Also create traditional ST/DT summaries if they exist
            st_conditions = [cond for cond in unique_conditions if 'LongWalk_ST' in cond]
            dt_conditions = [cond for cond in unique_conditions if 'LongWalk_DT' in cond]

            if st_conditions:
                for st_condition in st_conditions:
                    summary_ST = self._filter_and_format_summary(combined_stats_df, st_condition)
                    cv_status = "CV_Filtered" if "CV_Filtered" in st_condition else "All_Channels"
                    summary_ST.to_csv(os.path.join(output_folder, f'summary_ST_{cv_status}{suffix}.csv'), index=False)

            if dt_conditions:
                for dt_condition in dt_conditions:
                    summary_DT = self._filter_and_format_summary(combined_stats_df, dt_condition)
                    cv_status = "CV_Filtered" if "CV_Filtered" in dt_condition else "All_Channels"
                    summary_DT.to_csv(os.path.join(output_folder, f'summary_DT_{cv_status}{suffix}.csv'), index=False)

            logger.info(" Summary sheets creation completed.")

        except Exception as e:
            logger.error(f"Error creating summary sheets: {str(e)}", exc_info=True)
            raise

    def _create_cv_comparison_summaries(self, combined_stats_df: pd.DataFrame, output_folder: str,
                                        suffix: str = "") -> None:
        """Create comparison summaries between CV filtered and non-filtered results."""
        try:
            # Check if we have both CV filtered and non-filtered data
            cv_filtered_data = combined_stats_df[combined_stats_df['CV_Filtering_Applied'] == True]
            all_channels_data = combined_stats_df[combined_stats_df['CV_Filtering_Applied'] == False]

            if not cv_filtered_data.empty and not all_channels_data.empty:
                logger.info("📊 Creating CV filtering comparison summaries")

                # Get unique task types
                task_types = combined_stats_df['TaskType'].unique()

                for task_type in task_types:
                    cv_filtered_task = cv_filtered_data[cv_filtered_data['TaskType'] == task_type]
                    all_channels_task = all_channels_data[all_channels_data['TaskType'] == task_type]

                    if not cv_filtered_task.empty and not all_channels_task.empty:
                        # Create comparison DataFrame
                        comparison_data = []

                        # Get common subjects
                        common_subjects = set(cv_filtered_task['Subject']) & set(all_channels_task['Subject'])

                        for subject in common_subjects:
                            cv_subject = cv_filtered_task[cv_filtered_task['Subject'] == subject]
                            all_subject = all_channels_task[all_channels_task['Subject'] == subject]

                            if not cv_subject.empty and not all_subject.empty:
                                comparison_row = {
                                    'Subject': subject,
                                    'TaskType': task_type,
                                    'HbO_All_Channels': all_subject['Overall grand oxy Mean'].iloc[0],
                                    'HbO_CV_Filtered': cv_subject['Overall grand oxy Mean'].iloc[0],
                                    'HbR_All_Channels': all_subject['Overall grand deoxy Mean'].iloc[0],
                                    'HbR_CV_Filtered': cv_subject['Overall grand deoxy Mean'].iloc[0],
                                }

                                # Calculate differences
                                comparison_row['HbO_Difference'] = (
                                        comparison_row['HbO_CV_Filtered'] - comparison_row['HbO_All_Channels']
                                )
                                comparison_row['HbR_Difference'] = (
                                        comparison_row['HbR_CV_Filtered'] - comparison_row['HbR_All_Channels']
                                )

                                comparison_data.append(comparison_row)

                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            comparison_filename = f'CV_filtering_comparison_{task_type}{suffix}.csv'
                            comparison_path = os.path.join(output_folder, comparison_filename)
                            comparison_df.to_csv(comparison_path, index=False)
                            logger.info(f"✅ Saved CV comparison: {comparison_filename}")
            else:
                logger.info(" No CV filtering comparison created - missing one or both CV variants")
                logger.debug(f"CV filtered data: {len(cv_filtered_data)} rows")
                logger.debug(f"All channels data: {len(all_channels_data)} rows")

        except Exception as e:
            logger.error(f"Error creating CV comparison summaries: {str(e)}")

    def _create_stats_dataframe(self, all_stats: List[Dict]) -> pd.DataFrame:
        """Convert list of stats dictionaries to DataFrame."""
        if all_stats:
            df = pd.DataFrame(all_stats)
            logger.info(f"📊 Created stats DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Log CV filtering distribution for verification
            if 'CV_Filtering_Applied' in df.columns:
                cv_counts = df['CV_Filtering_Applied'].value_counts()
                logger.info(f"📊 CV Filtering distribution: {cv_counts.to_dict()}")

            return df
        else:
            logger.warning("No statistics collected - returning empty DataFrame")
            return pd.DataFrame(columns=[
                'Subject', 'Timepoint', 'Condition', 'TaskType', 'CV_Filtering_Applied',
                'Overall grand oxy Mean', 'First Half grand oxy Mean',
                'Second Half grand oxy Mean',
            ])

    def _filter_and_format_summary(self,
                                   df: pd.DataFrame,
                                   condition: str) -> pd.DataFrame:
        """Filter and format summary for a specific condition."""
        filtered = df[df['Condition'] == condition].copy()

        # Select appropriate columns based on what's available
        base_columns = ['Subject', 'Timepoint', 'TaskType', 'CV_Filtering_Applied', 'Overall grand oxy Mean']

        optional_columns = [
            'First Half grand oxy Mean',
            'Second Half grand oxy Mean',
            'Overall grand deoxy Mean',
            'Task Execution HbO Mean',
            'Walking grand oxy Mean',
            'Turning grand oxy Mean',
            'Δ HbO Turning - Walking'
        ]

        # Include columns that exist in the data
        columns_to_include = base_columns + [col for col in optional_columns if col in filtered.columns]

        return filtered[columns_to_include]

    def calculate_subject_y_limits(self,
                                   processed_files: List[str],
                                   output_base_dir: str,
                                   input_base_dir: str) -> Dict[str, Dict[str, float]]:
        """Calculate consistent y-axis limits per subject across all processed files (both CV variants)."""
        logger.info("🔍 Calculating subject y-limits from processed files (including both CV variants)")

        subject_data = {}

        for file_path in processed_files:
            try:
                subject, timepoint, condition, task_type = self._extract_enhanced_metadata(file_path)

                # Load the processed file (will try both CV variants)
                processed_df = self._load_processed_file_enhanced(
                    file_path, input_base_dir, output_base_dir, task_type
                )

                if processed_df is None:
                    continue

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