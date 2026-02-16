import os
import pandas as pd
import numpy as np
import logging
from fnirs_PFC_2025.processing.stats_collector import StatsCollector

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Set your path ===
output_base_dir = "/Users/tsujik/Documents/TurningOct23"


# === Create a modified StatsCollector for direct CSV processing ===
class DirectStatsCollector(StatsCollector):
    """Modified StatsCollector that works directly with CSV files."""

    def run_statistics_from_csvs(self, csv_files: list) -> pd.DataFrame:
        """Run stats pipeline directly on CSV files."""
        all_stats = []

        for csv_file in csv_files:
            # Extract metadata from CSV file path
            subject, timepoint, condition = self._extract_metadata_from_csv_path(csv_file)

            # Load the CSV directly
            try:
                processed_df = pd.read_csv(csv_file)
                logger.info(f"✅ Loaded CSV: {os.path.basename(csv_file)}")
            except Exception as e:
                logger.error(f"❌ Failed to load {csv_file}: {e}")
                continue

            # Calculate stats
            stats = self._calculate_file_statistics(processed_df, subject, timepoint, condition)
            if stats is not None:
                all_stats.append(stats)
                logger.info(f"✅ Calculated stats for {subject} {timepoint} {condition}")

        return self._create_stats_dataframe(all_stats)

    def _extract_metadata_from_csv_path(self, csv_file_path: str) -> tuple:
        """Extract subject, timepoint, and condition from CSV file path.

        Expected filename: Turn_120_LongWalk_ST.txt_FULLY_PROCESSED_without_CV_filtering.csv
        Expected structure: .../[Subject]/[Baseline or Pre or Post]/LongWalk_without_CV_filtering/file.csv
        """
        # Normalize the path
        csv_file_path = os.path.normpath(csv_file_path)
        path_parts = csv_file_path.split(os.sep)
        file_name = os.path.basename(csv_file_path)

        logger.info(f"Processing file: {file_name}")
        logger.debug(f"Full path: {csv_file_path}")
        logger.debug(f"Path parts: {path_parts}")

        # Extract subject from filename: Turn_120 -> Turn_120 or OHSU_Turn_120
        subject = "Unknown"
        try:
            # First, try to get subject from folder structure (preferred)
            if "LongWalk_without_CV_filtering" in path_parts:
                longwalk_idx = path_parts.index("LongWalk_without_CV_filtering")
                if longwalk_idx >= 2:
                    subject = path_parts[longwalk_idx - 2]  # Subject folder name (e.g., OHSU_Turn_120)

            # Fallback: Extract from filename if folder approach didn't work
            if subject == "Unknown" and file_name.startswith("Turn_"):
                # Extract Turn_XXX from filename
                parts = file_name.split("_")
                if len(parts) >= 2:
                    subject = f"OHSU_{parts[0]}_{parts[1]}"  # OHSU_Turn_120
        except Exception as e:
            logger.warning(f"Could not extract subject: {e}")

        # Extract timepoint from directory path
        timepoint = "Unknown"
        try:
            if "LongWalk_without_CV_filtering" in path_parts:
                longwalk_idx = path_parts.index("LongWalk_without_CV_filtering")
                if longwalk_idx >= 1:
                    timepoint_candidate = path_parts[longwalk_idx - 1]
                    # Check if it's a valid timepoint (case insensitive)
                    if timepoint_candidate.lower() in ["pre", "post", "baseline"]:
                        # Capitalize first letter for consistency
                        timepoint = timepoint_candidate.capitalize()
        except Exception as e:
            logger.warning(f"Could not extract timepoint from path: {e}")

        # Extract condition from filename - look for _ST or _DT pattern
        condition = "Unknown"
        if "_ST." in file_name or "_ST_" in file_name:
            condition = "LongWalk_ST"
        elif "_DT." in file_name or "_DT_" in file_name:
            condition = "LongWalk_DT"
        else:
            # Fallback: check path for ST/DT folders
            if "ST" in path_parts:
                condition = "LongWalk_ST"
            elif "DT" in path_parts:
                condition = "LongWalk_DT"

        logger.info(f"✓ Extracted: Subject={subject}, Timepoint={timepoint}, Condition={condition}")
        return subject, timepoint, condition

    def _calculate_file_statistics(self,
                                   processed_df: pd.DataFrame,
                                   subject: str,
                                   timepoint: str,
                                   condition: str) -> dict:
        """Calculate statistics using the actual column names from your CSV files."""
        try:
            # Check what columns we actually have
            logger.info(f"Available columns: {list(processed_df.columns)}")

            # Use the actual column names from your CSV
            oxy_col = 'grand oxy'
            deoxy_col = 'grand deoxy'

            if oxy_col not in processed_df.columns or deoxy_col not in processed_df.columns:
                logger.warning(f"Missing required columns for {subject}")
                return None

            if len(processed_df) == 0:
                logger.warning(f"Empty DataFrame for {subject}")
                return None

            total_samples = len(processed_df)

            # Overall means
            oxy_overall = processed_df[oxy_col].mean(skipna=True)
            deoxy_overall = processed_df[deoxy_col].mean(skipna=True)

            # First/Second half means
            oxy_first_half = processed_df.iloc[:total_samples // 2][oxy_col].mean(skipna=True)
            oxy_second_half = processed_df.iloc[total_samples // 2:][oxy_col].mean(skipna=True)
            deoxy_first_half = processed_df.iloc[:total_samples // 2][deoxy_col].mean(skipna=True)
            deoxy_second_half = processed_df.iloc[total_samples // 2:][deoxy_col].mean(skipna=True)

            # Initialize stats dictionary
            stats = {
                'Subject': subject,
                'Timepoint': timepoint,
                'Condition': condition,
                # HbO (Oxy)
                'Overall grand oxy Mean': oxy_overall,
                'First Half grand oxy Mean': oxy_first_half,
                'Second Half grand oxy Mean': oxy_second_half,
                'Walking grand oxy Mean': np.nan,
                'Turning grand oxy Mean': np.nan,
                'Δ HbO Turning - Walking': np.nan,
                # HbR (Deoxy) - Complete parallel structure
                'Overall grand deoxy Mean': deoxy_overall,
                'First Half grand deoxy Mean': deoxy_first_half,
                'Second Half grand deoxy Mean': deoxy_second_half,
                'Walking grand deoxy Mean': np.nan,
                'Turning grand deoxy Mean': np.nan,
                'Δ HbR Turning - Walking': np.nan,
            }

            # Check if we have pre-calculated walking/turning means for HbO
            if 'walking_mean_hbo' in processed_df.columns and not processed_df['walking_mean_hbo'].isna().all():
                stats['Walking grand oxy Mean'] = processed_df['walking_mean_hbo'].iloc[0]

            if 'turning_mean_hbo' in processed_df.columns and not processed_df['turning_mean_hbo'].isna().all():
                stats['Turning grand oxy Mean'] = processed_df['turning_mean_hbo'].iloc[0]

            # Check if we have pre-calculated walking/turning means for HbR
            if 'walking_mean_hbr' in processed_df.columns and not processed_df['walking_mean_hbr'].isna().all():
                stats['Walking grand deoxy Mean'] = processed_df['walking_mean_hbr'].iloc[0]

            if 'turning_mean_hbr' in processed_df.columns and not processed_df['turning_mean_hbr'].isna().all():
                stats['Turning grand deoxy Mean'] = processed_df['turning_mean_hbr'].iloc[0]

            # If we have TaskPhase column, calculate walking/turning means directly
            if 'TaskPhase' in processed_df.columns:
                logger.info(f"TaskPhase values for {subject}: {processed_df['TaskPhase'].value_counts(dropna=False)}")

                walking_data = processed_df[processed_df['TaskPhase'] == 'Walking']
                turning_data = processed_df[processed_df['TaskPhase'] == 'Turning']

                if not walking_data.empty:
                    # Calculate both HbO and HbR means for walking
                    stats['Walking grand oxy Mean'] = walking_data[oxy_col].mean(skipna=True)
                    stats['Walking grand deoxy Mean'] = walking_data[deoxy_col].mean(skipna=True)

                if not turning_data.empty:
                    # Calculate both HbO and HbR means for turning
                    stats['Turning grand oxy Mean'] = turning_data[oxy_col].mean(skipna=True)
                    stats['Turning grand deoxy Mean'] = turning_data[deoxy_col].mean(skipna=True)

            # Calculate differences for HbO
            if pd.notna(stats['Walking grand oxy Mean']) and pd.notna(stats['Turning grand oxy Mean']):
                stats['Δ HbO Turning - Walking'] = (
                        stats['Turning grand oxy Mean'] - stats['Walking grand oxy Mean']
                )

            # Calculate differences for HbR
            if pd.notna(stats['Walking grand deoxy Mean']) and pd.notna(stats['Turning grand deoxy Mean']):
                stats['Δ HbR Turning - Walking'] = (
                        stats['Turning grand deoxy Mean'] - stats['Walking grand deoxy Mean']
                )

            logger.info(
                f"✅ Stats calculated for {subject}: "
                f"Walking HbO={stats['Walking grand oxy Mean']:.3f}, Turning HbO={stats['Turning grand oxy Mean']:.3f}, "
                f"Walking HbR={stats['Walking grand deoxy Mean']:.3f}, Turning HbR={stats['Turning grand deoxy Mean']:.3f}"
            )
            return stats

        except Exception as e:
            logger.error(f"Error calculating stats for {subject}: {str(e)}", exc_info=True)
            return None


# === Initialize the direct stats collector ===
stats_collector = DirectStatsCollector()

# === Find all FULLY_PROCESSED.csv files ===
processed_csv_files = []
for root, _, files in os.walk(output_base_dir):
    for file in files:
        # Match both patterns: _FULLY_PROCESSED.csv and _FULLY_PROCESSED_without_CV_filtering.csv
        if ("_FULLY_PROCESSED" in file and file.endswith(".csv") and
                "bad_SCI" not in file and "all_subjects" not in file):
            full_path = os.path.join(root, file)
            processed_csv_files.append(full_path)

print(f"🔍 Found {len(processed_csv_files)} processed CSVs:")
for csv_file in processed_csv_files:
    print(f"  - {os.path.relpath(csv_file, output_base_dir)}")

if not processed_csv_files:
    print("❌ No processed CSV files found!")
    print(f"\nSearched in: {output_base_dir}")
    print("\nPlease verify:")
    print("  1. The path is correct")
    print("  2. Files end with '_FULLY_PROCESSED.csv'")
    print("  3. The directory structure is accessible")
    exit(1)

# === Run stats directly on CSV files ===
print("\n📊 Running statistics...")
stats_df = stats_collector.run_statistics_from_csvs(processed_csv_files)

if stats_df.empty:
    print("❌ No statistics were calculated!")
    print("\nPossible issues:")
    print("  - Check if CSV files have 'grand oxy' and 'grand deoxy' columns")
    print("  - Check the log messages above for specific errors")
else:
    print(f"✅ Calculated statistics for {len(stats_df)} files")

    # Display what we got
    print(f"\n📋 Data shape: {stats_df.shape}")
    print(f"Columns: {list(stats_df.columns)}")
    print(f"\nConditions found: {stats_df['Condition'].unique()}")
    print(f"Subjects found: {stats_df['Subject'].unique()}")
    print(f"Timepoints found: {stats_df['Timepoint'].unique()}")

    # === Save ALL results ===
    output_stats_file = os.path.join(output_base_dir, "all_subjects_statistics_no_CV_filtering.csv")
    try:
        stats_df.to_csv(output_stats_file, index=False)
        print(f"\n💾 Saved ALL data: {output_stats_file}")
    except Exception as e:
        print(f"❌ Failed to save all subjects file: {e}")

    # === Save SEPARATE files for ST and DT ===
    try:
        # Filter for ST condition
        st_df = stats_df[stats_df['Condition'] == 'LongWalk_ST'].copy()
        if not st_df.empty:
            st_output_file = os.path.join(output_base_dir, "statistics_ST_no_CV_filtering.csv")
            st_df.to_csv(st_output_file, index=False)
            print(f"💾 Saved ST data ({len(st_df)} rows): {st_output_file}")
        else:
            print("⚠️ No ST data found")

        # Filter for DT condition
        dt_df = stats_df[stats_df['Condition'] == 'LongWalk_DT'].copy()
        if not dt_df.empty:
            dt_output_file = os.path.join(output_base_dir, "statistics_DT_no_CV_filtering.csv")
            dt_df.to_csv(dt_output_file, index=False)
            print(f"💾 Saved DT data ({len(dt_df)} rows): {dt_output_file}")
        else:
            print("⚠️ No DT data found")

    except Exception as e:
        print(f"❌ Failed to create separate ST/DT files: {e}")

    # Create summary sheets (if the method exists)
    try:
        stats_collector.create_summary_sheets(stats_df, output_base_dir)
        print("💾 Created summary sheets: summary_ST.csv and summary_DT.csv")
    except Exception as e:
        print(f"⚠️ Failed to create summary sheets: {e}")

    # Display preview showing both oxy and deoxy data
    print(f"\n📋 Statistics Preview (first 10 rows):")
    preview_cols = ['Subject', 'Timepoint', 'Condition', 'Overall grand oxy Mean', 'Overall grand deoxy Mean']
    print(stats_df[preview_cols].head(10))

    # Show summary statistics
    print(f"\n📊 Summary by Condition:")
    print(stats_df.groupby('Condition')[['Overall grand oxy Mean', 'Overall grand deoxy Mean']].describe())

print("\n✅ Stats-only processing complete!")