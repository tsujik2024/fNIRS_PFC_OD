import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple, Callable, List
import logging

# Import processing steps
from fnirs_PFC_2025.preprocessing.fir_filter import fir_filter
from fnirs_PFC_2025.preprocessing.short_channel_regression import scr_regression
from fnirs_PFC_2025.preprocessing.tddr import tddr
from fnirs_PFC_2025.preprocessing.baseline_correction import baseline_subtraction
from fnirs_PFC_2025.preprocessing.average_channels import average_channels
# Import plotting functions
from fnirs_PFC_2025.viz.plots import plot_channels_separately, plot_overall_signals

logger = logging.getLogger(__name__)
plt.ioff()  # Non-interactive backend


class FileProcessor:
    """Handles processing of individual fNIRS files through the complete pipeline with CV filtering and post-walking trimming options."""

    def __init__(self, fs: float = 50.0, cv_threshold: float = 50.0,
                 enable_cv_filtering: bool = False,
                 post_walking_trim_seconds: float = 3.0):
        """
        Initialize processor with parameters.

        Args:
            fs: Sampling frequency in Hz
            cv_threshold: CV threshold for quality assessment (default 50%)
            enable_cv_filtering: If True, exclude channels with CV > threshold from analysis
            post_walking_trim_seconds: Seconds to trim after walking start event for quality control (default 3.0)
        """
        self.fs = fs
        self.cv_threshold = cv_threshold
        self.enable_cv_filtering = enable_cv_filtering
        self.post_walking_trim_seconds = post_walking_trim_seconds

        # Define task types and their walking start events
        self.task_walking_events = {
            # Long walk tasks - look for walking start
            'DT': ['W1', 'WALK', 'START_WALK', 'WALKING'],
            'ST': ['W1', 'WALK', 'START_WALK', 'WALKING'],
            'LongWalk': ['W1', 'WALK', 'START_WALK', 'WALKING'],

            # Event-dependent tasks - look for task start (after baseline)
            'fTurn': ['S2', 'START', 'TASK_START', 'GO'],
            'LShape': ['W1', 'WALK', 'START_WALK', 'S3'],  # W1 after S2 baseline
            'Figure8': ['S2', 'START', 'TASK_START', 'GO'],
            'Obstacle': ['S2', 'START', 'TASK_START', 'GO'],
            'Navigation': ['S2', 'START', 'TASK_START', 'GO'],
        }

        # Define task types and their requirements
        self.task_types = {
            # Long walk tasks - can use time-based fallback
            'DT': {'type': 'long_walk', 'min_events': 0},
            'ST': {'type': 'long_walk', 'min_events': 0},
            'LongWalk': {'type': 'long_walk', 'min_events': 0},

            # Event-dependent tasks - require sufficient event markers
            'fTurn': {'type': 'event_dependent', 'min_events': 3},
            'LShape': {'type': 'event_dependent', 'min_events': 3},
            'Figure8': {'type': 'event_dependent', 'min_events': 3},
            'Obstacle': {'type': 'event_dependent', 'min_events': 3},
            'Navigation': {'type': 'event_dependent', 'min_events': 2},
        }

        filter_status = "ENABLED" if enable_cv_filtering else "DISABLED"
        logger.info(f"Initialized FileProcessor (fs={fs}, CV threshold={cv_threshold}%, "
                    f"CV filtering={filter_status}, post-walking trim={post_walking_trim_seconds}s)")

    def process_file(self,
                     file_path: str,
                     output_base_dir: str,
                     input_base_dir: str,
                     subject_y_limits: Optional[Dict] = None,
                     read_file_func: Callable = None,
                     baseline_duration: Optional[float] = None,
                     ) -> Optional[pd.DataFrame]:
        """
        Process a single fNIRS file through the complete pipeline with optional CV filtering and post-walking trimming.
        """
        try:
            self._event_index_remap = None
            # ─── Setup paths & names ───────────────────────────
            output_dir = self._create_output_dir(output_base_dir, input_base_dir, file_path)
            file_basename = os.path.basename(file_path)
            subject = self._extract_subject(file_path)
            raw_limits = self._get_plotting_limits(subject, subject_y_limits)

            logger.info(f" Starting: {file_path} (CV filtering: {'ON' if self.enable_cv_filtering else 'OFF'}, "
                        f"post-walking trim: {self.post_walking_trim_seconds}s)")

            # ─── 1) Load raw data ───────────────────────────────
            data_dict = read_file_func(file_path)
            if not data_dict or 'data' not in data_dict:
                logger.error(f" read_file_func failed or returned no 'data' for: {file_path}")
                return None

            raw_df = data_dict['data']
            if raw_df is None or not isinstance(raw_df, pd.DataFrame):
                logger.error(f" Invalid DataFrame returned for: {file_path}")
                return None

            data = self._prepare_data(raw_df)
            if data is None or data.empty:
                logger.error(f" Prepared data is empty for: {file_path}")
                return None

            logger.debug(f" Loaded {len(data)} rows × {len(data.columns)} cols")

            # ─── 2) Metadata ────────────────────────────────────
            metadata = data_dict.get('metadata', {})
            subject_id = metadata.get('Subject Public ID')
            record_date = metadata.get('Record Date/Time')
            sample_rate = int(self.fs)

            # ─── 3) Determine task type ─────────────────────────
            task_type = self._determine_task_type(file_basename)
            logger.debug(f"Detected task type: {task_type}")

            task_config = self.task_types.get(task_type, {'type': 'unknown', 'min_events': 2})

            # ─── 4) Extract and clean events ──────────────────────────────
            events = self._extract_and_clean_events(data_dict, data)

            # ─── 5) Validate task requirements ──────────────────────────────
            if not self._validate_task_requirements(task_type, task_config, events, file_basename):
                logger.error(f" Task requirements not met for {file_basename}")
                return None

            # ─── 6) Plot raw data ───────────────────────────────
            self._plot_raw_data(
                data=data,
                output_dir=output_dir,
                file_basename=file_basename,
                subject=subject,
                global_ylim=raw_limits,
                condition=task_type,
                events=events
            )

            # ─── 7) Processing pipeline with CV filtering ───────
            processed_data = self._process_pipeline_stages(
                data=data,
                output_dir=output_dir,
                file_basename=file_basename,
                events=events,
                task_type=task_type
            )
            if processed_data is None or processed_data.empty:
                logger.error(f" Pipeline stages returned empty for: {file_path}")
                return None

            # ─── 8) Final outputs with post-walking trimming ───────────────────────────────
            final_df = self._finalize_outputs(
                processed_data, output_dir, file_basename, subject,
                task_type, task_config, events
            )

            if final_df is None or final_df.empty:
                logger.error(f" _finalize_outputs failed for: {file_path}")
                return None

            logger.info(f" Finished processing: {file_path}")
            return final_df

        except Exception as e:
            logger.error(f" Exception in process_file for {file_path}: {e}", exc_info=True)
            return None

    def _process_pipeline_stages(self, data: pd.DataFrame,
                                 output_dir: str, file_basename: str,
                                 events: Optional[pd.DataFrame] = None,
                                 task_type: str = None) -> Optional[pd.DataFrame]:
        """Process data through all pipeline stages with optional CV filtering."""
        # Select only numeric HbO/HbR columns for processing
        signal_cols = [col for col in data.columns
                       if ('HbO' in col or 'HbR' in col or 'O2Hb' in col or 'HHb' in col)
                       and pd.api.types.is_numeric_dtype(data[col])]
        if not signal_cols:
            logger.error(" No fNIRS signal columns found")
            return None

        signal_data = data[signal_cols].copy()

        # Split by chromophore
        o2hb_cols = [c for c in signal_cols if ('O2Hb' in c or 'HbO' in c)]
        hhb_cols = [c for c in signal_cols if ('HHb' in c or 'HbR' in c)]

        # Filtering for CV estimation (separate from main processing)
        filtered_for_cv = fir_filter(signal_data, order=1000, Wn=[0.01, 0.1], fs=int(self.fs))

        # Calculate CV and determine channels to exclude
        excluded_channels = self._calculate_cv_quality_and_filter(
            filtered_for_cv, o2hb_cols, hhb_cols, output_dir, file_basename
        )

        # Build the working dataframe that will flow through SCR/TDDR/etc.
        if self.enable_cv_filtering and excluded_channels:
            logger.info(f"🔄 CV filtering enabled: Excluding {len(excluded_channels)} channels")
            working_data = data.drop(columns=[c for c in excluded_channels if c in data.columns]).copy()
            # Update list of signal columns accordingly
            signal_cols = [c for c in signal_cols if c not in excluded_channels]
            if not signal_cols:
                logger.error(" All channels excluded by CV filtering!")
                return None
        else:
            logger.info(f" CV filtering disabled: Using all {len(signal_cols)} channels")
            working_data = data.copy()

        # Recompute signal slice after exclusion
        signal_slice = working_data[signal_cols].copy()

        # ─── MAIN PROCESSING PIPELINE ──────────────────────────────────

        # 1) SCR
        scr_data = self._apply_scr(signal_slice)

        # 2) TDDR
        tddr_data = self._apply_tddr(scr_data)

        # 3) FIR FILTERING (MISSING STEP ADDED HERE)
        fir_filtered_data = self._apply_fir_filter(tddr_data)

        # Merge back into working_data (only for processed channels)
        for col in fir_filtered_data.columns:
            if col in working_data.columns:
                working_data[col] = fir_filtered_data[col]

        logger.info(f"Processing {len(fir_filtered_data.columns)} channels: {list(fir_filtered_data.columns)}")

        # 4) Baseline Correction (may return None on failure)
        return self._apply_baseline_correction(working_data, events, task_type)

    def _apply_fir_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply FIR filter to the data with appropriate parameters."""
        try:
            return fir_filter(data, order=1000, Wn=[0.01, 0.1], fs=int(self.fs))
        except Exception as e:
            logger.warning(f"FIR filtering failed: {str(e)}")
            return data

    def _apply_post_event_trimming(self, data: pd.DataFrame, events: pd.DataFrame,
                                   task_type: str) -> pd.DataFrame:
        """
        Apply trimming after walking/task start event for quality control and
        set a mapping to remap event sample indices to the new timeline.
        """
        self._event_index_remap = None  # reset per file

        if self.post_walking_trim_seconds <= 0:
            return data

        if events is None or events.empty:
            logger.warning(" No events available for post-event trimming")
            return data

        walking_start_sample = self._find_walking_start_event(events, task_type)
        if walking_start_sample is None:
            logger.warning(f" No walking start event found for {task_type}, skipping post-event trimming")
            return data

        trim_samples = int(self.post_walking_trim_seconds * self.fs)
        trim_start_sample = walking_start_sample + trim_samples

        if trim_start_sample >= len(data):
            logger.warning(" Post-event trim would remove all data after walking start, skipping")
            return data

        if walking_start_sample > 0:
            # Keep baseline up to walking_start, then drop [walking_start, trim_start) and keep the rest
            baseline_data = data.iloc[:walking_start_sample].copy()
            walking_data = data.iloc[trim_start_sample:].copy()
            trimmed_data = pd.concat([baseline_data, walking_data], ignore_index=True)

            baseline_len = len(baseline_data)

            def _remap(old_idx: int) -> Optional[int]:
                if old_idx < walking_start_sample:
                    return old_idx
                if old_idx < trim_start_sample:
                    return None  # fell inside removed gap
                return baseline_len + (old_idx - trim_start_sample)

            self._event_index_remap = _remap
        else:
            # Walking starts at the beginning; drop first trim_samples
            trimmed_data = data.iloc[trim_samples:].copy()

            def _remap(old_idx: int) -> Optional[int]:
                return (old_idx - trim_samples) if old_idx >= trim_samples else None

            self._event_index_remap = _remap

        # Reset sample numbers/time
        trimmed_data['Sample number'] = np.arange(len(trimmed_data))
        if 'Time (s)' in trimmed_data.columns:
            trimmed_data['Time (s)'] = trimmed_data['Sample number'] / self.fs

        removed = len(data) - len(trimmed_data)
        logger.info(f" Applied post-walking trimming: removed {removed} samples (~{removed / self.fs:.1f}s)")

        return trimmed_data

    def _find_walking_start_event(self, events: pd.DataFrame, task_type: str) -> Optional[int]:
        """
        Find the event marker that indicates walking/task execution start.
        Enhanced to handle mislabeled events for long walk tasks.

        Args:
            events: Event markers DataFrame
            task_type: Type of task being processed

        Returns:
            Sample number of walking start event, or None if not found
        """
        if task_type not in self.task_walking_events:
            logger.warning(f" Unknown task type for walking start detection: {task_type}")
            return None

        # Get possible walking start events for this task type
        possible_events = self.task_walking_events[task_type]

        # Clean event names for comparison
        events_clean = events.copy()
        events_clean['Event_Upper'] = events_clean['Event'].astype(str).str.strip().str.upper()

        # Strategy 1: Look for expected walking start events in order of preference
        for event_name in possible_events:
            matching_events = events_clean[events_clean['Event_Upper'] == event_name.upper()]

            if not matching_events.empty:
                walking_start_sample = matching_events.iloc[0]['Sample number']
                logger.info(f"Found walking start event '{event_name}' at sample {walking_start_sample}")
                return int(walking_start_sample)

        # Strategy 2: Handle long walk tasks with mislabeled events
        task_config = self.task_types.get(task_type, {})
        if task_config.get('type') == 'long_walk':
            logger.info(f"🔧 Long walk task ({task_type}) - checking for alternative event patterns")

            # For long walks, if we have S1, S2, S3 pattern, use S2 as walking start
            s_events = events_clean[events_clean['Event_Upper'].str.match(r'S[1-9]')]
            if len(s_events) >= 2:
                # Sort by sample number to get chronological order
                s_events_sorted = s_events.sort_values('Sample number').reset_index(drop=True)

                # Use the second S event as walking start (typically S2)
                second_s_event = s_events_sorted.iloc[1]
                walking_start_sample = second_s_event['Sample number']
                walking_start_name = second_s_event['Event']

                logger.info(
                    f" Long walk fallback: Using '{walking_start_name}' as walking start at sample {walking_start_sample}")
                event_sequence = []
                for _, row in s_events_sorted.iterrows():
                    event_sequence.append(f"{row['Event']}@{row['Sample number']}")
                logger.info(f"   Event sequence: {event_sequence}")
                return int(walking_start_sample)

            # If we have at least one S event after the first, use it
            s1_events = events_clean[events_clean['Event_Upper'] == 'S1']
            if not s1_events.empty:
                s1_sample = s1_events.iloc[0]['Sample number']

                # Look for any event after S1 that could be walking start
                events_after_s1 = events_clean[events_clean['Sample number'] > s1_sample]
                if not events_after_s1.empty:
                    # Use the first event after S1 as walking start
                    next_event = events_after_s1.sort_values('Sample number').iloc[0]
                    walking_start_sample = next_event['Sample number']
                    logger.info(f" Long walk fallback: Using first event after S1 ('{next_event['Event']}') "
                                f"as walking start at sample {walking_start_sample}")
                    return int(walking_start_sample)

        # Strategy 3: Special logic for event-dependent tasks with multiple S markers
        elif task_config.get('type') == 'event_dependent':
            # For event-dependent tasks, walking usually starts after baseline (S1-S2)
            # So look for S2 or first event after S1
            s1_events = events_clean[events_clean['Event_Upper'] == 'S1']
            if not s1_events.empty:
                s1_sample = s1_events.iloc[0]['Sample number']

                # Look for S2 after S1 (preferred for event-dependent tasks)
                s2_events = events_clean[events_clean['Event_Upper'] == 'S2']
                s2_after_s1 = s2_events[s2_events['Sample number'] > s1_sample]

                if not s2_after_s1.empty:
                    walking_start_sample = s2_after_s1.iloc[0]['Sample number']
                    logger.info(
                        f"Event-dependent task: Using S2 after S1 as walking start at sample {walking_start_sample}")
                    return int(walking_start_sample)

                # Look for next event after S1 (fallback)
                events_after_s1 = events_clean[events_clean['Sample number'] > s1_sample]
                if not events_after_s1.empty:
                    next_event = events_after_s1.iloc[0]
                    walking_start_sample = next_event['Sample number']
                    logger.info(f" Event-dependent fallback: Using first event after S1 ('{next_event['Event']}') "
                                f"at sample {walking_start_sample}")
                    return int(walking_start_sample)

        # Strategy 4: For L-Shape, special case: look for W1 after S2 baseline
        if task_type == 'LShape':
            s2_events = events_clean[events_clean['Event_Upper'] == 'S2']
            if not s2_events.empty:
                s2_sample = s2_events.iloc[0]['Sample number']

                # Look for W1 after S2
                w1_after_s2 = events_clean[
                    (events_clean['Event_Upper'] == 'W1') &
                    (events_clean['Sample number'] > s2_sample)
                    ]
                if not w1_after_s2.empty:
                    walking_start_sample = w1_after_s2.iloc[0]['Sample number']
                    logger.info(f" Found L-Shape walking start 'W1' after S2 at sample {walking_start_sample}")
                    return int(walking_start_sample)

                # Fallback: use next event after S2
                events_after_s2 = events_clean[events_clean['Sample number'] > s2_sample]
                if not events_after_s2.empty:
                    next_event = events_after_s2.iloc[0]
                    walking_start_sample = next_event['Sample number']
                    logger.info(f" L-Shape fallback: Using first event after S2 ('{next_event['Event']}') "
                                f"at sample {walking_start_sample}")
                    return int(walking_start_sample)

        # Final fallback: If we have multiple events, use the second one (common pattern)
        if len(events_clean) >= 2:
            events_sorted = events_clean.sort_values('Sample number').reset_index(drop=True)
            second_event = events_sorted.iloc[1]
            walking_start_sample = second_event['Sample number']
            logger.info(f" General fallback: Using second event ('{second_event['Event']}') "
                        f"as walking start at sample {walking_start_sample}")

            # Create event list for logging
            available_events = []
            for _, row in events_sorted.iterrows():
                available_events.append(f"{row['Event']}@{row['Sample number']}")
            logger.info(f"   Available events: {available_events}")
            return int(walking_start_sample)

        # No suitable walking start event found
        logger.warning(f" Could not find walking start event for {task_type}")
        logger.warning(f"   Available events: {list(events['Event'].unique())}")
        logger.warning(f"   Expected events: {possible_events}")
        return None
    def _calculate_cv_quality_and_filter(self, filtered_data: pd.DataFrame,
                                         o2hb_cols: List[str], hhb_cols: List[str],
                                         output_dir: str, file_basename: str) -> List[str]:
        """Calculate CV quality metrics and return list of channels to exclude if filtering is enabled."""
        all_channels = []
        flagged = []
        excluded_channels = []

        for o2hb_col in o2hb_cols:
            ch_id = o2hb_col.split()[0]
            hhb_col = next((c for c in hhb_cols if c.startswith(ch_id)), None)

            if hhb_col:
                # Calculate CV for both HbO and HbR
                hbo_signal = filtered_data[o2hb_col].to_numpy(dtype=np.float64)
                hbr_signal = filtered_data[hhb_col].to_numpy(dtype=np.float64)

                hbo_cv = self._calc_cv(hbo_signal)
                hbr_cv = self._calc_cv(hbr_signal)
                avg_cv = (hbo_cv + hbr_cv) / 2

                # Determine quality status
                hbo_quality = self._assess_cv_quality(hbo_cv)
                hbr_quality = self._assess_cv_quality(hbr_cv)

                # Overall channel quality (worst of the two)
                if hbo_cv > self.cv_threshold or hbr_cv > self.cv_threshold:
                    overall_quality = 'POOR'
                    flagged.append((ch_id, hbo_cv, hbr_cv, 'POOR'))

                    # Add to exclusion list if filtering is enabled
                    if self.enable_cv_filtering:
                        excluded_channels.extend([o2hb_col, hhb_col])

                elif max(hbo_cv, hbr_cv) > 25:
                    overall_quality = 'FAIR'
                elif max(hbo_cv, hbr_cv) > 15:
                    overall_quality = 'GOOD'
                else:
                    overall_quality = 'EXCELLENT'

                # Store all channel results
                all_channels.append((ch_id, hbo_cv, hbr_cv, avg_cv, hbo_quality, hbr_quality, overall_quality))

        # Save all CV values to a comprehensive log
        if all_channels:
            cv_suffix = "_with_CV_filtering" if self.enable_cv_filtering else "_without_CV_filtering"
            all_cv_log = os.path.join(output_dir,
                                      f"{os.path.splitext(file_basename)[0]}_all_CV_channels{cv_suffix}.txt")
            with open(all_cv_log, 'w') as f:
                f.write("Channel\tHbO_CV\tHbR_CV\tAvg_CV\tHbO_Quality\tHbR_Quality\tOverall_Quality\n")
                for ch_id, hbo_cv, hbr_cv, avg_cv, hbo_quality, hbr_quality, overall_quality in all_channels:
                    f.write(
                        f"{ch_id}\t{hbo_cv:.2f}\t{hbr_cv:.2f}\t{avg_cv:.2f}\t{hbo_quality}\t{hbr_quality}\t{overall_quality}\n")
            logger.info(f"Saved CV values to: {all_cv_log}")

        # Save poor quality channels information
        if flagged:
            poor_cv_log = os.path.join(output_dir,
                                       f"{os.path.splitext(file_basename)[0]}_poor_CV_channels{cv_suffix}.txt")
            with open(poor_cv_log, 'w') as f:
                f.write("Channel\tHbO_CV\tHbR_CV\tStatus\tExcluded_from_Analysis\n")
                for ch_id, hbo_cv, hbr_cv, status in flagged:
                    excluded_status = "YES" if self.enable_cv_filtering else "NO"
                    f.write(f"{ch_id}\t{hbo_cv:.2f}\t{hbr_cv:.2f}\t{status}\t{excluded_status}\n")

            if self.enable_cv_filtering:
                logger.warning(
                    f"Found {len(flagged)} poor quality channels (CV > {self.cv_threshold}%) - EXCLUDED from analysis")
            else:
                logger.warning(
                    f"Found {len(flagged)} poor quality channels (CV > {self.cv_threshold}%) - kept in analysis")
        else:
            logger.info(f"All {len(all_channels)} channels passed CV quality threshold (< {self.cv_threshold}%)")

        return excluded_channels

    def _calc_cv(self, signal: np.ndarray) -> float:
        """Calculate coefficient of variation for an fNIRS signal."""
        # Remove NaN and infinite values
        valid_signal = signal[np.isfinite(signal)]

        if len(valid_signal) < 10:  # Need minimum data points
            return 999.0  # Very high CV indicates poor quality

        mean_val = np.mean(valid_signal)
        std_val = np.std(valid_signal)

        if abs(mean_val) < 1e-10:  # Avoid division by very small numbers
            return 999.0

        cv = (std_val / abs(mean_val)) * 100
        return cv

    def _assess_cv_quality(self, cv_value: float) -> str:
        """Classify signal quality based on CV value."""
        if cv_value < 15:
            return "EXCELLENT"
        elif cv_value < 25:
            return "GOOD"
        elif cv_value < 50:
            return "FAIR"
        else:
            return "POOR"

    def _determine_task_type(self, file_basename: str) -> str:
        """Determine task type from filename with boundary-aware matching."""
        import re
        s = file_basename.upper()

        # Specific tasks first (more informative keywords first)
        if "FTURN" in s or "F_TURN" in s or re.search(r'\bTURN\b', s):
            return "fTurn"
        if "LSHAPE" in s or "L_SHAPE" in s:
            return "LShape"
        if "FIGURE8" in s or "FIG8" in s:
            return "Figure8"
        if "OBSTACLE" in s:
            return "Obstacle"
        if "NAVIGATION" in s or re.search(r'\bNAV\b', s):
            return "Navigation"
        if "WALK" in s:
            return "LongWalk"

        # Strict DT/ST tokens only (avoid POST→ST and IDENT→DT)
        if re.search(r'(^|[^A-Z])DT([^A-Z]|$)', s):
            return "DT"
        if re.search(r'(^|[^A-Z])ST([^A-Z]|$)', s):
            return "ST"

        logger.warning(f" Unknown task type from filename: {file_basename}")
        return "Unknown"

    def _validate_task_requirements(self, task_type: str, task_config: dict,
                                    events: pd.DataFrame, filename: str) -> bool:
        """Validate that task requirements are met before processing."""
        task_category = task_config['type']
        min_events = task_config['min_events']

        if task_category == 'event_dependent':
            # Event-dependent tasks MUST have sufficient event markers
            if events is None or len(events) < min_events:
                logger.error(f" {task_type} task requires at least {min_events} event markers, "
                             f"but only found {len(events) if events is not None else 0} in {filename}")
                if events is not None and not events.empty:
                    logger.error(f"Available events: {list(events['Event'].unique())}")
                return False

            # Special validation for L-Shape: needs at least 3 events
            if task_type == "LShape":
                if len(events) < 3:
                    logger.error(f" L-Shape task requires at least 3 event markers for 2nd→3rd baseline, "
                                 f"but only found {len(events)} in {filename}")
                    logger.error(f"Available events: {list(events['Event'].values)}")
                    return False

                # Log the event sequence for L-Shape
                events_sorted = events.sort_values('Sample number').reset_index(drop=True)
                event_sequence = [(row['Event'], row['Sample number']) for _, row in events_sorted.iterrows()]
                logger.info(f" L-Shape event sequence: {event_sequence}")
                logger.info(f"   Will use baseline: {event_sequence[1][0]} → {event_sequence[2][0]}")

            # For non-L-Shape event tasks, check for S1 baseline marker
            elif 'S1' not in events['Event'].str.upper().values:
                logger.error(f" {task_type} task requires 'S1' baseline marker, but not found in {filename}")
                return False

            logger.info(f" {task_type} task validation passed: {len(events)} events found")

        elif task_category == 'long_walk':
            # Long walk tasks can proceed with or without events (fallback available)
            if events is None or events.empty:
                logger.warning(f" {task_type} task has no event markers, will use time-based fallback")
            else:
                logger.info(f" {task_type} task has {len(events)} event markers available")

        else:
            logger.warning(f" Unknown task category '{task_category}' for task type '{task_type}'")

        return True

    def _extract_and_clean_events(self, data_dict: dict, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and properly clean event data to prevent plotting issues."""
        try:
            events = data_dict.get('events', None)

            # If no events in data_dict, try to extract from data columns
            if events is None or events.empty:
                if 'Event' in data.columns:
                    # Create events DataFrame from data
                    event_mask = data['Event'].notna() & (data['Event'] != '') & (data['Event'] != 'nan')
                    if event_mask.any():
                        events = data.loc[event_mask, ['Sample number', 'Event']].copy()
                        logger.debug(f" Extracted {len(events)} event markers from data columns")
                    else:
                        events = pd.DataFrame(columns=['Sample number', 'Event'])
                else:
                    logger.warning(f" No event data found")
                    events = pd.DataFrame(columns=['Sample number', 'Event'])

            if events is not None and not events.empty:
                # Clean the events DataFrame thoroughly
                events = events.copy()

                # Ensure required columns exist
                if 'Sample number' not in events.columns or 'Event' not in events.columns:
                    logger.warning(f" Events missing required columns, creating empty events DataFrame")
                    return pd.DataFrame(columns=['Sample number', 'Event'])

                # Clean event names - remove whitespace, convert to string
                events['Event'] = events['Event'].astype(str).str.strip()

                # Remove rows with invalid events
                invalid_events = ['', 'nan', 'None', 'NaN', 'null']
                events = events[~events['Event'].str.lower().isin([x.lower() for x in invalid_events])]

                # Clean sample numbers - ensure they're numeric and valid
                events['Sample number'] = pd.to_numeric(events['Sample number'], errors='coerce')
                events = events.dropna(subset=['Sample number'])

                # Remove events outside the valid sample range
                max_samples = len(data)
                events = events[
                    (events['Sample number'] >= 0) &
                    (events['Sample number'] <= max_samples)
                    ]

                # Remove duplicate events at the same sample
                events = events.drop_duplicates(subset=['Sample number'])

                # Sort by sample number
                events = events.sort_values('Sample number').reset_index(drop=True)

                logger.debug(f" Cleaned events: {len(events)} valid events found")
                logger.debug(f"Event types: {events['Event'].unique()}")

                return events
            else:
                logger.debug(" No events to clean")
                return pd.DataFrame(columns=['Sample number', 'Event'])

        except Exception as e:
            logger.warning(f"⚠ Error cleaning events: {str(e)}")
            return pd.DataFrame(columns=['Sample number', 'Event'])

    # Private helper methods
    @staticmethod
    def _create_output_dir(output_base: str, input_base: str, file_path: str) -> str:
        """Create output directory mirroring input structure."""
        relative_path = os.path.relpath(os.path.dirname(file_path), start=input_base)
        output_dir = os.path.join(output_base, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def _extract_subject(file_path: str) -> str:
        """Extract subject ID from file path."""
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if "OHSU_Turn" in part or any(x in part for x in ["Subject", "subj", "sub-"]):
                return part
        return "Unknown"

    @staticmethod
    def _get_plotting_limits(subject: str, subject_y_limits: Optional[Dict]) -> Tuple:
        """Get plotting limits for raw data."""
        if not subject_y_limits or subject not in subject_y_limits:
            return None
        return (subject_y_limits[subject]['raw_min'], subject_y_limits[subject]['raw_max'])

    @staticmethod
    def _prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw data DataFrame."""
        data = raw_data.copy()
        if "Sample number" not in data.columns:
            data.insert(0, "Sample number", np.arange(len(data)))
        return data

    def _plot_raw_data(self, data: pd.DataFrame, output_dir: str,
                       file_basename: str, subject: str,
                       global_ylim: Optional[Tuple[float, float]],
                       condition: str,
                       events: Optional[pd.DataFrame] = None) -> None:
        """Updated plotting function that includes properly cleaned events."""
        o2hb_cols = [col for col in data.columns if ('O2Hb' in col or 'HbO' in col)]
        hhb_cols = [col for col in data.columns if ('HHb' in col or 'HbR' in col)]
        combined_cols = o2hb_cols + hhb_cols
        if not combined_cols:
            return

        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        # Individual channels
        fig, axes, ylim = plot_channels_separately(
            data[combined_cols],
            fs=self.fs,
            title=f"{file_basename} - Raw Data",
            subject=subject,
            condition=condition,
            y_lim=global_ylim
        )
        self._save_figure(fig, os.path.join(condition_dir, f"raw_individual_channels_{condition}.png"))

        # Overall signals + (clean) events
        if o2hb_cols and hhb_cols:
            avg_o2hb = data[o2hb_cols].mean(axis=1)
            avg_hhb = data[hhb_cols].mean(axis=1)
            overall_df = pd.DataFrame({
                "Time (s)": np.arange(len(data)) / self.fs,
                "grand oxy": avg_o2hb,
                "grand deoxy": avg_hhb
            })

            clean_events = None
            if events is not None and not events.empty:
                valid = events[
                    events['Sample number'].notna() &
                    events['Event'].notna() &
                    (events['Event'] != '') &
                    (events['Sample number'] >= 0) &
                    (events['Sample number'] < len(data))
                    ].copy()
                if not valid.empty:
                    clean_events = valid

            fig, ylim = plot_overall_signals(
                overall_df,
                fs=self.fs,
                title=f"{file_basename} - Raw Overall",
                subject=subject,
                condition=condition,
                y_lim=global_ylim,
                events=clean_events
            )
            self._save_figure(fig, os.path.join(condition_dir, f"raw_overall_{condition}.png"))

    def _apply_scr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Short Channel Regression for prefrontal-only cap.

        Assumes loaders.py renamed channels to 'CH{i} HbO'/'CH{i} HbR'.
        Short channels are CH6 and CH7 (both HbO & HbR) for 0-based indexing.
        Falls back gracefully if some channels are missing.
        """
        try:
            # fNIRS signal columns (after loaders.py renaming)
            sig_cols = [c for c in data.columns if any(k in c for k in ("HbO", "O2Hb", "HHb", "HbR"))]
            if not sig_cols:
                return data

            # Identify channel prefixes, e.g., 'CH7' from 'CH7 HbO'
            prefixes = {c.split()[0] for c in sig_cols}

            # --- Prefrontal-only cap: short = CH6 & CH7 (both chromophores) ---
            short_ids = {"CH6", "CH7"}  # CHANGED FROM {"CH7", "CH8"}
            short_prefixes = prefixes & short_ids
            long_prefixes = prefixes - short_prefixes

            short_cols = [c for c in sig_cols if c.split()[0] in short_prefixes]
            long_cols = [c for c in sig_cols if c.split()[0] in long_prefixes]

            # Guardrails: need at least one short AND one long channel to run SCR
            if not short_cols or not long_cols:
                # Nothing to regress with — just return unchanged
                if not short_cols:
                    logger.warning("SCR skipped: no short-channel columns (CH6/CH7) present after renaming.")
                elif not long_cols:
                    logger.warning("SCR skipped: no long-channel columns present.")
                return data

            # Run SCR on the long channels using short channels as regressors
            scr_long = scr_regression(data[long_cols], data[short_cols])

            # Recombine: replace long columns with corrected versions; keep everything else (incl. short & metadata)
            out = data.copy()
            for col in scr_long.columns:
                if col in out.columns:
                    out[col] = scr_long[col]

            logger.info(f"SCR applied: {len(long_cols)} long channels, {len(short_cols)} short channels")
            return out

        except Exception as e:
            logger.warning(f"SCR failed: {str(e)}")
            return data
    def _apply_tddr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply TDDR correction."""
        try:
            return tddr(data, sample_rate=self.fs)
        except Exception as e:
            logger.warning(f"TDDR failed: {str(e)}")
            return data

    def _apply_baseline_correction(self, data: pd.DataFrame, events: pd.DataFrame,
                                   task_type: str = None) -> pd.DataFrame:
        """Apply task-specific baseline correction."""
        try:
            if events is None or events.empty:
                logger.warning("No events available for baseline correction, using fallback")
                return self._fallback_baseline_correction(data)

            # Clean and normalize event labels
            events = events.copy()
            events['Event'] = events['Event'].astype(str).str.strip().str.upper()

            # L-Shape specific baseline correction: S2 to W1
            if task_type == "LShape":
                return self._apply_lshape_baseline(data, events)

            # For all other tasks, use the original logic
            return self._apply_standard_baseline(data, events, task_type)

        except Exception as e:
            logger.error(f"Baseline correction failed: {str(e)}")
            return None

    def _apply_lshape_baseline(self, data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Apply L-Shape specific baseline correction using 2nd to 3rd event markers."""
        try:
            # Sort events by sample number to get chronological order
            events_sorted = events.sort_values('Sample number').reset_index(drop=True)

            if len(events_sorted) < 3:
                logger.error(f" L-Shape task requires at least 3 event markers, but only found {len(events_sorted)}")
                logger.error(
                    f"Available events: {list(events_sorted['Event'].values) if not events_sorted.empty else 'None'}")
                return None

            # Get the 2nd and 3rd events (index 1 and 2)
            second_event = events_sorted.iloc[1]
            third_event = events_sorted.iloc[2]

            second_sample = second_event['Sample number']
            third_sample = third_event['Sample number']
            second_name = second_event['Event']
            third_name = third_event['Event']

            # Verify that third event comes after second event
            if third_sample <= second_sample:
                logger.error(
                    f" L-Shape baseline error: 3rd event ({third_name}@{third_sample}) must come after 2nd event ({second_name}@{second_sample})")
                return None

            # Create baseline events for 2nd→3rd event period
            baseline_events = pd.DataFrame({
                'Sample number': [second_sample, third_sample],
                'Event': ['BaselineStart', 'BaselineEnd']
            })

            logger.info(
                f"✅ L-Shape baseline correction: {second_name} ({second_sample}) to {third_name} ({third_sample})")
            logger.info(f"   Baseline duration: {(third_sample - second_sample) / self.fs:.1f} seconds")

            return baseline_subtraction(data, baseline_events, baseline_type="lshape_task")

        except Exception as e:
            logger.error(f"L-Shape baseline correction failed: {str(e)}")
            return None

    def _apply_standard_baseline(self, data: pd.DataFrame, events: pd.DataFrame, task_type: str = None) -> pd.DataFrame:
        """Apply standard baseline correction for non-L-Shape tasks."""
        try:
            # Find S1 marker
            s1_markers = events[events['Event'] == 'S1']

            if not s1_markers.empty:
                s1_sample = s1_markers.iloc[0]['Sample number']

                # Strategy 1: Look for W1 after S1 (preferred for long walks)
                w1_markers = events[events['Event'] == 'W1']
                w1_after_s1 = w1_markers[w1_markers['Sample number'] > s1_sample]

                if not w1_after_s1.empty:
                    w1_sample = w1_after_s1.iloc[0]['Sample number']
                    baseline_events = pd.DataFrame({
                        'Sample number': [s1_sample, w1_sample],
                        'Event': ['BaselineStart', 'BaselineEnd']
                    })
                    logger.debug(f"Using S1-W1 baseline: {s1_sample} to {w1_sample}")
                    return baseline_subtraction(data, baseline_events, baseline_type="long_walk")

                # Strategy 2: Look for S2 after S1 (common for event-dependent tasks)
                s2_markers = events[events['Event'] == 'S2']
                s2_after_s1 = s2_markers[s2_markers['Sample number'] > s1_sample]

                if not s2_after_s1.empty:
                    s2_sample = s2_after_s1.iloc[0]['Sample number']
                    baseline_events = pd.DataFrame({
                        'Sample number': [s1_sample, s2_sample],
                        'Event': ['BaselineStart', 'BaselineEnd']
                    })
                    logger.debug(f"Using S1-S2 baseline: {s1_sample} to {s2_sample}")
                    return baseline_subtraction(data, baseline_events, baseline_type="event_based")

                # Strategy 3: Look for any other numbered markers (S3, S4, etc.)
                other_s_markers = events[events['Event'].str.match(r'S[3-9]')]
                other_s_after_s1 = other_s_markers[other_s_markers['Sample number'] > s1_sample]

                if not other_s_after_s1.empty:
                    next_s_sample = other_s_after_s1.iloc[0]['Sample number']
                    next_s_event = other_s_after_s1.iloc[0]['Event']
                    baseline_events = pd.DataFrame({
                        'Sample number': [s1_sample, next_s_sample],
                        'Event': ['BaselineStart', 'BaselineEnd']
                    })
                    logger.debug(f"Using S1-{next_s_event} baseline: {s1_sample} to {next_s_sample}")
                    return baseline_subtraction(data, baseline_events, baseline_type="event_based")

                # Strategy 4: Look for task-specific end markers
                end_markers = events[events['Event'].str.upper().isin(['END', 'FINISH', 'STOP', 'COMPLETE'])]
                end_after_s1 = end_markers[end_markers['Sample number'] > s1_sample]

                if not end_after_s1.empty:
                    end_sample = end_after_s1.iloc[0]['Sample number']
                    baseline_events = pd.DataFrame({
                        'Sample number': [s1_sample, end_sample],
                        'Event': ['BaselineStart', 'BaselineEnd']
                    })
                    logger.debug(f"Using S1-END baseline: {s1_sample} to {end_sample}")
                    return baseline_subtraction(data, baseline_events, baseline_type="event_based")

            # If no usable baseline found, check task type for fallback eligibility
            task_config = self.task_types.get(task_type, {})
            if task_config.get('type') == 'long_walk':
                # Long walks can use fallback
                logger.warning("No event-based baseline found for long walk task, using time-based fallback")
                return self._fallback_baseline_correction(data)
            else:
                # Event-dependent tasks fail without proper events
                logger.error(" No valid baseline markers found for event-dependent task")
                return None

        except Exception as e:
            logger.error(f"Standard baseline correction failed: {str(e)}")
            return None

    def _fallback_baseline_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback baseline correction using end-of-recording; clamps indices safely."""
        try:
            total = len(data)
            if total < int(20 * self.fs):
                logger.warning("Fallback baseline: record too short; returning data without subtraction")
                return data

            # Candidate windows near end; clamp into valid range
            s1 = max(0, min(total - 1, int(total - 150 * self.fs)))
            s2 = max(0, min(total - 1, int(total - 130 * self.fs)))
            s3 = max(0, min(total - 1, int(total - 10 * self.fs)))

            # Ensure monotone increasing and minimum span
            marks = sorted({s1, s2, s3})
            if len(marks) < 3 or marks[1] - marks[0] < int(2 * self.fs):
                # Stretch minimal window if needed
                base = max(0, total - int(150 * self.fs))
                end = max(base + int(2 * self.fs), min(total - 1, total - int(10 * self.fs)))
                marks = [base, base + int(2 * self.fs), end]

            fallback_events = pd.DataFrame({
                'Sample number': marks,
                'Event': ['S1', 'S2', 'S3']
            })
            logger.warning(f"Using fallback end-of-recording baseline markers: {marks}")
            return baseline_subtraction(data, fallback_events)

        except Exception as e:
            logger.warning(f"Fallback baseline correction failed: {str(e)}")
            return data

    def _finalize_outputs(self, data: pd.DataFrame, output_dir: str,
                          file_basename: str, subject: str,
                          task_type: str, task_config: dict,
                          events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Final output step with robust post-event trimming, averaging, safe smoothing,
        guaranteed CSV save, and final plots. Never returns None due to smoothing/plotting.
        """
        try:
            # 1) Post-event trimming (if events available)
            if events is not None and not events.empty:
                data = self._apply_post_event_trimming(data, events, task_type)

            # 2) Channel averaging (left/right + grand)
            averaged = average_channels(data)

            # Sanity check: ensure required columns exist (create if missing to avoid aborts)
            for col in ['grand oxy', 'grand deoxy']:
                if col not in averaged.columns:
                    logger.warning(f"Missing '{col}' after averaging; filling with NaNs.")
                    averaged[col] = np.nan

            # 3) Time column (always present)
            if 'Sample number' in averaged.columns:
                averaged['Time (s)'] = averaged['Sample number'] / self.fs
            else:
                averaged['Time (s)'] = np.arange(len(averaged)) / self.fs

            # 4) Safe Savitzky–Golay smoothing (non-fatal; NaN/Inf aware)
            from scipy.signal import savgol_filter

            def _odd_leq(n: int) -> int:
                """Largest odd integer ≤ n (returns 0 if n<=0)."""
                return n if n % 2 == 1 else (n - 1)

            def _savgol_safe(series: pd.Series, window_default: int, poly: int) -> np.ndarray:
                arr = np.asarray(series, dtype=float)
                # Replace inf with nan for processing
                arr[~np.isfinite(arr)] = np.nan
                L = arr.size
                if L == 0:
                    return arr

                # pick an odd window <= length; ensure minimum viable length for poly
                wmax = max(_odd_leq(L), 0)
                window = min(window_default, wmax)
                min_ok = max(poly + 2, 3)  # minimal window length
                if window < min_ok:
                    # too short for SG; return unsmoothed
                    return arr

                # simple internal NaN interpolation (keep edge NaNs if entire edge is NaN)
                nans = ~np.isfinite(arr)
                if nans.any() and (~nans).sum() >= 2:
                    idx = np.arange(L)
                    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

                try:
                    out = savgol_filter(arr, window_length=window, polyorder=poly, mode='interp')
                except Exception as e:
                    logger.warning(f"Savitzky–Golay failed (win={window}, poly={poly}): {e}. Using unsmoothed data.")
                    out = arr
                return out

            # choose ~1 s window, odd, not exceeding length
            poly = 2
            w_default = int(self.fs)
            if w_default % 2 == 0:
                w_default += 1  # ensure odd

            averaged['smoothed oxy'] = _savgol_safe(averaged['grand oxy'], w_default, poly)
            averaged['smoothed deoxy'] = _savgol_safe(averaged['grand deoxy'], w_default, poly)

            # 5) Output directory and filenames (encode CV status in both dir and file)
            cv_suffix = "_with_CV_filtering" if self.enable_cv_filtering else "_without_CV_filtering"
            condition_dir = os.path.join(output_dir, f"{task_type}{cv_suffix}")
            os.makedirs(condition_dir, exist_ok=True)

            output_file = os.path.join(
                condition_dir,
                f"{file_basename}_FULLY_PROCESSED{cv_suffix}.csv"
            )

            # 6) Add metadata and flags (after smoothing so they get saved too)
            task_category = task_config.get('type', 'unknown')
            filtering_status = "CV_Filtered" if self.enable_cv_filtering else "All_Channels"

            averaged['Condition'] = f"{task_category}_{task_type}_{filtering_status}"
            averaged['Subject'] = subject
            averaged['TaskType'] = task_type
            averaged['TaskCategory'] = task_category
            averaged['CV_Filtering_Applied'] = bool(self.enable_cv_filtering)
            averaged['Post_Event_Trim_Applied'] = bool(self.post_walking_trim_seconds > 0)

            # Preserve event column if present in input
            if 'Event' in data.columns and 'Event' not in averaged.columns:
                averaged['Event'] = data['Event'].reindex(averaged.index)

            # 7) Save CSV (always try to save, even if plots fail)
            averaged.to_csv(output_file, index=False)
            logger.info(f"Saved fully processed data (with post-event trimming) to {output_file}")

            # 8) Create final plots (non-fatal if anything goes wrong)
            try:
                self._create_final_plot(
                    averaged, condition_dir, file_basename, f"{task_type}{cv_suffix}",
                    ['grand oxy', 'grand deoxy'], f'final_overall{cv_suffix}',
                    f'Final Overall{cv_suffix}', events
                )

                if 'smoothed oxy' in averaged.columns and 'smoothed deoxy' in averaged.columns:
                    self._create_final_plot(
                        averaged, condition_dir, file_basename, f"{task_type}{cv_suffix}",
                        ['smoothed oxy', 'smoothed deoxy'], f'final_smoothed{cv_suffix}',
                        f'Final Smoothed{cv_suffix}', events
                    )
            except Exception as e:
                logger.warning(f"Plotting failed for {file_basename} ({task_type}{cv_suffix}): {e}")

            return averaged

        except Exception as e:
            logger.error(f"Final output generation failed: {str(e)}", exc_info=True)
            # Last resort: try to save whatever we have to help debugging
            try:
                fail_dir = os.path.join(output_dir, "FAILED_OUTPUTS")
                os.makedirs(fail_dir, exist_ok=True)
                fail_path = os.path.join(fail_dir, f"{os.path.splitext(file_basename)[0]}_FAILED{cv_suffix}.csv")
                if 'averaged' in locals() and isinstance(averaged, pd.DataFrame):
                    averaged.to_csv(fail_path, index=False)
                    logger.error(f"Saved partial data to {fail_path} for debugging.")
            except Exception:
                pass
            return None

    def _create_final_plot(self, data: pd.DataFrame, output_dir: str,
                           file_basename: str, condition: str,
                           columns: List[str], prefix: str,
                           title: str,
                           events: Optional[pd.DataFrame] = None) -> None:
        """Create standardized final plots with automatic scaling and (remapped) events."""
        try:
            # Prepare plot data
            if 'Time (s)' not in data.columns:
                data = data.copy()
                data['Time (s)'] = np.arange(len(data)) / self.fs

            # Rename to what plot_overall_signals expects
            plot_data = data[columns + ['Time (s)']].rename(columns={
                columns[0]: "grand oxy",
                columns[1]: "grand deoxy"
            })

            # FIXED: Better event remapping that preserves start events
            clean_events = None
            if events is not None and not events.empty:
                ev = events.copy()
                ev = ev.dropna(subset=['Sample number'])

                # DEBUG: Log original events
                logger.debug(f"Original events for {file_basename}: {len(ev)} events")
                for _, event in ev.iterrows():
                    logger.debug(f"  Event: '{event['Event']}' at sample {event['Sample number']}")

                # Apply event remap if trimming occurred
                if hasattr(self, "_event_index_remap") and callable(self._event_index_remap):
                    # Create new column for remapped samples instead of overwriting
                    ev['Remapped_Sample'] = ev['Sample number'].apply(self._event_index_remap)

                    # Define important start events that we want to preserve
                    walking_start_events = ['W1', 'WALK', 'START_WALK', 'WALKING', 'S2', 'START', 'TASK_START', 'GO',
                                            'S1']

                    # Convert to uppercase for case-insensitive comparison
                    ev['Event_Upper'] = ev['Event'].astype(str).str.upper()
                    start_events_upper = [e.upper() for e in walking_start_events]

                    # Identify which events are start events
                    is_start_event = ev['Event_Upper'].isin(start_events_upper)

                    logger.debug(f"Start events found: {list(ev[is_start_event]['Event'].values)}")

                    # Strategy: Preserve ALL events, but handle remapping carefully
                    # For start events that get mapped to None, keep them at original position
                    # but only if they're within the current data range
                    max_data_samples = len(data) - 1

                    def get_final_sample(row):
                        if pd.notna(row['Remapped_Sample']):
                            # Successfully remapped
                            return row['Remapped_Sample']
                        elif is_start_event[row.name]:
                            # This is a start event that was mapped to None - try to keep it
                            original_sample = row['Sample number']
                            if 0 <= original_sample <= max_data_samples:
                                logger.debug(
                                    f"Preserving start event '{row['Event']}' at original sample {original_sample} (was remapped to None)")
                                return original_sample
                        return None

                    ev['Final_Sample'] = ev.apply(get_final_sample, axis=1)
                    ev = ev.dropna(subset=['Final_Sample'])
                    ev['Sample number'] = ev['Final_Sample'].astype(int)
                    ev = ev.drop(columns=['Remapped_Sample', 'Final_Sample', 'Event_Upper'])
                else:
                    logger.debug("No event index remapping function found")

                # Keep only events in range
                max_idx = len(data) - 1
                ev = ev[(ev['Sample number'] >= 0) & (ev['Sample number'] <= max_idx)]

                if not ev.empty:
                    clean_events = ev
                    logger.info(f"Final plot will show {len(clean_events)} events for {file_basename}:")
                    for _, event in clean_events.iterrows():
                        logger.info(
                            f"  → '{event['Event']}' at sample {event['Sample number']} (time: {event['Sample number'] / self.fs:.1f}s)")
                else:
                    logger.warning(f"No events remaining after filtering for {file_basename}")
            else:
                logger.warning(f"No events available for plotting in {file_basename}")

            # Try to recover a sensible subject label
            subject = self._extract_subject(file_basename)

            # Generate plot
            fig, ylim = plot_overall_signals(
                plot_data,
                fs=self.fs,
                title=f"{file_basename} - {title}",
                subject=subject,
                condition=condition,
                y_lim=None,
                events=clean_events
            )

            # Save with consistent naming
            output_path = os.path.join(output_dir, f"{prefix}_{condition}.png")
            self._save_figure(fig, output_path)

            # Log successful plot creation
            event_count = len(clean_events) if clean_events is not None else 0
            logger.info(f"Successfully created {title} plot with {event_count} event markers: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create {title} plot for {file_basename}: {str(e)}", exc_info=True)
            raise

    @staticmethod
    @staticmethod
    def _save_figure(fig, path: str) -> None:
        """Save a matplotlib Figure object safely."""
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved figure to {path}")
        except Exception as e:
            plt.close(fig)
            logger.error(f"Failed to save figure {os.path.basename(path)}: {str(e)}")
            raise
