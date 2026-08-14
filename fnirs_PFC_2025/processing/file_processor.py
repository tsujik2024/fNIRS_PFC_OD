import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple, Callable, List
import logging
import re
# Import processing steps
from fnirs_PFC_2025.preprocessing.z_transformation import z_transformation
from fnirs_PFC_2025.preprocessing.fir_filter import fir_filter
from fnirs_PFC_2025.preprocessing.short_channel_regression import scr_regression
from fnirs_PFC_2025.preprocessing.tddr import tddr
from fnirs_PFC_2025.preprocessing.baseline_correction import baseline_subtraction
from fnirs_PFC_2025.preprocessing.average_channels import average_channels
from fnirs_PFC_2025.preprocessing.signalqualityindex import SQI, fir_filter as sqi_fir_filter
from fnirs_PFC_2025.preprocessing.sci import scalp_coupling_index
from fnirs_PFC_2025.preprocessing.psp import peak_spectral_power
from fnirs_PFC_2025.processing.quality_control import (
    ChannelQuality, QualityReport,
    DEFAULT_SQI_THRESHOLD, DEFAULT_SCI_THRESHOLD, DEFAULT_PSP_THRESHOLD,
)
# Import plotting functions
from fnirs_PFC_2025.viz.plots import plot_channels_separately, plot_overall_signals

logger = logging.getLogger(__name__)
plt.ioff()  # Non-interactive backend

# Short channel IDs to exclude from averaging/regression grouping.
# 0-based channel indices. Project convention (see quality_control.py,
# BatchProcessor, PipelineManager): the OctaMon short channels labelled
# 4 and 6 in 1-based optode numbering are CH3 and CH5 here (0-based).
SHORT_CHANNEL_IDS = {3, 5}


class FileProcessor:

    def __init__(self, fs: float = 50.0,
                 sqi_threshold: float = DEFAULT_SQI_THRESHOLD,
                 sci_threshold: float = DEFAULT_SCI_THRESHOLD,
                 psp_threshold: float = DEFAULT_PSP_THRESHOLD,
                 enabled_metrics: Tuple[str, ...] = ("sci", "psp"),
                 enable_quality_filtering: bool = True,
                 exclude_failing_short_channels: bool = False,
                 post_walking_trim_seconds: float = 3.0,
                 initial_crop_seconds: float = 1.0):
        """
        Initialize processor with parameters.

        Args:
            fs: Sampling frequency in Hz
            sqi_threshold: SQI threshold for quality assessment (1-5 scale, default 2.0)
            sci_threshold: Scalp Coupling Index threshold (default 0.75, Pollonini/PHOEBE)
            psp_threshold: Peak Spectral Power threshold (default 0.10, Pollonini/PHOEBE)
            enabled_metrics: which of "sqi", "sci", "psp" participate in the pass/fail
                decision. Metrics not listed here are not computed at all. Default
                ("sci", "psp") - SQI is opt-in.
            enable_quality_filtering: If True (default), channels that fail an enabled
                metric are dropped from SCR/FIR/baseline/output. If False, quality is
                still computed and reported, but no channels are removed.
            exclude_failing_short_channels: If True, short channels (CH3/CH5) that fail
                are also dropped. Default False - short channels are kept regardless of
                quality since they're used as SCR regressors, not signal.
            post_walking_trim_seconds: Seconds to trim after walking start event for quality control (default 3.0)
            initial_crop_seconds: Seconds to drop from the START of every recording
                (device/initialization artifacts), applied before anything else - events,
                quality metrics, TDDR, etc. all see the already-cropped recording.
                Default 1.0s, matching FullCapProcessor's initial crop.
        """
        self.fs = fs
        self.sqi_threshold = sqi_threshold
        self.sci_threshold = sci_threshold
        self.psp_threshold = psp_threshold
        self.enabled_metrics = tuple(m.lower() for m in enabled_metrics)
        invalid = set(self.enabled_metrics) - {"sqi", "sci", "psp"}
        if invalid:
            raise ValueError(f"Unknown quality metric(s) {invalid}; expected a subset of "
                             f"{{'sqi', 'sci', 'psp'}}.")
        self.enable_quality_filtering = enable_quality_filtering
        self.exclude_failing_short_channels = exclude_failing_short_channels
        self.post_walking_trim_seconds = post_walking_trim_seconds
        self.initial_crop_seconds = initial_crop_seconds

        # Define task types and their walking start events
        self.task_walking_events = {
            # Long walk tasks - look for walking start
            'DT': ['W1', 'WALK', 'START_WALK', 'WALKING'],
            'ST': ['W1', 'WALK', 'START_WALK', 'WALKING'],
            'LongWalk': ['W1', 'WALK', 'START_WALK', 'WALKING'],

            # Event-dependent tasks - look for task start (after baseline)
            'fTurn': ['S2', 'START', 'TASK_START', 'GO'],
            'LShape': ['W1', 'WALK', 'START_WALK', 'S3'],  # W1 after S2 baseline
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
            'Obstacle': {'type': 'event_dependent', 'min_events': 3},
            'Navigation': {'type': 'event_dependent', 'min_events': 2},
        }

        filter_status = "ENABLED" if enable_quality_filtering else "DISABLED"
        logger.info(f"Initialized FileProcessor (fs={fs}, metrics={self.enabled_metrics}, "
                    f"thresholds: SQI>={sqi_threshold} SCI>={sci_threshold} PSP>={psp_threshold}, "
                    f"quality filtering={filter_status}, initial crop={initial_crop_seconds}s, "
                    f"post-walking trim={post_walking_trim_seconds}s)")

    def _is_short_channel(self, col_name: str) -> bool:
        """Check if a column belongs to a short channel (CH3 or CH5, i.e. 1-based optodes 4 and 6)."""
        match = re.match(r'CH(\d+)', str(col_name))
        if match:
            ch_num = int(match.group(1))
            return ch_num in SHORT_CHANNEL_IDS
        return False

    def _get_long_channel_cols(self, columns: List[str], chromophore: str = 'both') -> List[str]:
        """
        Get column names for long channels only (excluding short channels CH3, CH5).

        Args:
            columns: List of column names to filter
            chromophore: 'oxy' for HbO/O2Hb, 'deoxy' for HbR/HHb, 'both' for all

        Returns:
            List of column names excluding short channels
        """
        if chromophore == 'oxy':
            keywords = ['HbO', 'O2Hb']
        elif chromophore == 'deoxy':
            keywords = ['HbR', 'HHb']
        else:
            keywords = ['HbO', 'O2Hb', 'HbR', 'HHb']

        long_cols = []
        for col in columns:
            # Check if it's a concentration column
            if any(kw in col for kw in keywords) and 'grand' not in col.lower():
                # Check if it's NOT a short channel
                if not self._is_short_channel(col):
                    long_cols.append(col)
        return long_cols

    def process_file(self,
                     file_path: str,
                     output_base_dir: str,
                     input_base_dir: str,
                     subject_y_limits: Optional[Dict] = None,
                     read_file_func: Callable = None,
                     baseline_duration: Optional[float] = None,
                     ) -> Optional[Dict]:
        """
        Process a single fNIRS file through the complete pipeline with optional SQI filtering and post-walking trimming.

        Returns:
            Dict with keys:
                - 'success': bool indicating if processing succeeded
                - 'data': DataFrame with processed data (if success=True)
                - 'subject': str subject ID (if success=True)
                - 'task_type': str task type (if success=True)
                - 'error': str error message (if success=False)
        """
        try:
            self._event_index_remap = None
            self._last_quality_report = None
            # ─── Setup paths & names ───────────────────────────
            output_dir = self._create_output_dir(output_base_dir, input_base_dir, file_path)
            file_basename = os.path.basename(file_path)
            subject = self._extract_subject(file_path)
            raw_limits = self._get_plotting_limits(subject, subject_y_limits)

            # Store file_basename for use in fallback timing
            self._current_file_basename = file_basename

            logger.info(f" Starting: {file_path} (quality filtering: {'ON' if self.enable_quality_filtering else 'OFF'}, "
                        f"post-walking trim: {self.post_walking_trim_seconds}s)")

            # ─── 1) Load raw data ───────────────────────────────
            data_dict = read_file_func(file_path)
            if not data_dict or 'data' not in data_dict:
                logger.error(f" read_file_func failed or returned no 'data' for: {file_path}")
                return {'success': False, 'error': 'Failed to read data from file'}

            raw_df = data_dict['data']
            if raw_df is None or not isinstance(raw_df, pd.DataFrame):
                logger.error(f" Invalid DataFrame returned for: {file_path}")
                return {'success': False, 'error': 'Invalid DataFrame returned'}

            data = self._prepare_data(raw_df)
            if data is None or data.empty:
                logger.error(f" Prepared data is empty for: {file_path}")
                return {'success': False, 'error': 'Prepared data is empty'}

            logger.debug(f" Loaded {len(data)} rows × {len(data.columns)} cols")

            # Drop the first `initial_crop_seconds` of the recording (device/
            # initialization artifacts), matching FullCapProcessor. Everything
            # downstream - event extraction, quality metrics, TDDR, etc. - sees
            # only the already-cropped recording.
            data = self._drop_initial_seconds(data, self.initial_crop_seconds)
            if data is None or data.empty:
                logger.error(f" Data empty after initial-seconds crop for: {file_path}")
                return {'success': False, 'error': 'Data empty after initial-seconds crop'}

            # ─── 2) Metadata ────────────────────────────────────
            metadata = data_dict.get('metadata', {})
            subject_id = metadata.get('Subject Public ID')
            record_date = metadata.get('Record Date/Time')
            sample_rate = int(self.fs)

            # ─── 3) Determine task type ─────────────────────────
            task_type = self._determine_task_type(file_basename)
            logger.debug(f"Detected task type: {task_type}")

            task_config = self.task_types.get(task_type, {'type': 'unknown', 'min_events': 2})

            # Log timing info for this file
            timing = self._get_task_timing(file_basename, task_type)
            logger.info(f"Task timing for {file_basename}: {timing['total_expected']}s total "
                        f"({timing['baseline_duration']}s baseline + {timing['task_duration']}s task + "
                        f"{timing['end_duration']}s end)")

            # ─── 4) Extract and clean events ──────────────────────────────
            events = self._extract_and_clean_events(data_dict, data)

            # ─── 5) Validate task requirements ──────────────────────────────
            if not self._validate_task_requirements(task_type, task_config, events, file_basename):
                logger.error(f" Task requirements not met for {file_basename}")
                return {'success': False, 'error': 'Task validation failed', 'validation_failed': True}

            # ─── 6) Processing pipeline with SQI filtering ───────
            # NOTE: "Raw" concentration plotting now happens INSIDE _process_pipeline_stages
            # after Beer-Lambert conversion (pre-TDDR) but before TDDR/SCR/filtering
            processed_data = self._process_pipeline_stages(
                data=data,
                output_dir=output_dir,
                file_basename=file_basename,
                events=events,
                task_type=task_type,
                subject=subject,
                global_ylim=raw_limits
            )
            if processed_data is None or processed_data.empty:
                logger.error(f" Pipeline stages returned empty for: {file_path}")
                return {'success': False, 'error': 'Pipeline processing returned empty data'}

            # ─── 7) Final outputs with post-walking trimming ───────────────────────────────
            final_df = self._finalize_outputs(
                processed_data, output_dir, file_basename, subject,
                task_type, task_config, events
            )

            if final_df is None or final_df.empty:
                logger.error(f" _finalize_outputs failed for: {file_path}")
                return {'success': False, 'error': 'Finalize outputs failed'}

            logger.info(f" Finished processing: {file_path}")
            return {
                'success': True,
                'data': final_df,
                'subject': subject,
                'task_type': task_type,
                'output_dir': output_dir,
                'quality': getattr(self, '_last_quality_report', None),
            }

        except Exception as e:
            logger.error(f" Exception in process_file for {file_path}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _get_task_timing(self, file_basename: str, task_type: str) -> dict:
        """
        Get expected timing parameters based on FILENAME (not just task type).

        This handles the distinction between:
          - 90s tasks: Turn_DT, Turn_ST, Walking_DT-AC, Walking_DT-TMB, Walking_DT_DM
          - 150s tasks: plain Walking_DT, Walking_ST, DT, ST

        Args:
            file_basename: Original filename for pattern matching
            task_type: Detected task type

        Returns:
            Dict with baseline_duration, task_duration, end_duration, total_expected
        """
        s = file_basename.upper()

        # --- 90-second tasks (20s baseline + 60s task + 10s end) ---
        # Turn_DT, Turn_ST (walking turn tasks, NOT fTurn)
        if re.search(r'TURN[_-]?(DT|ST)', s) and 'FTURN' not in s and 'F_TURN' not in s:
            logger.debug(f"Timing: 90s (Turn walking task)")
            return {
                "baseline_duration": 20.0,
                "task_duration": 60.0,
                "end_duration": 10.0,
                "total_expected": 90.0
            }

        # Walking_DT with suffixes: DT-AC, DT-TMB, DT_DM, ST-AC, etc.
        # Match DT or ST followed by a hyphen or underscore and more letters
        if re.search(r'(DT|ST)[_-](AC|TMB|DM)\b', s):
            logger.debug(f"Timing: 90s (DT/ST variant task)")
            return {
                "baseline_duration": 20.0,
                "task_duration": 60.0,
                "end_duration": 10.0,
                "total_expected": 90.0
            }

        # --- 150-second tasks (20s baseline + 120s task + 10s end) ---
        if task_type in ('DT', 'ST', 'LongWalk'):
            logger.debug(f"Timing: 150s (standard long walk)")
            return {
                "baseline_duration": 20.0,
                "task_duration": 120.0,
                "end_duration": 10.0,
                "total_expected": 150.0
            }

        # --- Event-dependent tasks: timing not used for baseline (events required) ---
        # Return a reasonable default anyway
        logger.debug(f"Timing: event-dependent task, default 150s")
        return {
            "baseline_duration": 20.0,
            "task_duration": 120.0,
            "end_duration": 10.0,
            "total_expected": 150.0
        }

    def _process_pipeline_stages(self, data: pd.DataFrame,
                                 output_dir: str, file_basename: str,
                                 events: Optional[pd.DataFrame] = None,
                                 task_type: str = None,
                                 subject: str = None,
                                 global_ylim: Optional[Tuple[float, float]] = None) -> Optional[pd.DataFrame]:
        """Process data through all pipeline stages with proper column handling for loader-renamed columns.

        Pipeline order (matches FullCapProcessor): SQI (pre-TDDR) -> TDDR ->
        OD-to-concentration (post-TDDR) -> SCR -> FIR -> baseline -> trim.
        """
        # ====== DIAGNOSTIC: Check input OD values ======
        od_cols = [col for col in data.columns
                   if any(kw in col for kw in ['WL', 'wavelength'])
                   and pd.api.types.is_numeric_dtype(data[col])]

        if od_cols:
            logger.info(f" DIAGNOSTIC: Input OD values (first 5 rows):")
            for col in od_cols[:2]:  # Show first 2 OD columns
                sample_vals = data[col].iloc[:5].values
                logger.info(f"   {col}: {sample_vals}")
        # CRITICAL FIX: Properly identify OD columns after loader renaming
        # Your loader renames columns to format: CH{i}_WL{wavelength}
        od_cols = [col for col in data.columns
                   if re.match(r'CH\d+_WL\d+', col)  # Matches CH0_WL846, CH0_WL757, etc.
                   and pd.api.types.is_numeric_dtype(data[col])]

        if not od_cols:
            logger.error(" No OD columns found with pattern CH*_WL*")
            # Fallback: look for any columns with WL
            od_cols = [col for col in data.columns
                       if 'WL' in col
                       and pd.api.types.is_numeric_dtype(data[col])]

        if not od_cols:
            logger.error(" No OD columns found after fallback")
            return None

        logger.info(f"Found {len(od_cols)} OD columns: {od_cols}")

        # DEBUG: Check OD values before conversion
        logger.info("DEBUG: Checking OD values before concentration conversion")
        for col in od_cols[:4]:  # Check first 4 columns
            od_values = data[col].values
            logger.info(f"   {col}: min={od_values.min():.6f}, max={od_values.max():.6f}, mean={od_values.mean():.6f}")

        # Group channels by CH number (not by adjacent columns)
        channel_groups = {}

        # Extract unique channel numbers and group their wavelengths
        for col in od_cols:
            match = re.match(r'CH(\d+)_WL(\d+)', col)
            if match:
                ch_num = match.group(1)
                wavelength = match.group(2)
                channel_id = f"CH{ch_num}"

                if channel_id not in channel_groups:
                    channel_groups[channel_id] = {}

                channel_groups[channel_id][wavelength] = col

        logger.info(f"Created {len(channel_groups)} channel groups:")
        for ch_id, wavelengths in channel_groups.items():
            logger.info(f"   {ch_id}: {wavelengths}")

        # Verify each channel has exactly 2 wavelengths
        valid_channels = {}
        for ch_id, wavelengths in channel_groups.items():
            if len(wavelengths) == 2:
                valid_channels[ch_id] = wavelengths
                logger.info(f" Valid channel {ch_id}: {list(wavelengths.keys())}nm")
            else:
                logger.warning(f" Channel {ch_id} has {len(wavelengths)} wavelengths, expected 2")

        if not valid_channels:
            logger.error(" No valid channels with 2 wavelengths found")
            return None

        # ─────────────────────────────────────────────────────────────────────────
        # PIPELINE ORDER (matches FullCapProcessor):
        #   quality metrics -> TDDR -> OD-to-concentration -> SCR -> FIR -> baseline
        #
        # SQI is computed on the PRE-TDDR signal, because TDDR is a
        # motion-correction step and SQI is designed to detect exactly the
        # artifacts TDDR removes; evaluating post-TDDR would systematically
        # inflate the score. The "raw" concentration plot is likewise
        # pre-TDDR, since it's meant to show the minimally processed signal.
        # TDDR is then applied to the OD, concentration is re-derived from the
        # corrected OD, and that post-TDDR concentration feeds SCR / FIR /
        # baseline / the final output.
        # ─────────────────────────────────────────────────────────────────────────

        # 1) CONVERT OD TO CONCENTRATION (PRE-TDDR) — used for SQI + "raw" plot
        logger.info("Converting pre-TDDR OD to concentration (for SQI + raw plotting)")
        concentration_data_pre_tddr = self._convert_od_to_concentration(data, od_cols, valid_channels)

        if concentration_data_pre_tddr is None or concentration_data_pre_tddr.empty:
            logger.error(" Failed to convert OD to concentration")
            return None

        logger.info(f" Converted to pre-TDDR concentration: {len(concentration_data_pre_tddr.columns)} columns")

        # ─────────────────────────────────────────────────────────────────────────
        # DIAGNOSTIC: Store data at each processing stage for comparison plots
        # ─────────────────────────────────────────────────────────────────────────
        diagnostic_stages = {}

        # Stage 1: Post-MBLL, pre-TDDR (raw concentration)
        diagnostic_stages["1_Post-MBLL"] = concentration_data_pre_tddr.copy()

        # ─────────────────────────────────────────────────────────────────────────
        # 1.5) PLOT "RAW" CONCENTRATION DATA (post-Beer-Lambert, pre-TDDR/SCR/filtering)
        # This is the "minimally processed" data showing physiologically meaningful
        # concentrations before signal processing removes noise/artifacts
        # ─────────────────────────────────────────────────────────────────────────
        self._plot_raw_concentration_data(
            data=data,
            concentration_data=concentration_data_pre_tddr,
            output_dir=output_dir,
            file_basename=file_basename,
            subject=subject,
            global_ylim=global_ylim,
            condition=task_type,
            events=events
        )

        # 2) Calculate quality (any combination of SQI/SCI/PSP per enabled_metrics)
        #    using the PRE-TDDR OD and PRE-TDDR concentration data.
        excluded_channels, quality_report = self._calculate_quality_and_filter(
            data, valid_channels, output_dir, file_basename, concentration_data_pre_tddr
        )
        self._last_quality_report = quality_report

        # 3) APPLY TDDR to the OD signals, then re-derive concentration from the
        #    corrected OD.
        logger.info("Applying TDDR motion correction to OD signals")
        data_tddr = self._apply_tddr(data)

        logger.info("Re-converting TDDR-corrected OD to concentration")
        concentration_data = self._convert_od_to_concentration(data_tddr, od_cols, valid_channels)

        if concentration_data is None or concentration_data.empty:
            logger.error(" Failed to convert TDDR-corrected OD to concentration")
            return None

        logger.info(f" Converted post-TDDR concentration: {len(concentration_data.columns)} columns")

        # Stage 2: Post-TDDR concentration
        diagnostic_stages["2_Post-TDDR"] = concentration_data.copy()

        # 4) Build working dataframe - START FRESH with only metadata + concentration
        logger.info(" Building working dataframe with ONLY concentration data")

        # CRITICAL: Start with ONLY metadata columns, NO OD columns
        metadata_cols = []
        if 'Sample number' in data.columns:
            metadata_cols.append('Sample number')
        if 'Time (s)' in data.columns:
            metadata_cols.append('Time (s)')
        if 'Event' in data.columns:
            metadata_cols.append('Event')

        working_data = data[metadata_cols].copy()
        logger.info(f"Started with {len(metadata_cols)} metadata columns: {metadata_cols}")

        for col in concentration_data.columns:
            working_data[col] = concentration_data[col]

        logger.info(f"Added {len(concentration_data.columns)} concentration columns")
        logger.info(f"Working data now has {len(working_data.columns)} total columns")

        # 5) Apply quality filtering if enabled (exclusions were computed pre-TDDR,
        #    using whichever of SQI/SCI/PSP are in self.enabled_metrics)
        if self.enable_quality_filtering and excluded_channels:
            logger.info(f" Quality filtering enabled: Excluding {len(excluded_channels)} channels")

            # Determine which concentration columns to exclude based on OD channel exclusions
            concentration_cols_to_exclude = []
            for excluded_od_col in excluded_channels:
                # Find corresponding concentration columns
                for conc_col in working_data.columns:
                    # Match channel pattern (e.g., CH0 corresponds to CH0_HbO and CH0_HbR)
                    channel_match = re.search(r'CH\d+', excluded_od_col)
                    if channel_match and channel_match.group() in conc_col:
                        concentration_cols_to_exclude.append(conc_col)

            if concentration_cols_to_exclude:
                concentration_cols_to_exclude = list(set(concentration_cols_to_exclude))
                working_data = working_data.drop(columns=concentration_cols_to_exclude)
                logger.info(f"Excluded {len(concentration_cols_to_exclude)} concentration columns based on quality metrics")
        else:
            logger.info(f" Quality filtering disabled or nothing to exclude: Using all {len(concentration_data.columns)} concentration channels")

        # Get concentration signal columns for further processing
        signal_cols = [col for col in working_data.columns
                       if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb'])
                       and pd.api.types.is_numeric_dtype(working_data[col])]

        if not signal_cols:
            logger.error(" No concentration signal columns found after SQI filtering")
            return None

        logger.info(f"Processing {len(signal_cols)} concentration channels")

        # Verify values are in reasonable range
        sample_mean = working_data[signal_cols].iloc[:10].mean().mean()
        logger.info(f" Sample concentration mean (first 10 rows): {sample_mean:.6f} µM")

        if abs(sample_mean) > 50:
            logger.error(f" ERROR: Concentration values are too large: {sample_mean:.6f} µM")
            logger.error(f"   Expected: -10 to +10 µM typically")
            logger.error(f"   This indicates OD data is being used instead of concentration!")

            logger.error(f"   Working data columns: {list(working_data.columns)}")

            # Check if any OD columns snuck in
            od_columns_present = [col for col in working_data.columns if 'WL' in col]
            if od_columns_present:
                logger.error(f"    FOUND OD COLUMNS IN WORKING DATA: {od_columns_present}")
                logger.error(f"   These should NOT be here!")

        signal_slice = working_data[signal_cols].copy()

        # ─── CONCENTRATION PROCESSING PIPELINE ──────────────────────────────────

        # 6) SCR (on post-TDDR concentration data)
        scr_data = self._apply_scr(signal_slice)

        # Stage 3: Post-SCR
        diagnostic_stages["3_Post-SCR"] = scr_data.copy()

        # 7) FIR FILTERING (on concentration data)
        logger.info("Applying FIR bandpass filter to concentration data (0.01-0.1 Hz)")
        fir_filtered_data = self._apply_fir_filter(scr_data)

        # Stage 4: Post-Filter
        diagnostic_stages["4_Post-Filter"] = fir_filtered_data.copy()

        # Replace ONLY the concentration columns in working_data with filtered versions
        for col in fir_filtered_data.columns:
            working_data[col] = fir_filtered_data[col]

        logger.info(f"Applied filtering to {len(fir_filtered_data.columns)} concentration channels")

        # 8) Baseline Correction
        baseline_corrected = self._apply_baseline_correction(working_data, events, task_type)

        if baseline_corrected is not None:
            # Stage 5: Post-Baseline
            # Extract just the signal columns for the diagnostic
            bc_signal_cols = [col for col in baseline_corrected.columns
                              if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb'])
                              and pd.api.types.is_numeric_dtype(baseline_corrected[col])]
            if bc_signal_cols:
                diagnostic_stages["5_Post-Baseline"] = baseline_corrected[bc_signal_cols].copy()

            # ─────────────────────────────────────────────────────────────────────
            # CREATE DIAGNOSTIC PLOTS FOR EACH STAGE
            # ─────────────────────────────────────────────────────────────────────
            logger.info(" Creating diagnostic plots for each processing stage...")
            for stage_name, stage_data in diagnostic_stages.items():
                stage_num = int(stage_name.split('_')[0])
                stage_label = stage_name.split('_', 1)[1]
                self._plot_diagnostic_stage(
                    data=stage_data,
                    output_dir=output_dir,
                    file_basename=file_basename,
                    subject=subject,
                    condition=task_type,
                    stage_name=stage_label,
                    stage_number=stage_num,
                    events=events
                )

            # Create summary plot with all stages
            self._create_diagnostic_summary_plot(
                stages_data=diagnostic_stages,
                output_dir=output_dir,
                file_basename=file_basename,
                subject=subject,
                condition=task_type,
                events=events
            )

            # 9) Apply post-event trimming
            trimmed_data = self._apply_post_event_trimming(baseline_corrected, events, task_type)

            # Final verification
            final_signal_cols = [col for col in trimmed_data.columns
                                 if any(kw in col for kw in ['HbO', 'HbR'])
                                 and pd.api.types.is_numeric_dtype(trimmed_data[col])]

            if final_signal_cols:
                final_mean = trimmed_data[final_signal_cols].mean().mean()
                logger.info(f" Final processed data mean: {final_mean:.6f} µM")

                if abs(final_mean) > 50:
                    logger.error(f" WARNING: Final values still too large: {final_mean:.6f} µM")

            # Return the fully processed CONCENTRATION data
            return trimmed_data

        return None

    def _plot_raw_concentration_data(self, data: pd.DataFrame,
                                     concentration_data: pd.DataFrame,
                                     output_dir: str,
                                     file_basename: str,
                                     subject: str,
                                     global_ylim: Optional[Tuple[float, float]],
                                     condition: str,
                                     events: Optional[pd.DataFrame] = None) -> None:
        """
        Plot "raw" concentration data - post-Beer-Lambert conversion, pre-TDDR/SCR/filtering.


        Note: Short channels (CH3, CH5) are excluded from averaging to match final output.

        Args:
            data: Original data with metadata columns (Sample number, Time, Event)
            concentration_data: DataFrame with HbO/HbR concentration columns
            output_dir: Directory for saving plots
            file_basename: Base filename for plot naming
            subject: Subject identifier
            global_ylim: Optional y-axis limits for consistent scaling
            condition: Task type/condition label
            events: Optional event markers DataFrame
        """
        # Find HbO and HbR columns, exclude short channels
        o2hb_cols = self._get_long_channel_cols(concentration_data.columns, 'oxy')
        hhb_cols = self._get_long_channel_cols(concentration_data.columns, 'deoxy')
        combined_cols = o2hb_cols + hhb_cols

        if not combined_cols:
            logger.warning("No long-channel concentration columns found for raw plotting")
            return

        logger.info(
            f" Raw concentration plot using {len(o2hb_cols)} HbO and {len(hhb_cols)} HbR long channels (excluding CH3, CH5)")

        # Create condition-specific output directory
        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        # Build a combined df for plotting
        plot_df = concentration_data[combined_cols].copy()

        # aDdd time information
        if 'Time (s)' in data.columns:
            plot_df['Time (s)'] = data['Time (s)'].values
        else:
            plot_df['Time (s)'] = np.arange(len(plot_df)) / self.fs

        # ─── Plot 1: Individual channels (long channels only) ───
        try:
            fig, axes, ylim = plot_channels_separately(
                plot_df[combined_cols],
                fs=self.fs,
                title=f"{file_basename} - Raw Concentration (Post-MBLL, Pre-Processing)",
                subject=subject,
                condition=condition,
                y_lim=global_ylim
            )
            self._save_figure(fig,
                              os.path.join(condition_dir, f"raw_concentration_individual_channels_{condition}.png"))
            logger.info(f"Saved raw concentration individual channels plot")
        except Exception as e:
            logger.warning(f"Failed to create individual channels plot: {e}")

        # ─── Plot 2: Overall averaged signals with events ───
        if o2hb_cols and hhb_cols:
            try:
                avg_o2hb = plot_df[o2hb_cols].mean(axis=1)
                avg_hhb = plot_df[hhb_cols].mean(axis=1)

                overall_df = pd.DataFrame({
                    "Time (s)": np.arange(len(plot_df)) / self.fs,
                    "grand oxy": avg_o2hb,
                    "grand deoxy": avg_hhb
                })

                # Clean events for plotting
                clean_events = None
                if events is not None and not events.empty:
                    valid = events[
                        events['Sample number'].notna() &
                        events['Event'].notna() &
                        (events['Event'] != '') &
                        (events['Sample number'] >= 0) &
                        (events['Sample number'] < len(plot_df))
                        ].copy()
                    if not valid.empty:
                        clean_events = valid

                fig, ylim = plot_overall_signals(
                    overall_df,
                    fs=self.fs,
                    title=f"{file_basename} - Raw Concentration Overall (Post-MBLL, Pre-Processing)",
                    subject=subject,
                    condition=condition,
                    y_lim=global_ylim,
                    events=clean_events
                )
                self._save_figure(fig, os.path.join(condition_dir, f"raw_concentration_overall_{condition}.png"))
                logger.info(f"Saved raw concentration overall plot")

                # Log some statistics about the raw concentration data
                logger.info(f"Raw concentration statistics (long channels only):")
                logger.info(f"   HbO mean: {avg_o2hb.mean():.4f} µM, std: {avg_o2hb.std():.4f} µM")
                logger.info(f"   HbR mean: {avg_hhb.mean():.4f} µM, std: {avg_hhb.std():.4f} µM")

            except Exception as e:
                logger.warning(f"Failed to create overall signals plot: {e}")

    def _plot_diagnostic_stage(self, data: pd.DataFrame,
                               output_dir: str,
                               file_basename: str,
                               subject: str,
                               condition: str,
                               stage_name: str,
                               stage_number: int,
                               events: Optional[pd.DataFrame] = None) -> None:
        """
        Plot data at a specific processing stage for diagnostic purposes.

        Note: Short channels (CH3, CH5) are excluded from averaging to match final output.

        Args:
            data: DataFrame with HbO/HbR concentration columns
            output_dir: Directory for saving plots
            file_basename: Base filename for plot naming
            subject: Subject identifier
            condition: Task type/condition label
            stage_name: e.g., "Post-SCR"
            stage_number: Numeric stage identifier for ordering (1, 2, 3, etc.)
            events: Optional event markers DataFrame
        """
        # Find HbO and HbR columns, EXCLUDING short channels
        o2hb_cols = self._get_long_channel_cols(data.columns, 'oxy')
        hhb_cols = self._get_long_channel_cols(data.columns, 'deoxy')

        if not o2hb_cols or not hhb_cols:
            logger.warning(f"No long-channel concentration columns found for diagnostic plot at stage: {stage_name}")
            return

        # Create diagnostic output directory
        diag_dir = os.path.join(output_dir, condition, "diagnostic_stages")
        os.makedirs(diag_dir, exist_ok=True)

        try:
            # Calculate averages (long channels only)
            avg_o2hb = data[o2hb_cols].mean(axis=1)
            avg_hhb = data[hhb_cols].mean(axis=1)

            overall_df = pd.DataFrame({
                "Time (s)": np.arange(len(data)) / self.fs,
                "grand oxy": avg_o2hb,
                "grand deoxy": avg_hhb
            })

            # Clean events for plotting
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
                title=f"{file_basename} - Stage {stage_number}: {stage_name}",
                subject=subject,
                condition=condition,
                y_lim=None,
                events=clean_events
            )

            output_path = os.path.join(diag_dir,
                                       f"stage_{stage_number}_{stage_name.replace(' ', '_').replace('-', '_')}_{condition}.png")
            self._save_figure(fig, output_path)

            # Log statistics for this stage
            logger.info(f" Stage {stage_number} ({stage_name}) statistics (long channels only):")
            logger.info(f"   HbO mean: {avg_o2hb.mean():.4f} µM, std: {avg_o2hb.std():.4f} µM")
            logger.info(f"   HbR mean: {avg_hhb.mean():.4f} µM, std: {avg_hhb.std():.4f} µM")
            logger.info(f"   Saved to: {output_path}")

        except Exception as e:
            logger.warning(f"Failed to create diagnostic plot for stage {stage_name}: {e}")

    def _create_diagnostic_summary_plot(self, stages_data: Dict[str, pd.DataFrame],
                                        output_dir: str,
                                        file_basename: str,
                                        subject: str,
                                        condition: str,
                                        events: Optional[pd.DataFrame] = None) -> None:
        """
        Create a multi-panel summary plot showing all processing stages side by side.

        Note: Short channels (CH3, CH5) are excluded from averaging to match final output.

        Args:
            stages_data: Dict mapping stage names to DataFrames
            output_dir: Directory for saving plots
            file_basename: Base filename for plot naming
            subject: Subject identifier
            condition: Task type/condition label
            events: Optional event markers DataFrame
        """
        n_stages = len(stages_data)
        if n_stages == 0:
            return

        # Create diagnostic output directory
        diag_dir = os.path.join(output_dir, condition, "diagnostic_stages")
        os.makedirs(diag_dir, exist_ok=True)

        fig, axes = plt.subplots(n_stages, 1, figsize=(14, 4 * n_stages), sharex=True)
        if n_stages == 1:
            axes = [axes]

        fig.suptitle(
            f"{file_basename} - Processing Pipeline Stages\nSubject: {subject}\n(Long channels only, excluding CH3 & CH5)",
            fontsize=12)

        # Clean events once
        clean_events = None
        if events is not None and not events.empty:
            # Get max length from any stage
            max_len = max(len(df) for df in stages_data.values())
            valid = events[
                events['Sample number'].notna() &
                events['Event'].notna() &
                (events['Event'] != '') &
                (events['Sample number'] >= 0) &
                (events['Sample number'] < max_len)
                ].copy()
            if not valid.empty:
                clean_events = valid

        for idx, (stage_name, data) in enumerate(stages_data.items()):
            ax = axes[idx]

            # Find concentration columns, EXCLUDING short channels
            o2hb_cols = self._get_long_channel_cols(data.columns, 'oxy')
            hhb_cols = self._get_long_channel_cols(data.columns, 'deoxy')

            if not o2hb_cols or not hhb_cols:
                ax.text(0.5, 0.5, f"No long-channel data for {stage_name}", ha='center', va='center')
                ax.set_title(stage_name)
                continue

            # Calculate averages (long channels only)
            avg_o2hb = data[o2hb_cols].mean(axis=1)
            avg_hhb = data[hhb_cols].mean(axis=1)
            time = np.arange(len(data)) / self.fs

            # Plot signals
            ax.plot(time, avg_o2hb, 'r-', label='HbO', linewidth=1.2)
            ax.plot(time, avg_hhb, 'b-', label='HbR', linewidth=1.2)

            # Add events
            if clean_events is not None:
                ylim = ax.get_ylim()
                for _, row in clean_events.iterrows():
                    event_time = float(row['Sample number']) / self.fs
                    if 0 <= event_time <= time[-1]:
                        ax.axvline(x=event_time, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                        ax.text(event_time, ylim[1] * 0.95, str(row['Event']),
                                rotation=90, va='top', ha='right', fontsize=7, alpha=0.8)

            # Formatting
            ax.set_title(
                f"{stage_name} (HbO: {avg_o2hb.mean():.2f}±{avg_o2hb.std():.2f}, HbR: {avg_hhb.mean():.2f}±{avg_hhb.std():.2f} µM)")
            ax.set_ylabel("Δ[Hb] (µM)")
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(diag_dir, f"SUMMARY_all_stages_{condition}.png")
        self._save_figure(fig, output_path)
        logger.info(f" Saved diagnostic summary plot to: {output_path}")

    def _apply_fir_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply FIR filter to the data with appropriate parameters."""
        try:
            logger.info("Applying FIR bandpass filter: 0.01-0.1 Hz, order=1000")
            return fir_filter(data, order=1000, Wn=[0.01, 0.1], fs=int(self.fs))
        except Exception as e:
            logger.warning(f"FIR filtering failed: {str(e)}")
            return data

    def _apply_post_event_trimming(self, data: pd.DataFrame, events: pd.DataFrame,
                                   task_type: str) -> pd.DataFrame:
        """
        Apply trimming after walking/task start event for quality control.
        IMPROVED: Better preservation of critical start events.
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

        # IMPROVED: Check if any critical events fall in the trim region
        critical_event_names = self.task_walking_events.get(task_type, [])
        critical_event_names_upper = [e.upper() for e in critical_event_names]

        events_clean = events.copy()
        events_clean['Event_Upper'] = events_clean['Event'].astype(str).str.upper()

        # Find any critical events in the trim region
        events_in_trim_region = events_clean[
            (events_clean['Sample number'] >= walking_start_sample) &
            (events_clean['Sample number'] < trim_start_sample) &
            (events_clean['Event_Upper'].isin(critical_event_names_upper))
            ]

        if not events_in_trim_region.empty:
            logger.warning(f" Found {len(events_in_trim_region)} critical events in trim region:")
            for _, evt in events_in_trim_region.iterrows():
                logger.warning(f"   - {evt['Event']} at sample {evt['Sample number']}")
            logger.warning(f"   Adjusting trim to preserve these events")

            # Adjust trim to start after the last critical event in the region
            last_critical_sample = events_in_trim_region['Sample number'].max()
            trim_start_sample = int(last_critical_sample) + 1

            logger.info(f"   Adjusted trim start to sample {trim_start_sample}")

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
        logger.info(f"Applied post-walking trimming: removed {removed} samples (~{removed / self.fs:.1f}s)")

        return trimmed_data

    def _apply_z_transformation(self, data: pd.DataFrame, signal_cols: List[str]) -> pd.DataFrame:
        """Apply Z-transformation using the dedicated module."""
        return z_transformation(data, signal_cols)

    def _find_walking_start_event(self, events: pd.DataFrame, task_type: str) -> Optional[int]:
        """
        Find the event marker that indicates walking/task execution start.
        Enhanced to handle mislabeled events for long walk tasks.
        """
        if task_type not in self.task_walking_events:
            logger.warning(f" Unknown task type for walking start detection: {task_type}")
            return None

        possible_events = self.task_walking_events[task_type]
        events_clean = events.copy()
        events_clean['Event_Upper'] = events_clean['Event'].astype(str).str.strip().str.upper()

        for event_name in possible_events:
            matching_events = events_clean[events_clean['Event_Upper'] == event_name.upper()]
            if not matching_events.empty:
                walking_start_sample = matching_events.iloc[0]['Sample number']
                logger.info(f"Found walking start event '{event_name}' at sample {walking_start_sample}")
                return int(walking_start_sample)

        task_config = self.task_types.get(task_type, {})
        if task_config.get('type') == 'long_walk':
            logger.info(f" Long walk task ({task_type}) - checking for alternative event patterns")
            s_events = events_clean[events_clean['Event_Upper'].str.match(r'S[1-9]')]
            if len(s_events) >= 2:
                s_events_sorted = s_events.sort_values('Sample number').reset_index(drop=True)
                second_s_event = s_events_sorted.iloc[1]
                walking_start_sample = second_s_event['Sample number']
                walking_start_name = second_s_event['Event']
                logger.info(
                    f" Long walk fallback: Using '{walking_start_name}' as walking start at sample {walking_start_sample}")
                return int(walking_start_sample)

            s1_events = events_clean[events_clean['Event_Upper'] == 'S1']
            if not s1_events.empty:
                s1_sample = s1_events.iloc[0]['Sample number']
                events_after_s1 = events_clean[events_clean['Sample number'] > s1_sample]
                if not events_after_s1.empty:
                    next_event = events_after_s1.sort_values('Sample number').iloc[0]
                    walking_start_sample = next_event['Sample number']
                    logger.info(
                        f" Long walk fallback: Using first event after S1 ('{next_event['Event']}') as walking start at sample {walking_start_sample}")
                    return int(walking_start_sample)

        elif task_config.get('type') == 'event_dependent':
            s1_events = events_clean[events_clean['Event_Upper'] == 'S1']
            if not s1_events.empty:
                s1_sample = s1_events.iloc[0]['Sample number']
                s2_events = events_clean[events_clean['Event_Upper'] == 'S2']
                s2_after_s1 = s2_events[s2_events['Sample number'] > s1_sample]
                if not s2_after_s1.empty:
                    walking_start_sample = s2_after_s1.iloc[0]['Sample number']
                    logger.info(
                        f"Event-dependent task: Using S2 after S1 as walking start at sample {walking_start_sample}")
                    return int(walking_start_sample)

                events_after_s1 = events_clean[events_clean['Sample number'] > s1_sample]
                if not events_after_s1.empty:
                    next_event = events_after_s1.iloc[0]
                    walking_start_sample = next_event['Sample number']
                    logger.info(
                        f" Event-dependent fallback: Using first event after S1 ('{next_event['Event']}') at sample {walking_start_sample}")
                    return int(walking_start_sample)

        if task_type == 'LShape':
            s2_events = events_clean[events_clean['Event_Upper'] == 'S2']
            if not s2_events.empty:
                s2_sample = s2_events.iloc[0]['Sample number']
                w1_after_s2 = events_clean[
                    (events_clean['Event_Upper'] == 'W1') & (events_clean['Sample number'] > s2_sample)]
                if not w1_after_s2.empty:
                    walking_start_sample = w1_after_s2.iloc[0]['Sample number']
                    logger.info(f" Found L-Shape walking start 'W1' after S2 at sample {walking_start_sample}")
                    return int(walking_start_sample)

                events_after_s2 = events_clean[events_clean['Sample number'] > s2_sample]
                if not events_after_s2.empty:
                    next_event = events_after_s2.iloc[0]
                    walking_start_sample = next_event['Sample number']
                    logger.info(
                        f" L-Shape fallback: Using first event after S2 ('{next_event['Event']}') at sample {walking_start_sample}")
                    return int(walking_start_sample)

        if len(events_clean) >= 2:
            events_sorted = events_clean.sort_values('Sample number').reset_index(drop=True)
            second_event = events_sorted.iloc[1]
            walking_start_sample = second_event['Sample number']
            logger.info(
                f" General fallback: Using second event ('{second_event['Event']}') as walking start at sample {walking_start_sample}")
            return int(walking_start_sample)

        logger.warning(f" Could not find walking start event for {task_type}")
        return None

    def _calculate_quality_and_filter(self, data: pd.DataFrame,
                                      channel_groups: dict,
                                      output_dir: str, file_basename: str,
                                      concentration_data: pd.DataFrame) -> Tuple[List[str], QualityReport]:
        """Calculate channel quality using whichever of SQI/SCI/PSP are in
        `self.enabled_metrics`, all computed on PRE-TDDR OD (+ PRE-TDDR
        concentration, for SQI only). A channel fails if ANY enabled,
        successfully-computed metric fails its threshold. Short channels are
        spared from exclusion unless `exclude_failing_short_channels=True`.

        Returns
        -------
        (excluded_od_columns, quality_report)
            excluded_od_columns : OD column names for channels that should be
                dropped downstream (empty unless enable_quality_filtering=True).
            quality_report : QualityReport with one ChannelQuality per channel,
                for logging/CSV output and for the caller (process_file) to
                surface to BatchProcessor/PipelineManager.
        """
        report = QualityReport(
            metrics_used=self.enabled_metrics,
            sqi_threshold=self.sqi_threshold,
            sci_threshold=self.sci_threshold,
            psp_threshold=self.psp_threshold,
        )
        excluded_channels: List[str] = []

        logger.info(f" Quality calculation ({'+'.join(self.enabled_metrics) or 'none'}) for {file_basename}")
        logger.info(f"   Thresholds: SQI>={self.sqi_threshold} SCI>={self.sci_threshold} PSP>={self.psp_threshold}")
        logger.info(f"   Quality filtering: {self.enable_quality_filtering} "
                    f"(exclude failing short channels: {self.exclude_failing_short_channels})")
        logger.info(f"   Total channels to evaluate: {len(channel_groups)}")

        for ch_id_str, wavelengths in channel_groups.items():
            ch_num = int(re.match(r'CH(\d+)', ch_id_str).group(1))
            is_short = ch_num in SHORT_CHANNEL_IDS

            if len(wavelengths) != 2:
                logger.warning(f" Channel {ch_id_str} has {len(wavelengths)} wavelength(s), need 2 for quality metrics")
                cq = ChannelQuality(ch_num, is_short, passed=False,
                                    reasons=("insufficient wavelengths",))
                report.channels.append(cq)
                if self.enable_quality_filtering and not (is_short and not self.exclude_failing_short_channels):
                    excluded_channels.extend(list(wavelengths.values()))
                continue

            wl_keys = list(wavelengths.keys())
            od1_col, od2_col = wavelengths[wl_keys[0]], wavelengths[wl_keys[1]]
            od1_signal = data[od1_col].to_numpy(dtype=np.float64)
            od2_signal = data[od2_col].to_numpy(dtype=np.float64)

            sqi_val = sci_val = psp_val = None
            reasons: List[str] = []

            try:
                if "sqi" in self.enabled_metrics:
                    oxy_col = deoxy_col = None
                    for col in concentration_data.columns:
                        if ch_id_str in col:
                            if any(kw in col for kw in ['HbO', 'O2Hb']):
                                oxy_col = col
                            elif any(kw in col for kw in ['HbR', 'HHb']):
                                deoxy_col = col
                    if oxy_col and deoxy_col:
                        oxy_signal = concentration_data[oxy_col].to_numpy(dtype=np.float64)
                        deoxy_signal = concentration_data[deoxy_col].to_numpy(dtype=np.float64)
                    else:
                        logger.warning(f"No concentration data found for {ch_id_str}, using OD data as proxy for SQI")
                        oxy_signal, deoxy_signal = od1_signal, od2_signal

                    sqi_val = float(SQI(od1_signal, od2_signal, oxy_signal, deoxy_signal, self.fs))
                    if np.isnan(sqi_val):
                        reasons.append("SQI could not be computed (NaN)")
                    elif sqi_val < self.sqi_threshold:
                        reasons.append(f"SQI {sqi_val:.2f} < {self.sqi_threshold:.2f}")

                if "sci" in self.enabled_metrics:
                    sci_val = float(scalp_coupling_index(od1_signal, od2_signal, self.fs))
                    if np.isnan(sci_val):
                        reasons.append("SCI could not be computed (NaN)")
                    elif sci_val < self.sci_threshold:
                        reasons.append(f"SCI {sci_val:.3f} < {self.sci_threshold:.2f}")

                if "psp" in self.enabled_metrics:
                    psp_val = float(peak_spectral_power(od1_signal, od2_signal, self.fs))
                    if np.isnan(psp_val):
                        reasons.append("PSP could not be computed (NaN)")
                    elif psp_val < self.psp_threshold:
                        reasons.append(f"PSP {psp_val:.3f} < {self.psp_threshold:.2f}")

            except Exception as e:
                logger.warning(f" Quality calculation failed for {ch_id_str}: {str(e)}")
                reasons.append(f"quality calculation error: {e}")

            failed = bool(reasons)
            spared = failed and is_short and not self.exclude_failing_short_channels
            if spared:
                reasons = (f"short channel kept despite: " + "; ".join(reasons),)
                passed = True
            else:
                passed = not failed
                reasons = tuple(reasons)

            cq = ChannelQuality(ch_num, is_short, passed=passed,
                                sqi=sqi_val, sci=sci_val, psp=psp_val, reasons=reasons)
            report.channels.append(cq)

            if failed:
                status = "SPARED (short channel)" if spared else "EXCLUDED" if self.enable_quality_filtering else "KEPT (filtering disabled)"
                logger.info(f"    FAIL {ch_id_str} [{status}]: {cq.reason_text} "
                           f"(SQI={sqi_val}, SCI={sci_val}, PSP={psp_val})")
                if self.enable_quality_filtering and not spared:
                    excluded_channels.extend(list(wavelengths.values()))
            else:
                logger.debug(f"    PASS {ch_id_str}: SQI={sqi_val}, SCI={sci_val}, PSP={psp_val}")

        logger.info(f" {report.summary_line()}")

        # Save a single combined quality report (replaces the old two-txt-file SQI-only output)
        quality_suffix = "_filtered" if self.enable_quality_filtering else "_unfiltered"
        report_path = os.path.join(
            output_dir, f"{os.path.splitext(file_basename)[0]}_quality_report{quality_suffix}.csv"
        )
        try:
            report.to_dataframe().to_csv(report_path, index=False)
            logger.info(f"Saved quality report to: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save quality report: {e}")

        logger.info(f"QUALITY SUMMARY: {len(report.rejected)} channels excluded of {len(report.channels)} evaluated")
        return excluded_channels, report

    def _apply_tddr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply TDDR (temporal derivative distribution repair) motion correction.

        Only the OD/wavelength signal columns (plus 'Sample number', needed by
        the tddr implementation) are corrected; all other columns (Event,
        Time (s), etc.) pass through unchanged. Mirrors
        FullCapProcessor._apply_tddr so both pipelines apply TDDR the same way,
        in the same position in the pipeline (after quality metrics, before
        OD-to-concentration conversion).
        """
        try:
            signals = data.filter(regex="WL|Sample")
            corrected = tddr(signals, sample_rate=self.fs)
            out = data.copy()
            for col in corrected.columns:
                out[col] = corrected[col]
            return out
        except Exception as e:
            logger.warning(f"TDDR failed: {str(e)}; continuing with uncorrected OD data")
            return data


    def _baseline_window_samples(self, events, task_type, n_samples):
        """Locate the pre-task baseline window (start, end) sample indices used to
        reference OD -> Delta-OD, mirroring _apply_baseline_correction's choice so
        the OD reference and the later concentration baseline agree.

        Priority:
            LShape            : 2nd -> 3rd event markers
            S1 present + W1    : [S1, W1]   (standard pre-walk standing baseline)
            S1 present + S2    : [S1, S2]
            S1 present only    : [S1, S1 + baseline_duration]
            no usable events   : [0, baseline_duration]   (first baseline_duration s)

        Returns (start, end) clamped to [0, n_samples) and required to span at
        least ~1 s, or None if nothing usable (caller then falls back to the
        whole-record per-channel mean).
        """
        min_span = max(1, int(1.0 * self.fs))

        def _clamp(a, b):
            a = int(max(0, min(a, n_samples - 1)))
            b = int(max(0, min(b, n_samples)))
            if b - a < min_span:
                return None
            return (a, b)

        # Task-timing durations for the event-poor fallbacks.
        file_basename = getattr(self, "_current_file_basename", "")
        timing = self._get_task_timing(file_basename, task_type or "Unknown")
        base_samples = int(timing.get("baseline_duration", 20.0) * self.fs)

        if events is not None and not events.empty and "Event" in events.columns:
            ev = events.copy()
            ev["Sample number"] = pd.to_numeric(ev["Sample number"], errors="coerce")
            ev = ev.dropna(subset=["Sample number"])
            ev["_E"] = ev["Event"].astype(str).str.strip().str.upper()
            ev = ev.sort_values("Sample number").reset_index(drop=True)

            # LShape uses the 2nd -> 3rd markers (matches _apply_lshape_baseline).
            if task_type == "LShape" and len(ev) >= 3:
                w = _clamp(ev.iloc[1]["Sample number"], ev.iloc[2]["Sample number"])
                if w:
                    logger.info(f" OD baseline reference: LShape 2nd->3rd events {w}")
                    return w

            s1 = ev[ev["_E"] == "S1"]
            if not s1.empty:
                s1_s = s1.iloc[0]["Sample number"]

                # Preferred: S1 -> W1 (the requested pre-walk standing baseline).
                w1 = ev[(ev["_E"] == "W1") & (ev["Sample number"] > s1_s)]
                if not w1.empty:
                    w = _clamp(s1_s, w1.iloc[0]["Sample number"])
                    if w:
                        logger.info(f" OD baseline reference: S1->W1 window {w}")
                        return w

                # Secondary: S1 -> S2.
                s2 = ev[(ev["_E"] == "S2") & (ev["Sample number"] > s1_s)]
                if not s2.empty:
                    w = _clamp(s1_s, s2.iloc[0]["Sample number"])
                    if w:
                        logger.info(f" OD baseline reference: S1->S2 window {w}")
                        return w

                # S1 only: take baseline_duration seconds after S1.
                w = _clamp(s1_s, s1_s + base_samples)
                if w:
                    logger.info(f" OD baseline reference: S1 + {base_samples} samples {w}")
                    return w

        # No usable events: reference to the first baseline_duration seconds.
        w = _clamp(0, base_samples)
        if w:
            logger.warning(f" OD baseline reference: no S1/W1 markers; using first "
                           f"{base_samples} samples {w}")
        else:
            logger.warning(" OD baseline reference: could not build a baseline window; "
                           "falling back to whole-record mean.")
        return w


    def _baseline_window_samples(self, events, task_type, n_samples):
        """Locate the pre-task baseline window (start, end) sample indices used to
        reference OD -> Delta-OD, mirroring _apply_baseline_correction's choice so
        the OD reference and the later concentration baseline agree.

        Priority:
            LShape            : 2nd -> 3rd event markers
            S1 present + W1    : [S1, W1]   (standard pre-walk standing baseline)
            S1 present + S2    : [S1, S2]
            S1 present only    : [S1, S1 + baseline_duration]
            no usable events   : [0, baseline_duration]   (first baseline_duration s)

        Returns (start, end) clamped to [0, n_samples) and required to span at
        least ~1 s, or None if nothing usable (caller then falls back to the
        whole-record per-channel mean).
        """
        min_span = max(1, int(1.0 * self.fs))

        def _clamp(a, b):
            a = int(max(0, min(a, n_samples - 1)))
            b = int(max(0, min(b, n_samples)))
            if b - a < min_span:
                return None
            return (a, b)

        # Task-timing durations for the event-poor fallbacks.
        file_basename = getattr(self, "_current_file_basename", "")
        timing = self._get_task_timing(file_basename, task_type or "Unknown")
        base_samples = int(timing.get("baseline_duration", 20.0) * self.fs)

        if events is not None and not events.empty and "Event" in events.columns:
            ev = events.copy()
            ev["Sample number"] = pd.to_numeric(ev["Sample number"], errors="coerce")
            ev = ev.dropna(subset=["Sample number"])
            ev["_E"] = ev["Event"].astype(str).str.strip().str.upper()
            ev = ev.sort_values("Sample number").reset_index(drop=True)

            # LShape uses the 2nd -> 3rd markers (matches _apply_lshape_baseline).
            if task_type == "LShape" and len(ev) >= 3:
                w = _clamp(ev.iloc[1]["Sample number"], ev.iloc[2]["Sample number"])
                if w:
                    logger.info(f" OD baseline reference: LShape 2nd->3rd events {w}")
                    return w

            s1 = ev[ev["_E"] == "S1"]
            if not s1.empty:
                s1_s = s1.iloc[0]["Sample number"]

                # Preferred: S1 -> W1 (the requested pre-walk standing baseline).
                w1 = ev[(ev["_E"] == "W1") & (ev["Sample number"] > s1_s)]
                if not w1.empty:
                    w = _clamp(s1_s, w1.iloc[0]["Sample number"])
                    if w:
                        logger.info(f" OD baseline reference: S1->W1 window {w}")
                        return w

                # Secondary: S1 -> S2.
                s2 = ev[(ev["_E"] == "S2") & (ev["Sample number"] > s1_s)]
                if not s2.empty:
                    w = _clamp(s1_s, s2.iloc[0]["Sample number"])
                    if w:
                        logger.info(f" OD baseline reference: S1->S2 window {w}")
                        return w

                # S1 only: take baseline_duration seconds after S1.
                w = _clamp(s1_s, s1_s + base_samples)
                if w:
                    logger.info(f" OD baseline reference: S1 + {base_samples} samples {w}")
                    return w

        # No usable events: reference to the first baseline_duration seconds.
        w = _clamp(0, base_samples)
        if w:
            logger.warning(f" OD baseline reference: no S1/W1 markers; using first "
                           f"{base_samples} samples {w}")
        else:
            logger.warning(" OD baseline reference: could not build a baseline window; "
                           "falling back to whole-record mean.")
        return w

    def _convert_od_to_concentration(self, data, od_cols, channel_groups,
                                     events=None, task_type=None):
        """Convert OD to Delta-concentration using MBLL and Prahl/OMLC extinction
        coefficients (cm^-1/M), with per-channel pathlength and OD referenced to the
        S1->W1 pre-walk baseline window.

        Each channel's OD is referenced to its mean over the baseline window
        (Delta-OD = OD - mean_baseline(OD)) before the MBLL solve, so the output is
        Delta[Hb] anchored to the resting/standing period. If no baseline window can
        be located (events/task_type not supplied, or markers missing), the method
        falls back to referencing against the whole-record per-channel mean.

        The extinction table is restricted to the fine 2 nm range around this
        device's light-source wavelengths (756-759 nm and 846-848 nm); odd-nm
        entries are linearly interpolated from tabulated even-nm neighbours and
        847 nm is interpolated from the 846/848 anchors by np.interp.
        """
        try:
            DPF = 6.0  # confirmed fixed for this device/protocol
            DISTANCE_LONG_CM = 3.5  # long channels: 35 mm source-detector separation
            DISTANCE_SHORT_CM = 1.5  # short channels (CH3/CH5): 15 mm separation

            # Locate the pre-walk baseline window ONCE (same for every channel).
            n_samples = len(data)
            window = self._baseline_window_samples(events, task_type, n_samples)
            if window is not None:
                b_start, b_end = window
                logger.info(f" Converting OD to Delta[Hb] referenced to baseline samples "
                            f"{b_start}-{b_end} (~{(b_end - b_start) / self.fs:.1f}s), "
                            f"per-channel pathlength")
            else:
                b_start = b_end = None
                logger.info(" Converting OD to Delta[Hb] referenced to whole-record mean "
                            "(no baseline window), per-channel pathlength")

            PRAHL_CM1_PER_M = {
                750: (518.0, 1405.24),
                752: (533.2, 1515.32),
                754: (548.4, 1541.76),
                756: (562.0, 1560.48),
                757: (568.0, 1560.48),  # interpolated between 756 and 758
                758: (574.0, 1560.48),
                759: (580.0, 1554.50),  # interpolated between 758 and 760
                760: (586.0, 1548.52),
                762: (598.0, 1508.44),
                764: (610.0, 1459.56),
                836: (1001.2, 692.64),
                838: (1011.6, 692.48),
                839: (1016.8, 692.42),  # interpolated between 838 and 840
                840: (1022.0, 692.36),
                842: (1032.4, 692.20),
                844: (1042.8, 691.96),
                846: (1050.0, 691.76),
                848: (1054.0, 691.52),
                850: (1058.0, 691.32),
                852: (1062.0, 691.08),
            }

            wl_grid = np.array(sorted(PRAHL_CM1_PER_M.keys()), dtype=float)
            hbO2_grid = np.array([PRAHL_CM1_PER_M[int(w)][0] for w in wl_grid], dtype=float)
            hb_grid = np.array([PRAHL_CM1_PER_M[int(w)][1] for w in wl_grid], dtype=float)

            def _eps_at_nm(w):
                if w < wl_grid.min() or w > wl_grid.max():
                    raise ValueError(
                        f"Wavelength {w} nm outside Prahl table range ({wl_grid.min()}-{wl_grid.max()} nm).")
                eps_hbo2 = float(np.interp(w, wl_grid, hbO2_grid))
                eps_hb = float(np.interp(w, wl_grid, hb_grid))
                return eps_hbo2, eps_hb

            def _baseline_ref(sig):
                """Mean of `sig` over the baseline window, or whole-record mean if
                no window / the window is all-NaN."""
                if b_start is not None:
                    ref = np.nanmean(sig[b_start:b_end])
                    if np.isfinite(ref):
                        return ref
                return np.nanmean(sig)

            concentration_data = pd.DataFrame(index=data.index)
            converted_channels = 0

            for ch_id, wavelengths in channel_groups.items():
                if len(wavelengths) != 2:
                    continue

                # Per-channel pathlength: short vs long.
                ch_match = re.match(r'CH(\d+)', str(ch_id))
                ch_num = int(ch_match.group(1)) if ch_match else -1
                is_short = ch_num in SHORT_CHANNEL_IDS
                distance_cm = DISTANCE_SHORT_CM if is_short else DISTANCE_LONG_CM
                L_eff_cm = DPF * distance_cm
                if L_eff_cm <= 0:
                    logger.error(f"Effective pathlength non-positive for {ch_id}; skipping.")
                    continue

                wl_keys = list(wavelengths.keys())
                wl1 = float(wl_keys[0])
                wl2 = float(wl_keys[1])
                od1_col = wavelengths[wl_keys[0]]
                od2_col = wavelengths[wl_keys[1]]

                if od1_col not in data.columns or od2_col not in data.columns:
                    logger.warning(f"Missing OD columns for {ch_id}: {od1_col}, {od2_col}")
                    continue

                od1 = data[od1_col].to_numpy(dtype=np.float64)
                od2 = data[od2_col].to_numpy(dtype=np.float64)

                # Reference to the pre-walk baseline window: absolute OD -> Delta-OD.
                od1 = od1 - _baseline_ref(od1)
                od2 = od2 - _baseline_ref(od2)

                eps1_hbo, eps1_hbr = _eps_at_nm(wl1)
                eps2_hbo, eps2_hbr = _eps_at_nm(wl2)

                det = (eps1_hbo * eps2_hbr - eps2_hbo * eps1_hbr)
                if abs(det) < 1e-12:
                    logger.error(f"Near-singular extinction matrix for {ch_id} ({wl1}nm, {wl2}nm); det={det:e}")
                    continue

                hbO_M = (eps2_hbr * od1 - eps1_hbr * od2) / (L_eff_cm * det)
                hbR_M = (-eps2_hbo * od1 + eps1_hbo * od2) / (L_eff_cm * det)

                hbO_uM = hbO_M * 1e6
                hbR_uM = hbR_M * 1e6

                concentration_data[f"{ch_id} HbO"] = hbO_uM
                concentration_data[f"{ch_id} HbR"] = hbR_uM
                converted_channels += 1

                logger.debug(
                    f"{ch_id} ({'short' if is_short else 'long'}, L={L_eff_cm:.1f}cm, "
                    f"{wl1:.0f}/{wl2:.0f} nm) baseline-referenced Delta means "
                    f"HbO={np.nanmean(hbO_uM):.3f} uM, HbR={np.nanmean(hbR_uM):.3f} uM")

            logger.info(f" Converted {converted_channels} channels OD -> Delta[Hb] (uM), "
                        f"pre-walk baseline-referenced")
            return concentration_data if not concentration_data.empty else None

        except Exception as e:
            logger.error(f"OD to concentration conversion failed: {str(e)}", exc_info=True)
            return None

    def _determine_task_type(self, file_basename: str) -> str:
        """
        Determine task type from filename with boundary-aware matching.

        IMPORTANT: fTurn is detected ONLY by explicit 'FTURN' or 'F_TURN' patterns.
        Turn_DT / Turn_ST are walking tasks (not fTurn) and are classified as DT/ST.
        """
        s = file_basename.upper()

        # 1. fTurn - ONLY explicit fTurn/F_TURN patterns
        if "FTURN" in s or "F_TURN" in s:
            return "fTurn"

        # 2. LShape
        if "LSHAPE" in s or "L_SHAPE" in s:
            return "LShape"

        # 3. Obstacle
        if "OBSTACLE" in s:
            return "Obstacle"

        # 4. Navigation
        if "NAVIGATION" in s or re.search(r'\bNAV\b', s):
            return "Navigation"

        # 5. DT/ST detection (catches Turn_DT, Walking_DT, Walking_DT-AC, etc.)
        #    These are all long_walk type tasks with event-based or fallback baseline
        if re.search(r'(^|[^A-Z])DT([^A-Z]|$)', s):
            return "DT"
        if re.search(r'(^|[^A-Z])ST([^A-Z]|$)', s):
            return "ST"

        # 6. Generic walk (no DT/ST suffix)
        if "WALK" in s:
            return "LongWalk"

        logger.warning(f" Unknown task type from filename: {file_basename}")
        return "Unknown"

    def _validate_task_requirements(self, task_type: str, task_config: dict, events: pd.DataFrame,
                                    filename: str) -> bool:
        """Validate that task requirements are met before processing."""
        task_category = task_config['type']
        min_events = task_config['min_events']

        if task_category == 'event_dependent':
            if events is None or len(events) < min_events:
                logger.error(
                    f" {task_type} task requires at least {min_events} event markers, but only found {len(events) if events is not None else 0} in {filename}")
                return False

            if task_type == "LShape":
                if len(events) < 3:
                    logger.error(
                        f" L-Shape task requires at least 3 event markers, but only found {len(events)} in {filename}")
                    return False
            elif 'S1' not in events['Event'].str.upper().values:
                logger.error(f" {task_type} task requires 'S1' baseline marker, but not found in {filename}")
                return False

            logger.info(f" {task_type} task validation passed: {len(events)} events found")

        elif task_category == 'long_walk':
            if events is None or events.empty:
                logger.warning(f" {task_type} task has no event markers, will use time-based fallback")
            else:
                logger.info(f" {task_type} task has {len(events)} event markers available")

        return True

    def _extract_and_clean_events(self, data_dict: dict, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and properly clean event data."""
        try:
            events = data_dict.get('events', None)

            if events is None or events.empty:
                if 'Event' in data.columns:
                    event_mask = data['Event'].notna() & (data['Event'] != '') & (data['Event'] != 'nan')
                    if event_mask.any():
                        events = data.loc[event_mask, ['Sample number', 'Event']].copy()
                    else:
                        events = pd.DataFrame(columns=['Sample number', 'Event'])
                else:
                    events = pd.DataFrame(columns=['Sample number', 'Event'])

            if events is not None and not events.empty:
                events = events.copy()

                if 'Sample number' not in events.columns or 'Event' not in events.columns:
                    return pd.DataFrame(columns=['Sample number', 'Event'])

                events['Event'] = events['Event'].astype(str).str.strip()
                invalid_events = ['', 'nan', 'None', 'NaN', 'null']
                events = events[~events['Event'].str.lower().isin([x.lower() for x in invalid_events])]
                events['Sample number'] = pd.to_numeric(events['Sample number'], errors='coerce')
                events = events.dropna(subset=['Sample number'])

                max_samples = len(data)
                events = events[(events['Sample number'] >= 0) & (events['Sample number'] <= max_samples)]
                events = events.drop_duplicates(subset=['Sample number'])
                events = events.sort_values('Sample number').reset_index(drop=True)

                return events
            else:
                return pd.DataFrame(columns=['Sample number', 'Event'])

        except Exception as e:
            logger.warning(f" Error cleaning events: {str(e)}")
            return pd.DataFrame(columns=['Sample number', 'Event'])

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

    def _drop_initial_seconds(self, data: pd.DataFrame, seconds: float) -> pd.DataFrame:
        """Drop the first `seconds` of a recording (device/initialization
        artifacts), matching FullCapProcessor's initial crop. Renumbers
        'Sample number' and 'Time (s)' (if present) afterward so downstream
        code sees a clean, zero-based timeline. No-op if seconds <= 0.
        """
        if seconds is None or seconds <= 0:
            return data

        n = int(seconds * self.fs)
        if n <= 0:
            return data
        if len(data) <= n:
            logger.warning(f"Recording has only {len(data)} samples (<= {n} to drop for a "
                           f"{seconds}s crop at {self.fs}Hz); skipping initial-seconds crop.")
            return data

        cropped = data.iloc[n:].reset_index(drop=True)
        if 'Sample number' in cropped.columns:
            cropped['Sample number'] = np.arange(len(cropped))
        if 'Time (s)' in cropped.columns:
            cropped['Time (s)'] = cropped['Sample number'] / self.fs

        logger.info(f"Dropped initial {seconds}s ({n} samples) from recording "
                    f"({len(data)} -> {len(cropped)} rows)")
        return cropped

    def _apply_scr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Short Channel Regression (long channels regressed against short channels)."""
        logger.warning("SCR CHECKPOINT: _apply_scr() was called")

        try:
            sig_cols = [c for c in data.columns if any(k in c for k in ("HbO", "O2Hb", "HHb", "HbR"))]
            if not sig_cols:
                return data

            prefixes = {str(c).split()[0] for c in sig_cols if str(c).startswith("CH")}
            short_ids = {f"CH{i}" for i in SHORT_CHANNEL_IDS}  # CH3, CH5 (0-based) == optodes 4, 6 (1-based)

            short_prefixes = prefixes & short_ids
            long_prefixes = prefixes - short_prefixes

            short_cols = [c for c in sig_cols if c.split()[0] in short_prefixes]
            long_cols = [c for c in sig_cols if c.split()[0] in long_prefixes]

            if not short_cols or not long_cols:
                if not short_cols:
                    logger.warning(f"SCR skipped: no short-channel columns ({sorted(short_ids)}) present.")
                else:
                    logger.warning("SCR skipped: no long-channel columns present.")
                return data

            scr_long = scr_regression(data[long_cols], data[short_cols])

            out = data.copy()
            for col in scr_long.columns:
                out[col] = scr_long[col]

            logger.info(f"SCR applied: {len(long_cols)} long cols, {len(short_cols)} short cols ({sorted(short_ids)})")
            return out

        except Exception as e:
            logger.warning(f"SCR failed: {str(e)}")
            return data

    def _apply_baseline_correction(self, data: pd.DataFrame, events: pd.DataFrame,
                                   task_type: str = None) -> pd.DataFrame:
        """Apply task-specific baseline correction."""
        try:
            if events is None or events.empty:
                logger.warning("No events available for baseline correction, using fallback")
                return self._fallback_baseline_correction(data, task_type)

            events = events.copy()
            events['Event'] = events['Event'].astype(str).str.strip().str.upper()

            if task_type == "LShape":
                return self._apply_lshape_baseline(data, events)

            return self._apply_standard_baseline(data, events, task_type)

        except Exception as e:
            logger.error(f"Baseline correction failed: {str(e)}")
            return None

    def _apply_lshape_baseline(self, data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Apply L-Shape specific baseline correction using 2nd to 3rd event markers."""
        try:
            events_sorted = events.sort_values('Sample number').reset_index(drop=True)

            if len(events_sorted) < 3:
                logger.error(f" L-Shape task requires at least 3 event markers, but only found {len(events_sorted)}")
                return None

            second_event = events_sorted.iloc[1]
            third_event = events_sorted.iloc[2]

            second_sample = second_event['Sample number']
            third_sample = third_event['Sample number']

            if third_sample <= second_sample:
                logger.error(f" L-Shape baseline error: events out of order")
                return None

            baseline_events = pd.DataFrame({
                'Sample number': [second_sample, third_sample],
                'Event': ['BaselineStart', 'BaselineEnd']
            })

            logger.info(f" L-Shape baseline correction: {second_event['Event']} to {third_event['Event']}")
            return baseline_subtraction(data, baseline_events, baseline_type="lshape_task")

        except Exception as e:
            logger.error(f"L-Shape baseline correction failed: {str(e)}")
            return None

    def _apply_standard_baseline(self, data: pd.DataFrame, events: pd.DataFrame, task_type: str = None) -> pd.DataFrame:
        """Apply standard baseline correction for non-L-Shape tasks."""
        try:
            s1_markers = events[events['Event'] == 'S1']

            if not s1_markers.empty:
                s1_sample = s1_markers.iloc[0]['Sample number']

                w1_markers = events[events['Event'] == 'W1']
                w1_after_s1 = w1_markers[w1_markers['Sample number'] > s1_sample]

                if not w1_after_s1.empty:
                    w1_sample = w1_after_s1.iloc[0]['Sample number']
                    baseline_events = pd.DataFrame({
                        'Sample number': [s1_sample, w1_sample],
                        'Event': ['BaselineStart', 'BaselineEnd']
                    })
                    return baseline_subtraction(data, baseline_events, baseline_type="long_walk")

                s2_markers = events[events['Event'] == 'S2']
                s2_after_s1 = s2_markers[s2_markers['Sample number'] > s1_sample]

                if not s2_after_s1.empty:
                    s2_sample = s2_after_s1.iloc[0]['Sample number']
                    baseline_events = pd.DataFrame({
                        'Sample number': [s1_sample, s2_sample],
                        'Event': ['BaselineStart', 'BaselineEnd']
                    })
                    return baseline_subtraction(data, baseline_events, baseline_type="event_based")

            task_config = self.task_types.get(task_type, {})
            if task_config.get('type') == 'long_walk':
                logger.warning("No event-based baseline found for long walk task, using time-based fallback")
                return self._fallback_baseline_correction(data, task_type)
            else:
                logger.error(" No valid baseline markers found for event-dependent task")
                return None

        except Exception as e:
            logger.error(f"Standard baseline correction failed: {str(e)}")
            return None

    def _fallback_baseline_correction(self, data: pd.DataFrame, task_type: str = None) -> pd.DataFrame:
        """
        Fallback baseline correction using task-specific timing calculated from
        end-of-recording.

        Uses _get_task_timing() to determine correct durations based on the
        actual filename (90s vs 150s tasks).
        """
        try:
            total = len(data)

            # Get task-specific timing from the current filename
            file_basename = getattr(self, '_current_file_basename', '')
            timing = self._get_task_timing(file_basename, task_type or 'Unknown')

            total_expected = timing['total_expected']
            baseline_duration = timing['baseline_duration']
            end_duration = timing['end_duration']

            logger.info(f"Fallback baseline using timing: {total_expected}s total "
                        f"({baseline_duration}s baseline, {end_duration}s end)")

            min_required = int(baseline_duration * self.fs)
            if total < min_required:
                logger.warning(f"Fallback baseline: record too short ({total} samples < {min_required} required); "
                               f"returning data without subtraction")
                return data

            # Calculate baseline window from end of recording
            s1 = max(0, min(total - 1, int(total - total_expected * self.fs)))
            s2 = max(0, min(total - 1, int(total - (total_expected - baseline_duration) * self.fs)))
            s3 = max(0, min(total - 1, int(total - end_duration * self.fs)))

            marks = sorted({s1, s2, s3})
            if len(marks) < 3 or marks[1] - marks[0] < int(2 * self.fs):
                base = max(0, total - int(total_expected * self.fs))
                end = max(base + int(2 * self.fs), min(total - 1, total - int(end_duration * self.fs)))
                marks = [base, base + int(2 * self.fs), end]

            fallback_events = pd.DataFrame({
                'Sample number': marks,
                'Event': ['S1', 'S2', 'S3']
            })
            logger.warning(f"Using fallback end-of-recording baseline markers: {marks} "
                           f"(timing: {total_expected}s total for '{file_basename}')")
            return baseline_subtraction(data, fallback_events)

        except Exception as e:
            logger.warning(f"Fallback baseline correction failed: {str(e)}")
            return data

    def _finalize_outputs(self, data: pd.DataFrame, output_dir: str,
                          file_basename: str, subject: str,
                          task_type: str, task_config: dict,
                          events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Final output step with Z-transformation applied BEFORE averaging.
        NOTE: Post-event trimming is already applied in _process_pipeline_stages.
        """
        try:
            # NOTE: Trimming already happened in _process_pipeline_stages - do NOT apply again

            logger.info("Creating non-Z-transformed averaged data")
            # Pass the montage explicitly rather than letting average_channels()
            # guess zero- vs one-based numbering from whether CH0 happens to be
            # present. If SQI filtering ever drops CH0 specifically, that guess
            # would silently flip to the wrong montage table (see
            # SHORT_CHANNEL_IDS comment at top of this module for the source of
            # truth); passing short/left/right explicitly avoids that entirely.
            averaged_raw = average_channels(
                data.copy(),
                short_ids=sorted(SHORT_CHANNEL_IDS),
                left_ids=[4, 6, 7],
                right_ids=[0, 1, 2],
            )

            for col in ['grand oxy', 'grand deoxy']:
                if col not in averaged_raw.columns:
                    logger.warning(f"Missing '{col}' after averaging; filling with NaNs.")
                    averaged_raw[col] = np.nan

            logger.info("Applying Z-transformation to INDIVIDUAL channels before averaging")

            signal_cols = [col for col in data.columns
                           if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb'])
                           and pd.api.types.is_numeric_dtype(data[col])
                           and 'grand' not in col.lower()]

            if signal_cols:
                logger.info(f"Z-transforming {len(signal_cols)} individual channels")
                z_transformed_data = z_transformation(data.copy(), signal_cols)
            else:
                logger.warning("No individual signal channels found for Z-transformation")
                z_transformed_data = data.copy()

            logger.info("Averaging Z-transformed channels")
            averaged_z = average_channels(
                z_transformed_data,
                short_ids=sorted(SHORT_CHANNEL_IDS),
                left_ids=[4, 6, 7],
                right_ids=[0, 1, 2],
            )

            for col in ['grand oxy', 'grand deoxy']:
                if col not in averaged_z.columns:
                    averaged_z[col] = np.nan

            for df_version in [averaged_raw, averaged_z]:
                if 'Sample number' in df_version.columns:
                    df_version['Time (s)'] = df_version['Sample number'] / self.fs
                else:
                    df_version['Time (s)'] = np.arange(len(df_version)) / self.fs

            task_category = task_config.get('type', 'unknown')
            # Use the original source filename (without extension) for traceability
            source_filename = os.path.splitext(file_basename)[0]

            for df_version in [averaged_raw, averaged_z]:
                df_version['Condition'] = source_filename
                df_version['Subject'] = subject
                df_version['TaskType'] = source_filename
                df_version['TaskCategory'] = task_category
                df_version['Quality_Filtering_Applied'] = bool(self.enable_quality_filtering)
                df_version['Quality_Metrics_Used'] = '+'.join(self.enabled_metrics) or 'none'
            metrics_tag = '_'.join(m.upper() for m in self.enabled_metrics) or 'NONE'
            quality_suffix = f"_with_{metrics_tag}_filtering" if self.enable_quality_filtering else "_without_quality_filtering"
            condition_dir = os.path.join(output_dir, f"{task_type}{quality_suffix}")
            os.makedirs(condition_dir, exist_ok=True)

            output_file_raw = os.path.join(condition_dir, f"{file_basename}_FULLY_PROCESSED_RAW{quality_suffix}.csv")
            averaged_raw.to_csv(output_file_raw, index=False)
            logger.info(f" Saved RAW (non-Z-scored) data to {output_file_raw}")
            logger.info(f"   Grand oxy mean: {averaged_raw['grand oxy'].mean():.6f}")
            logger.info(f"   Grand deoxy mean: {averaged_raw['grand deoxy'].mean():.6f}")

            output_file_z = os.path.join(condition_dir, f"{file_basename}_FULLY_PROCESSED_ZSCORE{quality_suffix}.csv")
            averaged_z.to_csv(output_file_z, index=False)
            logger.info(f" Saved Z-SCORED data to {output_file_z}")

            try:
                plot_condition = task_type

                self._create_final_plot(
                    averaged_raw, condition_dir, file_basename, f"{plot_condition}{quality_suffix}",
                    ['grand oxy', 'grand deoxy'],
                    f'final_overall_RAW{quality_suffix}',
                    f'Final Overall - Raw Concentrations{quality_suffix}', events
                )

                self._create_final_plot(
                    averaged_z, condition_dir, file_basename, f"{plot_condition}{quality_suffix}",
                    ['grand oxy', 'grand deoxy'],
                    f'final_overall_ZSCORE{quality_suffix}',
                    f'Final Overall - Z-scores{quality_suffix}', events
                )
            except Exception as e:
                logger.warning(f" Plotting failed for {file_basename}: {e}")

            return averaged_raw

        except Exception as e:
            logger.error(f" Final output generation failed: {str(e)}", exc_info=True)
            return None

    def _create_final_plot(self, data: pd.DataFrame, output_dir: str,
                           file_basename: str, condition: str,
                           columns: List[str], prefix: str,
                           title: str,
                           events: Optional[pd.DataFrame] = None) -> None:
        """Create standardized final plots with automatic scaling and (remapped) events."""
        try:
            if 'Time (s)' not in data.columns:
                data = data.copy()
                data['Time (s)'] = np.arange(len(data)) / self.fs

            plot_data = data[columns + ['Time (s)']].rename(columns={
                columns[0]: "grand oxy",
                columns[1]: "grand deoxy"
            })

            clean_events = None
            if events is not None and not events.empty:
                ev = events.copy()
                ev = ev.dropna(subset=['Sample number'])

                if hasattr(self, "_event_index_remap") and callable(self._event_index_remap):
                    ev['Remapped_Sample'] = ev['Sample number'].apply(self._event_index_remap)

                    walking_start_events = ['W1', 'WALK', 'START_WALK', 'WALKING', 'S2', 'START', 'TASK_START', 'GO',
                                            'S1']
                    ev['Event_Upper'] = ev['Event'].astype(str).str.upper()
                    start_events_upper = [e.upper() for e in walking_start_events]
                    is_start_event = ev['Event_Upper'].isin(start_events_upper)

                    max_data_samples = len(data) - 1

                    def get_final_sample(row):
                        if pd.notna(row['Remapped_Sample']):
                            return row['Remapped_Sample']
                        elif is_start_event[row.name]:
                            original_sample = row['Sample number']
                            if 0 <= original_sample <= max_data_samples:
                                return original_sample
                        return None

                    ev['Final_Sample'] = ev.apply(get_final_sample, axis=1)
                    ev = ev.dropna(subset=['Final_Sample'])
                    ev['Sample number'] = ev['Final_Sample'].astype(int)
                    ev = ev.drop(columns=['Remapped_Sample', 'Final_Sample', 'Event_Upper'])

                max_idx = len(data) - 1
                ev = ev[(ev['Sample number'] >= 0) & (ev['Sample number'] <= max_idx)]

                if not ev.empty:
                    clean_events = ev

            subject = self._extract_subject(file_basename)

            fig, ylim = plot_overall_signals(
                plot_data,
                fs=self.fs,
                title=f"{file_basename} - {title}",
                subject=subject,
                condition=condition,
                y_lim=None,
                events=clean_events
            )

            output_path = os.path.join(output_dir, f"{prefix}_{condition}.png")
            self._save_figure(fig, output_path)

            event_count = len(clean_events) if clean_events is not None else 0
            logger.info(f"Successfully created {title} plot with {event_count} event markers: {output_path}")

        except Exception as e:
            logger.error(f"Failed to create {title} plot for {file_basename}: {str(e)}", exc_info=True)
            raise

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
