import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple, Callable, List, Iterable
import logging
import re
# Import processing steps
from fnirs_PFC_2025.preprocessing.z_transformation import z_transformation
from fnirs_PFC_2025.preprocessing.butterworth_filter import butterworth_bandpass
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

# Montage (0-based channel numbering). Optode order on the cap is
# 1(R) 2(R) 3(R) 4(short) 5(L) 6(short) 7(L) 8(L) in 1-based numbering.
# Channel 6 falls inside the left block, so I'm treating it as the left
# short channel, which leaves channel 4 (right after the right block) as
# the right short channel. That's my best read of the layout, not
# something I've confirmed against the probe documentation. If SCR
# results look off, double check the cap layout first and swap these two
# constants if the assignment turns out to be backwards.
RIGHT_LONG_IDS = (0, 1, 2)
LEFT_LONG_IDS = (4, 6, 7)
RIGHT_SHORT_ID = 3
LEFT_SHORT_ID = 5
SHORT_CHANNEL_IDS = {RIGHT_SHORT_ID, LEFT_SHORT_ID}


class FileProcessor:

    def __init__(self, fs: float = 50.0,
                 sqi_threshold: float = DEFAULT_SQI_THRESHOLD,
                 sci_threshold: float = DEFAULT_SCI_THRESHOLD,
                 psp_threshold: float = DEFAULT_PSP_THRESHOLD,
                 enabled_metrics: Tuple[str, ...] = ("sci", "psp"),
                 enable_quality_filtering: bool = True,
                 exclude_failing_short_channels: bool = False,
                 post_walking_trim_seconds: float = 3.0,
                 initial_crop_seconds: float = 1.0,
                 skip_diagnostic_plots: bool = False,
                 compute_zscore: bool = True):
        """
        Set up a processor for one run of the pipeline.

        fs is the sampling rate in Hz. sqi_threshold, sci_threshold and
        psp_threshold are the pass/fail cutoffs for the three quality
        metrics (SQI is on a 1-5 scale, default 2.0; SCI and PSP default to
        the Pollonini/PHOEBE values of 0.75 and 0.10). Only the metrics
        named in enabled_metrics actually get computed - anything not in
        that tuple is skipped entirely, not just ignored. Default is
        ("sci", "psp"); SQI has to be turned on explicitly.

        enable_quality_filtering controls whether a channel that fails gets
        dropped from SCR/bandpass/baseline/output, or just flagged in the
        report while staying in the data (default is to drop it).
        exclude_failing_short_channels is a separate switch for CH3/CH5
        specifically - by default they're kept even if they fail, since
        they're only used as SCR regressors and never end up in the signal.

        post_walking_trim_seconds is how much to cut after the walking-start
        marker (3.0s default). initial_crop_seconds trims the very start of
        every recording for device warm-up artifacts, before anything else
        touches the data - events, quality metrics, TDDR, all of it see the
        already-cropped recording. Defaults to 1.0s to match
        FullCapProcessor.

        skip_diagnostic_plots turns off the five per-stage plots (Post-MBLL,
        Post-TDDR, Post-SCR, Post-Filter, Post-Baseline) plus the combined
        summary panel - these are debugging aids, so the raw-concentration
        plot and the final RAW/ZSCORE plots still get made regardless.
        Defaults to False, i.e. plots are on.

        compute_zscore, if set to False, skips Z-transformation altogether:
        no Z-scored averaging, no ZSCORE csv, no Z-score plot, just the RAW
        output. Defaults to True so both get produced, same as before.
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
        self.skip_diagnostic_plots = skip_diagnostic_plots
        self.compute_zscore = compute_zscore

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
            if any(kw in col for kw in keywords) and 'grand' not in col.lower():
                if not self._is_short_channel(col):
                    long_cols.append(col)
        return long_cols

    def process_file(self,
                     file_path: str,
                     output_base_dir: str,
                     input_base_dir: str,
                     read_file_func: Callable = None,
                     baseline_duration: Optional[float] = None,
                     ) -> Optional[Dict]:
        """
        Run one fNIRS file all the way through the pipeline - quality
        filtering and post-walking trimming included.

        On success, returns a dict with success=True, data (the processed
        DataFrame), subject, task_type, and whatever quality report got
        generated along the way. On failure, success is False and error
        holds a message explaining why.
        """
        try:
            self._event_index_remap = None
            self._last_quality_report = None
            output_dir = self._create_output_dir(output_base_dir, input_base_dir, file_path)
            file_basename = os.path.basename(file_path)
            subject = self._extract_subject(file_path)

            self._current_file_basename = file_basename

            logger.info(f"Starting: {file_path}")

            # 1) Load raw data
            data_dict = read_file_func(file_path)
            if not data_dict or 'data' not in data_dict:
                logger.error(f"read_file_func failed or returned no 'data' for: {file_path}")
                return {'success': False, 'error': 'Failed to read data from file'}

            raw_df = data_dict['data']
            if raw_df is None or not isinstance(raw_df, pd.DataFrame):
                logger.error(f"Invalid DataFrame returned for: {file_path}")
                return {'success': False, 'error': 'Invalid DataFrame returned'}

            data = self._prepare_data(raw_df)
            if data is None or data.empty:
                logger.error(f"Prepared data is empty for: {file_path}")
                return {'success': False, 'error': 'Prepared data is empty'}

            # Drop the first `initial_crop_seconds` of the recording (device/
            # initialization artifacts), matching FullCapProcessor. Everything
            # downstream - event extraction, quality metrics, TDDR, etc. - sees
            # only the already-cropped recording.
            data = self._drop_initial_seconds(data, self.initial_crop_seconds)
            if data is None or data.empty:
                logger.error(f"Data empty after initial-seconds crop for: {file_path}")
                return {'success': False, 'error': 'Data empty after initial-seconds crop'}

            # 2) Metadata
            metadata = data_dict.get('metadata', {})
            subject_id = metadata.get('Subject Public ID')
            record_date = metadata.get('Record Date/Time')
            sample_rate = int(self.fs)

            # 3) Determine task type
            task_type = self._determine_task_type(file_basename)

            task_config = self.task_types.get(task_type, {'type': 'unknown', 'min_events': 2})

            timing = self._get_task_timing(file_basename, task_type)

            # 4) Extract and clean events
            events = self._extract_and_clean_events(data_dict, data)

            # 5) Validate task requirements
            if not self._validate_task_requirements(task_type, task_config, events, file_basename):
                logger.error(f"Task requirements not met for {file_basename}")
                return {'success': False, 'error': 'Task validation failed', 'validation_failed': True}

            # 6) Processing pipeline with SQI filtering.
            # "Raw" concentration plotting happens INSIDE _process_pipeline_stages,
            # after Beer-Lambert conversion (pre-TDDR) but before TDDR/SCR/filtering.
            processed_data = self._process_pipeline_stages(
                data=data,
                output_dir=output_dir,
                file_basename=file_basename,
                events=events,
                task_type=task_type,
                subject=subject,
            )
            if processed_data is None or processed_data.empty:
                logger.error(f"Pipeline stages returned empty for: {file_path}")
                return {'success': False, 'error': 'Pipeline processing returned empty data'}

            # 7) Final outputs with post-walking trimming
            final_df = self._finalize_outputs(
                processed_data, output_dir, file_basename, subject,
                task_type, task_config, events
            )

            if final_df is None or final_df.empty:
                logger.error(f"_finalize_outputs failed for: {file_path}")
                return {'success': False, 'error': 'Finalize outputs failed'}

            logger.info(f"Finished processing: {file_path}")
            return {
                'success': True,
                'data': final_df,
                'subject': subject,
                'task_type': task_type,
                'output_dir': output_dir,
                'quality': getattr(self, '_last_quality_report', None),
            }

        except Exception as e:
            logger.error(f"Exception in process_file for {file_path}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _get_task_timing(self, file_basename: str, task_type: str) -> dict:
        """
        Work out expected baseline/task/end durations from the filename
        itself, not just the task_type label - we need this distinction
        because Turn_DT, Turn_ST, Walking_DT-AC, Walking_DT-TMB and
        Walking_DT_DM are all 90s tasks, while plain Walking_DT, Walking_ST,
        DT and ST run 150s. Returns a dict with baseline_duration,
        task_duration, end_duration and total_expected.
        """
        s = file_basename.upper()

        # 90s tasks: 20s baseline + 60s task + 10s end.
        # Turn_DT / Turn_ST are walking-turn tasks, not fTurn - keep them separate.
        if re.search(r'TURN[_-]?(DT|ST)', s) and 'FTURN' not in s and 'F_TURN' not in s:
            return {"baseline_duration": 20.0, "task_duration": 60.0, "end_duration": 10.0, "total_expected": 90.0}

        # Walking_DT variants with a suffix (DT-AC, DT-TMB, DT_DM, ST-AC...).
        # I switched to a negative lookahead here instead of \b because \w
        # counts underscore as a word char, so \b was silently failing to
        # match whenever the suffix was itself followed by another
        # underscore - which is basically always, since these filenames
        # keep going with "_OD" or similar right after. That meant this
        # branch just never fired on real data before this fix.
        if re.search(r'(DT|ST)[_-](AC|TMB|DM)(?![A-Z])', s):
            return {"baseline_duration": 20.0, "task_duration": 60.0, "end_duration": 10.0, "total_expected": 90.0}

        # 150s tasks: 20s baseline + 120s task + 10s end.
        if task_type in ('DT', 'ST', 'LongWalk'):
            return {"baseline_duration": 20.0, "task_duration": 120.0, "end_duration": 10.0, "total_expected": 150.0}

        # Event-dependent tasks don't actually use this for baseline timing
        # (they require real events instead), but return something sane anyway.
        return {"baseline_duration": 20.0, "task_duration": 120.0, "end_duration": 10.0, "total_expected": 150.0}

    def _process_pipeline_stages(self, data: pd.DataFrame,
                                 output_dir: str, file_basename: str,
                                 events: Optional[pd.DataFrame] = None,
                                 task_type: str = None,
                                 subject: str = None) -> Optional[pd.DataFrame]:
        """Run one recording through every stage of the pipeline, handling the
        column renaming the loader does along the way.

        Order matches FullCapProcessor: SQI (pre-TDDR), then TDDR, then
        OD-to-concentration (post-TDDR), then SCR, bandpass, baseline, trim.
        """
        # The loader renames columns to CH{i}_WL{wavelength}, e.g. CH0_WL846, CH0_WL757.
        od_cols = [col for col in data.columns
                   if re.match(r'CH\d+_WL\d+', col)
                   and pd.api.types.is_numeric_dtype(data[col])]

        if not od_cols:
            logger.error("No OD columns found with pattern CH*_WL*")
            od_cols = [col for col in data.columns
                       if 'WL' in col and pd.api.types.is_numeric_dtype(data[col])]

        if not od_cols:
            logger.error("No OD columns found after fallback")
            return None

        # Group by CH number rather than assuming wavelengths sit in adjacent columns.
        channel_groups = {}
        for col in od_cols:
            match = re.match(r'CH(\d+)_WL(\d+)', col)
            if match:
                ch_num = match.group(1)
                wavelength = match.group(2)
                channel_id = f"CH{ch_num}"
                channel_groups.setdefault(channel_id, {})[wavelength] = col

        valid_channels = {}
        for ch_id, wavelengths in channel_groups.items():
            if len(wavelengths) == 2:
                valid_channels[ch_id] = wavelengths
            else:
                logger.warning(f"Channel {ch_id} has {len(wavelengths)} wavelengths, expected 2")

        if not valid_channels:
            logger.error("No valid channels with 2 wavelengths found")
            return None

        # SQI runs on the pre-TDDR signal on purpose. TDDR is a motion
        # correction step and SQI is meant to catch exactly the kind of
        # artifacts TDDR removes, so scoring it after TDDR would just
        # inflate the numbers. Same reasoning for the "raw" concentration
        # plot - it's supposed to show the minimally processed signal.
        # After this, TDDR gets applied to the OD, concentration gets
        # re-derived from the corrected OD, and everything downstream (SCR,
        # bandpass, baseline, final output) works off that post-TDDR version.

        # 1) OD -> concentration, pre-TDDR, used for SQI and the raw plot.
        # events/task_type get passed through here so baseline referencing
        # can use the real S1->W1 (or S1->S2, or the L-Shape 2nd->3rd
        # marker) window rather than just defaulting to "first N seconds".
        concentration_data_pre_tddr = self._convert_od_to_concentration(
            data, od_cols, valid_channels, events=events, task_type=task_type,
        )

        if concentration_data_pre_tddr is None or concentration_data_pre_tddr.empty:
            logger.error("Failed to convert OD to concentration")
            return None

        diagnostic_stages = {}
        diagnostic_stages["1_Post-MBLL"] = concentration_data_pre_tddr.copy()

        # 1.5) Plot rawconcentration data: post-Beer-Lambert, before
        # TDDR/SCR/filtering touch it - the least-processed view we have.
        self._plot_raw_concentration_data(
            data=data,
            concentration_data=concentration_data_pre_tddr,
            output_dir=output_dir,
            file_basename=file_basename,
            subject=subject,
            condition=task_type,
            events=events
        )

        # 2) Quality scoring (whichever of SQI/SCI/PSP are enabled), on the
        #    pre-TDDR OD and pre-TDDR concentration.
        excluded_channels, quality_report = self._calculate_quality_and_filter(
            data, valid_channels, output_dir, file_basename, concentration_data_pre_tddr
        )
        self._last_quality_report = quality_report

        # 3) TDDR on the OD signals, then re-derive concentration from the
        #    corrected OD using the same events/task_type as above.
        data_tddr = self._apply_tddr(data)

        concentration_data = self._convert_od_to_concentration(
            data_tddr, od_cols, valid_channels, events=events, task_type=task_type,
        )

        if concentration_data is None or concentration_data.empty:
            logger.error("Failed to convert TDDR-corrected OD to concentration")
            return None

        diagnostic_stages["2_Post-TDDR"] = concentration_data.copy()

        # 4) Start a fresh working frame with just metadata + concentration.
        metadata_cols = []
        if 'Sample number' in data.columns:
            metadata_cols.append('Sample number')
        if 'Time (s)' in data.columns:
            metadata_cols.append('Time (s)')
        if 'Event' in data.columns:
            metadata_cols.append('Event')

        working_data = data[metadata_cols].copy()
        for col in concentration_data.columns:
            working_data[col] = concentration_data[col]

        # 5) Drop excluded channels if filtering is on. The exclusion list
        #    was computed pre-TDDR from whichever metrics are enabled.
        if self.enable_quality_filtering and excluded_channels:
            concentration_cols_to_exclude = []
            for excluded_od_col in excluded_channels:
                for conc_col in working_data.columns:
                    channel_match = re.search(r'CH\d+', excluded_od_col)
                    if channel_match and channel_match.group() in conc_col:
                        concentration_cols_to_exclude.append(conc_col)

            if concentration_cols_to_exclude:
                concentration_cols_to_exclude = list(set(concentration_cols_to_exclude))
                working_data = working_data.drop(columns=concentration_cols_to_exclude)

        signal_cols = [col for col in working_data.columns
                       if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb'])
                       and pd.api.types.is_numeric_dtype(working_data[col])]

        if not signal_cols:
            logger.error("No concentration signal columns found after quality filtering")
            return None

        # Quick sanity check on magnitude - if OD ended up here instead of
        # concentration, values would come out two or three orders too big.
        sample_mean = working_data[signal_cols].iloc[:10].mean().mean()
        if abs(sample_mean) > 50:
            logger.error(
                f"Concentration values are too large ({sample_mean:.6f} uM, expected -10 to +10 "
                f"typically), which suggests OD data is being used instead of concentration."
            )

        signal_slice = working_data[signal_cols].copy()

        # 6) SCR on the post-TDDR concentration data.
        scr_data = self._apply_scr(signal_slice, quality_report)
        diagnostic_stages["3_Post-SCR"] = scr_data.copy()

        # 7) Bandpass filter the concentration data.
        filtered_data = self._apply_bandpass_filter(scr_data)
        diagnostic_stages["4_Post-Filter"] = filtered_data.copy()

        for col in filtered_data.columns:
            working_data[col] = filtered_data[col]

        # 8) Baseline correction.
        baseline_corrected = self._apply_baseline_correction(working_data, events, task_type)

        if baseline_corrected is not None:
            bc_signal_cols = [col for col in baseline_corrected.columns
                              if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb'])
                              and pd.api.types.is_numeric_dtype(baseline_corrected[col])]
            if bc_signal_cols:
                diagnostic_stages["5_Post-Baseline"] = baseline_corrected[bc_signal_cols].copy()

            if not self.skip_diagnostic_plots:
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

                self._create_diagnostic_summary_plot(
                    stages_data=diagnostic_stages,
                    output_dir=output_dir,
                    file_basename=file_basename,
                    subject=subject,
                    condition=task_type,
                    events=events
                )

            # 9) Post-event trimming.
            trimmed_data = self._apply_post_event_trimming(baseline_corrected, events, task_type)

            final_signal_cols = [col for col in trimmed_data.columns
                                 if any(kw in col for kw in ['HbO', 'HbR'])
                                 and pd.api.types.is_numeric_dtype(trimmed_data[col])]

            if final_signal_cols:
                final_mean = trimmed_data[final_signal_cols].mean().mean()
                if abs(final_mean) > 50:
                    logger.error(f"Final values still too large: {final_mean:.6f} uM")

            return trimmed_data

        return None

    def _plot_raw_concentration_data(self, data: pd.DataFrame,
                                     concentration_data: pd.DataFrame,
                                     output_dir: str,
                                     file_basename: str,
                                     subject: str,
                                     condition: str,
                                     events: Optional[pd.DataFrame] = None) -> None:
        """
        Plot the "raw" concentration data - right after the Beer-Lambert
        conversion, before TDDR/SCR/filtering touch anything.

        Short channels left out of the averaging so
        matches how the final output is built. Y-limits are derived from
        each plot's own data rather than kept consistent across recordings.
        """
        o2hb_cols = self._get_long_channel_cols(concentration_data.columns, 'oxy')
        hhb_cols = self._get_long_channel_cols(concentration_data.columns, 'deoxy')
        combined_cols = o2hb_cols + hhb_cols

        if not combined_cols:
            logger.warning("No long-channel concentration columns found for raw plotting")
            return

        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        plot_df = concentration_data[combined_cols].copy()

        if 'Time (s)' in data.columns:
            plot_df['Time (s)'] = data['Time (s)'].values
        else:
            plot_df['Time (s)'] = np.arange(len(plot_df)) / self.fs

        try:
            fig, axes, ylim = plot_channels_separately(
                plot_df[combined_cols],
                fs=self.fs,
                title=f"{file_basename} - Raw Concentration (Post-MBLL, Pre-Processing)",
                subject=subject,
                condition=condition,
                y_lim=None
            )
            self._save_figure(fig,
                              os.path.join(condition_dir, f"raw_concentration_individual_channels_{condition}.png"))
        except Exception as e:
            logger.warning(f"Failed to create individual channels plot: {e}")

        if o2hb_cols and hhb_cols:
            try:
                avg_o2hb = plot_df[o2hb_cols].mean(axis=1)
                avg_hhb = plot_df[hhb_cols].mean(axis=1)

                overall_df = pd.DataFrame({
                    "Time (s)": np.arange(len(plot_df)) / self.fs,
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
                    y_lim=None,
                    events=clean_events
                )
                self._save_figure(fig, os.path.join(condition_dir, f"raw_concentration_overall_{condition}.png"))

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
        One diagnostic plot for a single stage of the pipeline (e.g.
        "Post-SCR"). Short channels stay out of the averaging here too, for
        consistency with the final output.
        """
        o2hb_cols = self._get_long_channel_cols(data.columns, 'oxy')
        hhb_cols = self._get_long_channel_cols(data.columns, 'deoxy')

        if not o2hb_cols or not hhb_cols:
            logger.warning(f"No long-channel concentration columns found for diagnostic plot at stage: {stage_name}")
            return

        diag_dir = os.path.join(output_dir, condition, "diagnostic_stages")
        os.makedirs(diag_dir, exist_ok=True)

        try:
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
                title=f"{file_basename} - Stage {stage_number}: {stage_name}",
                subject=subject,
                condition=condition,
                y_lim=None,
                events=clean_events
            )

            output_path = os.path.join(diag_dir,
                                       f"stage_{stage_number}_{stage_name.replace(' ', '_').replace('-', '_')}_{condition}.png")
            self._save_figure(fig, output_path)

        except Exception as e:
            logger.warning(f"Failed to create diagnostic plot for stage {stage_name}: {e}")

    def _create_diagnostic_summary_plot(self, stages_data: Dict[str, pd.DataFrame],
                                        output_dir: str,
                                        file_basename: str,
                                        subject: str,
                                        condition: str,
                                        events: Optional[pd.DataFrame] = None) -> None:
        """
        One multi-panel figure with every processing stage stacked so they
        can be compared side by side. Short channels are excluded here too.
        """
        n_stages = len(stages_data)
        if n_stages == 0:
            return

        diag_dir = os.path.join(output_dir, condition, "diagnostic_stages")
        os.makedirs(diag_dir, exist_ok=True)

        fig, axes = plt.subplots(n_stages, 1, figsize=(14, 4 * n_stages), sharex=True)
        if n_stages == 1:
            axes = [axes]

        fig.suptitle(
            f"{file_basename} - Processing Pipeline Stages\nSubject: {subject}\n(Long channels only, excluding CH3 & CH5)",
            fontsize=12)

        clean_events = None
        if events is not None and not events.empty:
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

            o2hb_cols = self._get_long_channel_cols(data.columns, 'oxy')
            hhb_cols = self._get_long_channel_cols(data.columns, 'deoxy')

            if not o2hb_cols or not hhb_cols:
                ax.text(0.5, 0.5, f"No long-channel data for {stage_name}", ha='center', va='center')
                ax.set_title(stage_name)
                continue

            avg_o2hb = data[o2hb_cols].mean(axis=1)
            avg_hhb = data[hhb_cols].mean(axis=1)
            time = np.arange(len(data)) / self.fs

            ax.plot(time, avg_o2hb, 'r-', label='HbO', linewidth=1.2)
            ax.plot(time, avg_hhb, 'b-', label='HbR', linewidth=1.2)

            if clean_events is not None:
                ylim = ax.get_ylim()
                for _, row in clean_events.iterrows():
                    event_time = float(row['Sample number']) / self.fs
                    if 0 <= event_time <= time[-1]:
                        ax.axvline(x=event_time, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                        ax.text(event_time, ylim[1] * 0.95, str(row['Event']),
                                rotation=90, va='top', ha='right', fontsize=7, alpha=0.8)

            ax.set_title(
                f"{stage_name} (HbO: {avg_o2hb.mean():.2f}+/-{avg_o2hb.std():.2f}, HbR: {avg_hhb.mean():.2f}+/-{avg_hhb.std():.2f} uM)")
            ax.set_ylabel("Delta[Hb] (uM)")
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(diag_dir, f"SUMMARY_all_stages_{condition}.png")
        self._save_figure(fig, output_path)

    def _apply_bandpass_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the zero-phase Butterworth bandpass to concentration data."""
        try:
            return butterworth_bandpass(data, order=4, Wn=[0.01, 0.1], fs=int(self.fs))
        except Exception as e:
            logger.warning(f"Bandpass filtering failed: {str(e)}")
            return data

    def _apply_post_event_trimming(self, data: pd.DataFrame, events: pd.DataFrame,
                                   task_type: str) -> pd.DataFrame:
        """
        Trim the data down to the mobility-task window: cut a bit after the
        walking/task-start event, and for long_walk tasks also cut the
        trailing rest period at the end, so what's left is just the walking
        portion between the two rest blocks.
        """
        self._event_index_remap = None  # reset for each new file

        if self.post_walking_trim_seconds <= 0:
            return data

        if events is None or events.empty:
            logger.warning("No events available for post-event trimming; using synthetic "
                           "fallback boundaries (counting backward from the end of the recording)")
            return self._fallback_post_event_trimming(data, task_type)

        # When W1 (or something like it) is missing but have real S1
        # and S2 markers, figure out what S2 actually means from the S1->S2
        # gap before falling back to _find_walking_start_event's simpler
        # "just take the second S-marker" logic - that fallback has no way
        # to tell a task-end S2 from a walking-start one.
        forced_task_end_sample = None
        resolved = self._resolve_long_walk_boundaries(events, task_type, len(data))
        if resolved is not None:
            _, walking_start_sample, forced_task_end_sample, description = resolved
        else:
            walking_start_sample = self._find_walking_start_event(events, task_type)
        if walking_start_sample is None:
            logger.warning(f"No walking start event found for {task_type}; using synthetic "
                           f"fallback boundaries (counting backward from the end of the recording)")
            return self._fallback_post_event_trimming(data, task_type)

        trim_samples = int(self.post_walking_trim_seconds * self.fs)
        trim_start_sample = walking_start_sample + trim_samples

        if trim_start_sample >= len(data):
            logger.warning(f"Walking-start detection returned sample {walking_start_sample}, which "
                           f"leaves nothing to keep after the {self.post_walking_trim_seconds}s trim "
                           f"(recording is only {len(data)} samples). This usually means "
                           f"_find_walking_start_event's own internal fallback picked an event that "
                           f"isn't a real walking-start marker. Using synthetic fallback boundaries instead.")
            return self._fallback_post_event_trimming(data, task_type)

        critical_event_names = self.task_walking_events.get(task_type, [])
        critical_event_names_upper = [e.upper() for e in critical_event_names]

        events_clean = events.copy()
        events_clean['Event_Upper'] = events_clean['Event'].astype(str).str.upper()

        events_in_trim_region = events_clean[
            (events_clean['Sample number'] >= walking_start_sample) &
            (events_clean['Sample number'] < trim_start_sample) &
            (events_clean['Event_Upper'].isin(critical_event_names_upper))
            ]

        if not events_in_trim_region.empty:
            logger.warning(f"Found {len(events_in_trim_region)} critical events in trim region; "
                           f"adjusting trim to preserve them")

            last_critical_sample = events_in_trim_region['Sample number'].max()
            trim_start_sample = int(last_critical_sample) + 1

        # This next bit handles the trailing rest period at the end of the
        # recording (the "end_duration" from _get_task_timing - typically
        # 10s of standing still once the mobility task is done). It only
        # gets trimmed for long_walk tasks (DT/ST/LongWalk, plus the 90s
        # Turn_DT/Turn_ST variants) because those have a fixed task_duration
        # by protocol design - I've checked this against real event data and
        # the W1->S2 gap lines up with task_duration almost exactly.
        # Event-dependent tasks (fTurn/LShape/Obstacle/Navigation) end
        # whenever the participant finishes, not on a schedule, so their
        # durations from _get_task_timing aren't reliable enough to trim
        # against - those recordings only get trimmed at the start, as before.
        task_config = self.task_types.get(task_type, {})
        task_end_sample = len(data)
        if forced_task_end_sample is not None:
            # _resolve_long_walk_boundaries already found a real S2 marker
            # for the task-end position, so use it directly - an actual
            # recorded marker beats a duration-based guess.
            if forced_task_end_sample > trim_start_sample:
                task_end_sample = min(forced_task_end_sample, len(data))
            else:
                logger.warning("Resolved task-end sample is not after trim_start_sample; "
                               "keeping data through the end of the recording instead.")
        elif task_config.get('type') == 'long_walk':
            timing = self._get_task_timing(getattr(self, "_current_file_basename", ""), task_type)
            task_duration_samples = int(timing['task_duration'] * self.fs)
            candidate_end = walking_start_sample + task_duration_samples
            if candidate_end > trim_start_sample:
                task_end_sample = min(candidate_end, len(data))
            else:
                logger.warning("Computed task-end sample is not after trim_start_sample; "
                               "keeping data through the end of the recording instead.")

        # The pre-walking baseline already did its job - _apply_baseline_correction
        # used it as the reference window before this method ever runs, so it
        # doesn't belong in the final dataset. Leaving it in would mix baseline
        # samples into the task period, and anything downstream computing stats
        # over the result (stats_collector's "Overall Mean", for instance) would
        # end up diluting the actual walking response. Same reasoning applies to
        # the trailing rest period wherever we can pin it down.
        trimmed_data = data.iloc[trim_start_sample:task_end_sample].copy()

        def _remap(old_idx: int) -> Optional[int]:
            if old_idx < trim_start_sample or old_idx >= task_end_sample:
                return None
            return old_idx - trim_start_sample

        self._event_index_remap = _remap

        trimmed_data['Sample number'] = np.arange(len(trimmed_data))
        if 'Time (s)' in trimmed_data.columns:
            trimmed_data['Time (s)'] = trimmed_data['Sample number'] / self.fs

        return trimmed_data

    def _fallback_post_event_trimming(self, data: pd.DataFrame, task_type: str) -> pd.DataFrame:
        """Falls back to a synthetic start/end trim when there's no usable
        walking-start event - either no events at all, or none that match a
        known walking-start pattern. Same approach as
        _fallback_baseline_correction: count backward from the recording's
        actual length instead of assuming a fixed absolute start time,
        since the length is the one thing we can always trust.

        This only ever gets called for long_walk tasks. Event-dependent
        tasks need a minimum number of events per _validate_task_requirements
        and get rejected earlier if they don't have it, so we never end up
        guessing a walking-start position for those.
        """
        total = len(data)
        file_basename = getattr(self, "_current_file_basename", "")
        timing = self._get_task_timing(file_basename, task_type or "Unknown")
        total_expected = timing["total_expected"]
        baseline_duration = timing["baseline_duration"]
        end_duration = timing["end_duration"]

        min_required = int(total_expected * self.fs)
        if total < min_required:
            logger.warning(f"Fallback trim: record too short ({total} samples < {min_required} "
                           f"expected); skipping trim entirely (keeping the full recording).")
            return data

        # Synthetic walking-start = (total_expected - baseline_duration)
        # seconds before the end of the recording, same as
        # _fallback_baseline_correction's s2.
        walking_start_sample = int(total - (total_expected - baseline_duration) * self.fs)
        trim_samples = int(self.post_walking_trim_seconds * self.fs)
        trim_start_sample = max(0, walking_start_sample + trim_samples)

        # Synthetic task-end = end_duration seconds before the end of the
        # recording (mirrors _fallback_baseline_correction's s3) - this is
        # what keeps the trailing standing-still period out even when there's
        # no real event marking where it begins.
        task_config = self.task_types.get(task_type, {})
        task_end_sample = total
        if task_config.get("type") == "long_walk":
            task_end_sample = max(trim_start_sample, int(total - end_duration * self.fs))

        if trim_start_sample >= task_end_sample:
            logger.warning("Fallback trim: computed boundaries left nothing to keep; "
                           "skipping trim entirely (keeping the full recording).")
            return data

        trimmed_data = data.iloc[trim_start_sample:task_end_sample].copy()

        def _remap(old_idx: int) -> Optional[int]:
            if old_idx < trim_start_sample or old_idx >= task_end_sample:
                return None
            return old_idx - trim_start_sample

        self._event_index_remap = _remap

        trimmed_data["Sample number"] = np.arange(len(trimmed_data))
        if "Time (s)" in trimmed_data.columns:
            trimmed_data["Time (s)"] = trimmed_data["Sample number"] / self.fs

        removed = total - len(trimmed_data)
        logger.warning(
            f"Used synthetic fallback trim (backward from end of recording, "
            f"total_expected={total_expected}s): kept samples {trim_start_sample}-{task_end_sample}, "
            f"removed {removed} samples (~{removed / self.fs:.1f}s) including baseline"
            f"{' and the trailing end-of-recording rest' if task_end_sample < total else ''}."
        )
        return trimmed_data

    def _apply_z_transformation(self, data: pd.DataFrame, signal_cols: List[str]) -> pd.DataFrame:
        """Apply Z-transformation using the dedicated module."""
        return z_transformation(data, signal_cols)

    def _resolve_long_walk_boundaries(
        self, events: pd.DataFrame, task_type: str, n_samples: int,
    ) -> Optional[Tuple[int, int, Optional[int], str]]:
        """Work out baseline-start, walking-start, and (where possible)
        task-end for a long_walk task, but treat every marker's label as a
        hint rather than gospel - it gets checked against the expected
        timing before we trust it. This still runs even when there's a
        marker explicitly labeled as walking-start (W1/WALK/etc), because a
        marker that's present but simply mislabeled - say, sitting where
        baseline-start should really be - would otherwise sail through with
        zero verification.

        The rule of thumb: a real recorded marker wins over a computed
        estimate as long as its position is plausible, meaning within about
        50% of the expected protocol duration. So markers keep their actual
        recorded positions in the common case, and only get overridden by an
        inferred value when they clearly don't fit the rest of the file's
        timing.

        The two things we can lean on: S1 when it's there (the most direct
        anchor for baseline-start), and the recording's own length minus
        end_duration (an anchor for task-end, since the file's length is
        always known - same assumption _fallback_post_event_trimming makes).
        From those, walking-start is expected around S1 + baseline_duration
        (or, without S1, worked backward from a task-end anchor instead),
        and if a real W1-type or S2 marker sits near that expected spot, we
        use it directly rather than the pure calculation. Task-end works the
        same way - a real S2 only counts if it lines up with
        walking_start + task_duration. baseline_start is just S1 when we
        have it, since nothing more reliable exists to check it against and
        it's already the anchor for everything else here; without S1, it's
        computed as walking_start - baseline_duration so this stays
        consistent with what _apply_post_event_trimming would compute too.

        Returns (baseline_start_sample, walking_start_sample,
        task_end_sample_or_None, description). Returns None if there's
        nothing to go on at all - no S1, no W1-type marker, no S2 - in
        which case the caller falls back to its own synthetic approach.
        """
        task_config = self.task_types.get(task_type, {})
        if task_config.get('type') != 'long_walk':
            return None

        events_clean = events.copy()
        events_clean['Event_Upper'] = events_clean['Event'].astype(str).str.strip().str.upper()
        events_clean['Sample number'] = pd.to_numeric(events_clean['Sample number'], errors='coerce')
        events_clean = events_clean.dropna(subset=['Sample number']).sort_values('Sample number')

        def _first(label: str) -> Optional[int]:
            m = events_clean[events_clean['Event_Upper'] == label]
            return int(m.iloc[0]['Sample number']) if not m.empty else None

        s1_sample = _first('S1')
        s2_sample = _first('S2')
        walking_labels = {e.upper() for e in self.task_walking_events.get(task_type, [])}
        w_matches = events_clean[events_clean['Event_Upper'].isin(walking_labels)]
        w1_sample = int(w_matches.iloc[0]['Sample number']) if not w_matches.empty else None

        if s1_sample is None and w1_sample is None and s2_sample is None:
            return None  # nothing to reason from at all

        file_basename = getattr(self, "_current_file_basename", "")
        timing = self._get_task_timing(file_basename, task_type)
        baseline_duration = timing['baseline_duration']
        task_duration = timing['task_duration']
        end_duration = timing['end_duration']
        anchor_task_end = n_samples - int(end_duration * self.fs)

        def _close(a: Optional[int], b: Optional[int], ref_duration: float) -> bool:
            """True if a and b land within 50% of ref_duration seconds of
            each other - close enough to be normal timing jitter rather than
            a mislabeled or unrelated marker."""
            if a is None or b is None:
                return False
            return abs(a - b) / self.fs <= 0.5 * max(ref_duration, 1.0)

        # Start with a pure timing estimate, before tying it to any specific marker.
        if s1_sample is not None:
            inferred_walking_start = s1_sample + int(baseline_duration * self.fs)
        elif s2_sample is not None:
            inferred_walking_start = s2_sample - int(task_duration * self.fs)
        elif w1_sample is not None:
            inferred_walking_start = w1_sample
        else:
            inferred_walking_start = anchor_task_end - int(task_duration * self.fs)

        # If a real marker actually fits, use it instead of the pure estimate -
        # a recorded timestamp beats an assumption whenever the two roughly agree.
        walking_start_sample = inferred_walking_start
        walking_start_source = f"inferred position {inferred_walking_start} (no marker close enough to trust)"
        for label, sample in (("W1/WALK-type marker", w1_sample), ("S2", s2_sample)):
            if sample is not None and _close(sample, inferred_walking_start, baseline_duration):
                walking_start_sample = sample
                walking_start_source = f"{label} at sample {sample} (matches expected walking-start timing)"
                break

        if w1_sample is not None and w1_sample != walking_start_sample:
            gap_from_s1 = "n/a (no S1)" if s1_sample is None else f"{(w1_sample - s1_sample) / self.fs:.1f}s"
            logger.warning(
                f"Walking-start marker (W1/WALK/etc) at sample {w1_sample} rejected as "
                f"implausible (S1->marker gap={gap_from_s1}, expected ~{baseline_duration}s); "
                f"using {walking_start_source} instead."
            )

        # Only accept a real S2 as task-end if it actually fits
        # walking_start + task_duration; otherwise leave it to the caller's
        # time-based estimate.
        task_end_sample = None
        if (s2_sample is not None and s2_sample != walking_start_sample
                and s2_sample > walking_start_sample
                and _close(s2_sample, walking_start_sample + int(task_duration * self.fs), task_duration)):
            task_end_sample = s2_sample

        # baseline_start is S1 directly when we have it - it's already the
        # anchor everything above got checked against - otherwise it's
        # worked back from walking_start so the two stay consistent.
        if s1_sample is not None:
            baseline_start_sample = s1_sample
        else:
            baseline_start_sample = walking_start_sample - int(baseline_duration * self.fs)

        description = (
            f"baseline-start: {'S1 at sample ' + str(s1_sample) if s1_sample is not None else 'inferred at sample ' + str(baseline_start_sample)}; "
            f"walking-start: {walking_start_source}; task-end: "
            + (f"S2 at sample {task_end_sample} (matches expected task-end timing)"
               if task_end_sample is not None else "left to time-based estimate")
        )
        return (baseline_start_sample, walking_start_sample, task_end_sample, description)

    def _find_walking_start_event(self, events: pd.DataFrame, task_type: str) -> Optional[int]:
        """
        Find the marker that signals walking/task start. Falls through a
        chain of alternatives when the events for a long-walk task are
        mislabeled or missing the expected marker.
        """
        if task_type not in self.task_walking_events:
            logger.warning(f"Unknown task type for walking start detection: {task_type}")
            return None

        possible_events = self.task_walking_events[task_type]
        events_clean = events.copy()
        events_clean['Event_Upper'] = events_clean['Event'].astype(str).str.strip().str.upper()

        for event_name in possible_events:
            matching_events = events_clean[events_clean['Event_Upper'] == event_name.upper()]
            if not matching_events.empty:
                walking_start_sample = matching_events.iloc[0]['Sample number']
                return int(walking_start_sample)

        task_config = self.task_types.get(task_type, {})
        if task_config.get('type') == 'long_walk':
            s_events = events_clean[events_clean['Event_Upper'].str.match(r'S[1-9]')]
            if len(s_events) >= 2:
                s_events_sorted = s_events.sort_values('Sample number').reset_index(drop=True)
                second_s_event = s_events_sorted.iloc[1]
                walking_start_sample = second_s_event['Sample number']
                return int(walking_start_sample)

            s1_events = events_clean[events_clean['Event_Upper'] == 'S1']
            if not s1_events.empty:
                s1_sample = s1_events.iloc[0]['Sample number']
                events_after_s1 = events_clean[events_clean['Sample number'] > s1_sample]
                if not events_after_s1.empty:
                    next_event = events_after_s1.sort_values('Sample number').iloc[0]
                    walking_start_sample = next_event['Sample number']
                    return int(walking_start_sample)

        elif task_config.get('type') == 'event_dependent':
            s1_events = events_clean[events_clean['Event_Upper'] == 'S1']
            if not s1_events.empty:
                s1_sample = s1_events.iloc[0]['Sample number']
                s2_events = events_clean[events_clean['Event_Upper'] == 'S2']
                s2_after_s1 = s2_events[s2_events['Sample number'] > s1_sample]
                if not s2_after_s1.empty:
                    walking_start_sample = s2_after_s1.iloc[0]['Sample number']
                    return int(walking_start_sample)

                events_after_s1 = events_clean[events_clean['Sample number'] > s1_sample]
                if not events_after_s1.empty:
                    next_event = events_after_s1.iloc[0]
                    walking_start_sample = next_event['Sample number']
                    return int(walking_start_sample)

        if task_type == 'LShape':
            s2_events = events_clean[events_clean['Event_Upper'] == 'S2']
            if not s2_events.empty:
                s2_sample = s2_events.iloc[0]['Sample number']
                w1_after_s2 = events_clean[
                    (events_clean['Event_Upper'] == 'W1') & (events_clean['Sample number'] > s2_sample)]
                if not w1_after_s2.empty:
                    walking_start_sample = w1_after_s2.iloc[0]['Sample number']
                    return int(walking_start_sample)

                events_after_s2 = events_clean[events_clean['Sample number'] > s2_sample]
                if not events_after_s2.empty:
                    next_event = events_after_s2.iloc[0]
                    walking_start_sample = next_event['Sample number']
                    return int(walking_start_sample)

        if len(events_clean) >= 2:
            events_sorted = events_clean.sort_values('Sample number').reset_index(drop=True)
            second_event = events_sorted.iloc[1]
            walking_start_sample = second_event['Sample number']
            return int(walking_start_sample)

        logger.warning(f"Could not find walking start event for {task_type}")
        return None

    def _calculate_quality_and_filter(self, data: pd.DataFrame,
                                      channel_groups: dict,
                                      output_dir: str, file_basename: str,
                                      concentration_data: pd.DataFrame) -> Tuple[List[str], QualityReport]:
        """Score each channel on whichever of SQI/SCI/PSP are enabled, all
        computed on the pre-TDDR OD (and pre-TDDR concentration, for SQI
        only). A channel fails if any enabled metric that could actually be
        computed comes in under threshold. Short channels get a pass on
        exclusion unless exclude_failing_short_channels is set.

        Returns a tuple of (excluded_od_columns, quality_report):
        excluded_od_columns is the list of OD column names that should be
        dropped downstream (empty unless enable_quality_filtering is on),
        and quality_report is a QualityReport with one ChannelQuality entry
        per channel - used for the CSV output and passed back up to
        BatchProcessor/PipelineManager via process_file's return value.
        """
        report = QualityReport(
            metrics_used=self.enabled_metrics,
            sqi_threshold=self.sqi_threshold,
            sci_threshold=self.sci_threshold,
            psp_threshold=self.psp_threshold,
        )
        excluded_channels: List[str] = []

        for ch_id_str, wavelengths in channel_groups.items():
            ch_num = int(re.match(r'CH(\d+)', ch_id_str).group(1))
            is_short = ch_num in SHORT_CHANNEL_IDS

            if len(wavelengths) != 2:
                logger.warning(f"Channel {ch_id_str} has {len(wavelengths)} wavelength(s), need 2 for quality metrics")
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
                logger.warning(f"Quality calculation failed for {ch_id_str}: {str(e)}")
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
                logger.warning(f"  FAIL {ch_id_str} [{status}]: {cq.reason_text} "
                           f"(SQI={sqi_val}, SCI={sci_val}, PSP={psp_val})")
                if self.enable_quality_filtering and not spared:
                    excluded_channels.extend(list(wavelengths.values()))

        quality_suffix = "_filtered" if self.enable_quality_filtering else "_unfiltered"
        report_path = os.path.join(
            output_dir, f"{os.path.splitext(file_basename)[0]}_quality_report{quality_suffix}.csv"
        )
        try:
            report.to_dataframe().to_csv(report_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save quality report: {e}")

        return excluded_channels, report

    def _apply_tddr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run TDDR (temporal derivative distribution repair) motion correction.

        Only the OD/wavelength columns get corrected, plus 'Sample number'
        since the tddr implementation needs it - everything else (Event,
        Time (s), etc.) passes through untouched. This mirrors
        FullCapProcessor._apply_tddr, so both pipelines run TDDR the same
        way and in the same spot: after quality metrics, before the
        OD-to-concentration conversion.
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
        """Find the pre-task baseline window (start, end sample indices)
        used to reference OD -> Delta-OD. This has to agree with whatever
        _apply_baseline_correction picks for its own window, so the OD
        reference and the later concentration-level baseline correction
        are anchored to the same period.

        For LShape this is the 2nd -> 3rd event markers. For long_walk
        tasks it goes through the same cross-checked resolution that
        _apply_post_event_trimming uses (_resolve_long_walk_boundaries), so
        a mislabeled marker - W1 sitting where baseline-start should be, say
        - gets rejected here exactly like it would there, rather than being
        trusted on label alone. For everything else, if there's an S1
        marker, it prefers [S1, W1], then [S1, S2], then finally
        [S1, S1 + baseline_duration]. With no usable events at all, it
        just takes the first baseline_duration seconds of the recording.

        Returns (start, end) clamped into [0, n_samples) and guaranteed to
        span at least about a second, or None if nothing usable came out of
        that - in which case the caller falls back to a whole-record mean.
        """
        min_span = max(1, int(1.0 * self.fs))

        def _clamp(a, b):
            a = int(max(0, min(a, n_samples - 1)))
            b = int(max(0, min(b, n_samples)))
            if b - a < min_span:
                return None
            return (a, b)

        file_basename = getattr(self, "_current_file_basename", "")
        timing = self._get_task_timing(file_basename, task_type or "Unknown")
        base_samples = int(timing.get("baseline_duration", 20.0) * self.fs)

        if events is not None and not events.empty and "Event" in events.columns:
            ev = events.copy()
            ev["Sample number"] = pd.to_numeric(ev["Sample number"], errors="coerce")
            ev = ev.dropna(subset=["Sample number"])
            ev["_E"] = ev["Event"].astype(str).str.strip().str.upper()
            ev = ev.sort_values("Sample number").reset_index(drop=True)

            # LShape: use the 2nd -> 3rd markers, matching _apply_lshape_baseline.
            if task_type == "LShape" and len(ev) >= 3:
                w = _clamp(ev.iloc[1]["Sample number"], ev.iloc[2]["Sample number"])
                if w:
                    return w

            task_config = self.task_types.get(task_type, {})
            if task_config.get("type") == "long_walk":
                # Same cross-checked resolution as _apply_post_event_trimming,
                # so the OD baseline reference and the trimming boundaries
                # always land on the same real-world timeline. A mislabeled
                # marker gets rejected here the same way it would there.
                resolved = self._resolve_long_walk_boundaries(events, task_type, n_samples)
                if resolved is not None:
                    baseline_start_sample, walking_start_sample, _task_end, description = resolved
                    w = _clamp(baseline_start_sample, walking_start_sample)
                    if w:
                        return w
                    logger.warning(f"OD baseline reference: resolved boundaries ({description}) "
                                   f"produced an unusable window; falling back to first "
                                   f"{base_samples} samples.")
                # resolved came back None, nothing to work with - drop through
                # to the "no usable events" branch below.
            else:
                s1 = ev[ev["_E"] == "S1"]
                if not s1.empty:
                    s1_s = s1.iloc[0]["Sample number"]

                    # Best case: S1 -> W1, the pre-walk standing baseline.
                    w1 = ev[(ev["_E"] == "W1") & (ev["Sample number"] > s1_s)]
                    if not w1.empty:
                        w = _clamp(s1_s, w1.iloc[0]["Sample number"])
                        if w:
                            return w

                    # Next best: S1 -> S2.
                    s2 = ev[(ev["_E"] == "S2") & (ev["Sample number"] > s1_s)]
                    if not s2.empty:
                        w = _clamp(s1_s, s2.iloc[0]["Sample number"])
                        if w:
                            return w

                    # Just S1: take baseline_duration seconds after it.
                    w = _clamp(s1_s, s1_s + base_samples)
                    if w:
                        return w

        # No usable events at all - reference the first baseline_duration seconds.
        w = _clamp(0, base_samples)
        if w:
            logger.warning(f"OD baseline reference: no S1/W1 markers; using first {base_samples} samples {w}")
        else:
            logger.warning("OD baseline reference: could not build a baseline window; "
                           "falling back to whole-record mean.")
        return w

    def _convert_od_to_concentration(self, data, od_cols, channel_groups,
                                     events=None, task_type=None):
        """Convert OD to Delta-concentration via MBLL, using Prahl/OMLC
        extinction coefficients (cm^-1/M), per-channel pathlength, and OD
        referenced to the pre-walk baseline window (normally S1->W1).

        Each channel's OD gets referenced to its own mean over that
        baseline window before the MBLL solve - Delta-OD = OD minus the
        baseline mean - so the resulting Delta[Hb] is anchored to the
        resting/standing period. If no baseline window can be found
        (missing events/task_type, or missing markers), it falls back to
        the first baseline_duration seconds, and if even that can't be
        built, to the whole-record per-channel mean.

        The extinction table only covers the narrow 2nm band around this
        device's actual light-source wavelengths (756-759nm and
        846-848nm). Odd-nm entries are linearly interpolated from the
        tabulated even-nm neighbors, and 847nm comes from interpolating
        between the 846/848 anchors via np.interp.
        """
        try:
            DPF = 6.0  # fixed for this device/protocol
            DISTANCE_LONG_CM = 3.5  # long channels: 35mm source-detector separation
            DISTANCE_SHORT_CM = 1.5  # short channels (CH3/CH5): 15mm separation

            n_samples = len(data)
            window = self._baseline_window_samples(events, task_type, n_samples)
            if window is not None:
                b_start, b_end = window
            else:
                b_start = b_end = None

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

            return concentration_data if not concentration_data.empty else None

        except Exception as e:
            logger.error(f"OD to concentration conversion failed: {str(e)}", exc_info=True)
            return None

    def _determine_task_type(self, file_basename: str) -> str:
        """
        Figure out the task type from the filename using boundary-aware
        matching.

        Worth flagging: fTurn only gets detected from an explicit 'FTURN'
        or 'F_TURN' in the name. Turn_DT and Turn_ST are walking tasks, not
        fTurn, and get classified as DT/ST instead.
        """
        s = file_basename.upper()

        if "FTURN" in s or "F_TURN" in s:
            return "fTurn"

        if "LSHAPE" in s or "L_SHAPE" in s:
            return "LShape"

        if "OBSTACLE" in s:
            return "Obstacle"

        if "NAVIGATION" in s or re.search(r'\bNAV\b', s):
            return "Navigation"

        # Catches Turn_DT, Walking_DT, Walking_DT-AC, etc. - all long_walk
        # tasks with event-based or fallback baseline handling.
        if re.search(r'(^|[^A-Z])DT([^A-Z]|$)', s):
            return "DT"
        if re.search(r'(^|[^A-Z])ST([^A-Z]|$)', s):
            return "ST"

        if "WALK" in s:
            return "LongWalk"

        logger.warning(f"Unknown task type from filename: {file_basename}")
        return "Unknown"

    def _validate_task_requirements(self, task_type: str, task_config: dict, events: pd.DataFrame,
                                    filename: str) -> bool:
        """Check whether a task has what it needs before we bother processing it."""
        task_category = task_config['type']
        min_events = task_config['min_events']

        if task_category == 'event_dependent':
            if events is None or len(events) < min_events:
                logger.error(
                    f"{task_type} task requires at least {min_events} event markers, but only found {len(events) if events is not None else 0} in {filename}")
                return False

            if task_type == "LShape":
                if len(events) < 3:
                    logger.error(
                        f"L-Shape task requires at least 3 event markers, but only found {len(events)} in {filename}")
                    return False
            elif 'S1' not in events['Event'].str.upper().values:
                logger.error(f"{task_type} task requires 'S1' baseline marker, but not found in {filename}")
                return False

        elif task_category == 'long_walk':
            if events is None or events.empty:
                logger.warning(f"{task_type} task has no event markers, will use time-based fallback")

        return True

    def _extract_and_clean_events(self, data_dict: dict, data: pd.DataFrame) -> pd.DataFrame:
        """Pull events out of the data dict (or the raw data if needed) and clean them up."""
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
            logger.warning(f"Error cleaning events: {str(e)}")
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
    def _prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw data DataFrame."""
        data = raw_data.copy()
        if "Sample number" not in data.columns:
            data.insert(0, "Sample number", np.arange(len(data)))
        return data

    def _drop_initial_seconds(self, data: pd.DataFrame, seconds: float) -> pd.DataFrame:
        """Cut the first few seconds off a recording to get rid of
        device/init artifacts, same as FullCapProcessor does. Renumbers
        'Sample number' and 'Time (s)' afterward so everything downstream
        sees a clean, zero-based timeline. No-op when seconds <= 0.
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

        return cropped

    def _apply_scr(self, data: pd.DataFrame, quality_report: QualityReport) -> pd.DataFrame:
        """Short channel regression, matched by hemisphere: CH3 (right
        short) corrects the right long channels, CH5 (left short) corrects
        the left ones. See the RIGHT_LONG_IDS/LEFT_LONG_IDS/RIGHT_SHORT_ID/
        LEFT_SHORT_ID constants up top for the exact mapping, and the
        caveat about how it was inferred.

        If one side's short channel genuinely failed quality (not just
        "spared" - see _short_channel_genuinely_passed), we borrow the
        other side's short channel for both hemispheres instead. If
        neither one genuinely passed, SCR gets skipped entirely and the
        long channels stay uncorrected. Either fallback gets written to
        quality_report.scr_note so it shows up in the stats output too,
        not just buried in the logs.
        """
        try:
            sig_cols = [c for c in data.columns if any(k in c for k in ("HbO", "O2Hb", "HHb", "HbR"))]
            if not sig_cols:
                quality_report.scr_note = "SCR skipped: no signal columns present."
                return data

            def cols_for(ids: Iterable[int]) -> List[str]:
                wanted = {f"CH{i}" for i in ids}
                return [c for c in sig_cols if c.split()[0] in wanted]

            right_long_cols = cols_for(RIGHT_LONG_IDS)
            left_long_cols = cols_for(LEFT_LONG_IDS)
            right_short_cols = cols_for([RIGHT_SHORT_ID])
            left_short_cols = cols_for([LEFT_SHORT_ID])

            right_ok = bool(right_short_cols) and self._short_channel_genuinely_passed(quality_report, RIGHT_SHORT_ID)
            left_ok = bool(left_short_cols) and self._short_channel_genuinely_passed(quality_report, LEFT_SHORT_ID)

            if not right_ok and not left_ok:
                note = (f"SCR skipped: neither short channel (CH{RIGHT_SHORT_ID}, CH{LEFT_SHORT_ID}) "
                        f"genuinely passed quality; long channels left uncorrected.")
                logger.warning(note)
                quality_report.scr_note = note
                return data

            out = data.copy()

            def run_side(long_cols: List[str], short_cols: List[str]) -> None:
                if not long_cols or not short_cols:
                    return
                corrected = scr_regression(data[long_cols], data[short_cols])
                for col in corrected.columns:
                    out[col] = corrected[col]

            if right_ok and left_ok:
                run_side(right_long_cols, right_short_cols)
                run_side(left_long_cols, left_short_cols)
            elif right_ok:  # left failed
                run_side(right_long_cols, right_short_cols)
                run_side(left_long_cols, right_short_cols)
                note = (f"SCR: left short channel (CH{LEFT_SHORT_ID}) failed quality; "
                        f"used right short channel (CH{RIGHT_SHORT_ID}) for both hemispheres.")
                logger.warning(note)
                quality_report.scr_note = note
            else:  # right failed, left_ok
                run_side(left_long_cols, left_short_cols)
                run_side(right_long_cols, left_short_cols)
                note = (f"SCR: right short channel (CH{RIGHT_SHORT_ID}) failed quality; "
                        f"used left short channel (CH{LEFT_SHORT_ID}) for both hemispheres.")
                logger.warning(note)
                quality_report.scr_note = note

            return out

        except Exception as e:
            logger.warning(f"SCR failed: {str(e)}")
            quality_report.scr_note = f"SCR failed with exception: {e}"
            return data

    @staticmethod
    def _short_channel_genuinely_passed(quality_report: QualityReport, channel: int) -> bool:
        """True for a clean pass only, not for a short channel that got
        "spared" despite failing. With exclude_failing_short_channels=False
        (the default), a failing short channel still gets marked
        passed=True so it stays in the dataset - but SCR needs to know it
        actually failed, otherwise the fallback logic above would never
        kick in for a genuinely bad short channel. A clean pass has an
        empty reasons tuple; a spared-but-failing one always carries a
        "short channel kept despite: ..." reason even though passed=True.
        """
        for c in quality_report.channels:
            if c.channel == channel:
                return c.passed and not c.reasons
        return False

    def _apply_baseline_correction(self, data: pd.DataFrame, events: pd.DataFrame,
                                   task_type: str = None) -> pd.DataFrame:
        """Dispatch to whichever baseline correction approach fits the task."""
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
        """L-Shape's baseline correction: uses the 2nd and 3rd event markers as start/end."""
        try:
            events_sorted = events.sort_values('Sample number').reset_index(drop=True)

            if len(events_sorted) < 3:
                logger.error(f"L-Shape task requires at least 3 event markers, but only found {len(events_sorted)}")
                return None

            second_event = events_sorted.iloc[1]
            third_event = events_sorted.iloc[2]

            second_sample = second_event['Sample number']
            third_sample = third_event['Sample number']

            if third_sample <= second_sample:
                logger.error("L-Shape baseline error: events out of order")
                return None

            baseline_events = pd.DataFrame({
                'Sample number': [second_sample, third_sample],
                'Event': ['BaselineStart', 'BaselineEnd']
            })

            return baseline_subtraction(data, baseline_events, baseline_type="lshape_task")

        except Exception as e:
            logger.error(f"L-Shape baseline correction failed: {str(e)}")
            return None

    def _apply_standard_baseline(self, data: pd.DataFrame, events: pd.DataFrame, task_type: str = None) -> pd.DataFrame:
        """Baseline correction for everything except L-Shape."""
        try:
            task_config = self.task_types.get(task_type, {})

            if task_config.get('type') == 'long_walk':
                # Same cross-checked resolution as _baseline_window_samples
                # and _apply_post_event_trimming, so the OD reference, this
                # concentration-level correction, and the trimming
                # boundaries all agree on one consistent timeline. A
                # mislabeled marker gets rejected here the same as anywhere
                # else in the pipeline.
                resolved = self._resolve_long_walk_boundaries(events, task_type, len(data))
                if resolved is not None:
                    baseline_start_sample, walking_start_sample, _task_end, description = resolved
                    if walking_start_sample > baseline_start_sample:
                        baseline_events = pd.DataFrame({
                            'Sample number': [baseline_start_sample, walking_start_sample],
                            'Event': ['BaselineStart', 'BaselineEnd']
                        })
                        return baseline_subtraction(data, baseline_events, baseline_type="long_walk")
                logger.warning("No usable baseline markers found for long walk task, using time-based fallback")
                return self._fallback_baseline_correction(data, task_type)

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

            logger.error("No valid baseline markers found for event-dependent task")
            return None

        except Exception as e:
            logger.error(f"Standard baseline correction failed: {str(e)}")
            return None

    def _fallback_baseline_correction(self, data: pd.DataFrame, task_type: str = None) -> pd.DataFrame:
        """
        Baseline correction with no real events to work from - counts
        backward from the end of the recording using task-specific timing
        from _get_task_timing(), which already knows whether this is a 90s
        or 150s task based on the filename.
        """
        try:
            total = len(data)

            file_basename = getattr(self, '_current_file_basename', '')
            timing = self._get_task_timing(file_basename, task_type or 'Unknown')

            total_expected = timing['total_expected']
            baseline_duration = timing['baseline_duration']
            end_duration = timing['end_duration']

            min_required = int(baseline_duration * self.fs)
            if total < min_required:
                logger.warning(f"Fallback baseline: record too short ({total} samples < {min_required} required); "
                               f"returning data without subtraction")
                return data

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
        Build and save the final outputs, running Z-transformation before
        averaging rather than after. Post-event trimming has already
        happened back in _process_pipeline_stages, so this is purely
        averaging/saving/plotting. If compute_zscore is False, the whole
        Z-transformation path - averaging, CSV, plot - gets skipped, and
        only RAW comes out.
        """
        try:
            # Passing the montage explicitly here rather than letting
            # average_channels() guess zero- vs one-based numbering from
            # whether CH0 happens to be present. If SQI filtering ever drops
            # CH0 specifically, that guess could silently flip to the wrong
            # montage table (see the SHORT_CHANNEL_IDS comment up top for
            # where this mapping actually comes from) - being explicit here
            # avoids that failure mode entirely.
            averaged_raw = average_channels(
                data.copy(),
                short_ids=sorted(SHORT_CHANNEL_IDS),
                left_ids=list(LEFT_LONG_IDS),
                right_ids=list(RIGHT_LONG_IDS),
            )

            for col in ['grand oxy', 'grand deoxy']:
                if col not in averaged_raw.columns:
                    logger.warning(f"Missing '{col}' after averaging; filling with NaNs.")
                    averaged_raw[col] = np.nan

            averaged_z = None
            if self.compute_zscore:
                signal_cols = [col for col in data.columns
                               if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb'])
                               and pd.api.types.is_numeric_dtype(data[col])
                               and 'grand' not in col.lower()]

                if signal_cols:
                    z_transformed_data = z_transformation(data.copy(), signal_cols)
                else:
                    logger.warning("No individual signal channels found for Z-transformation")
                    z_transformed_data = data.copy()

                averaged_z = average_channels(
                    z_transformed_data,
                    short_ids=sorted(SHORT_CHANNEL_IDS),
                    left_ids=list(LEFT_LONG_IDS),
                    right_ids=list(RIGHT_LONG_IDS),
                )

                for col in ['grand oxy', 'grand deoxy']:
                    if col not in averaged_z.columns:
                        averaged_z[col] = np.nan

            df_versions = [averaged_raw] + ([averaged_z] if averaged_z is not None else [])
            for df_version in df_versions:
                if 'Sample number' in df_version.columns:
                    df_version['Time (s)'] = df_version['Sample number'] / self.fs
                else:
                    df_version['Time (s)'] = np.arange(len(df_version)) / self.fs

            task_category = task_config.get('type', 'unknown')
            source_filename = os.path.splitext(file_basename)[0]

            for df_version in df_versions:
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

            if averaged_z is not None:
                output_file_z = os.path.join(condition_dir, f"{file_basename}_FULLY_PROCESSED_ZSCORE{quality_suffix}.csv")
                averaged_z.to_csv(output_file_z, index=False)

            try:
                plot_condition = task_type

                self._create_final_plot(
                    averaged_raw, condition_dir, file_basename, f"{plot_condition}{quality_suffix}",
                    ['grand oxy', 'grand deoxy'],
                    f'final_overall_RAW{quality_suffix}',
                    f'Final Overall - Raw Concentrations{quality_suffix}', events
                )

                if averaged_z is not None:
                    self._create_final_plot(
                        averaged_z, condition_dir, file_basename, f"{plot_condition}{quality_suffix}",
                        ['grand oxy', 'grand deoxy'],
                        f'final_overall_ZSCORE{quality_suffix}',
                        f'Final Overall - Z-scores{quality_suffix}', events
                    )
            except Exception as e:
                logger.warning(f"Plotting failed for {file_basename}: {e}")

            return averaged_raw

        except Exception as e:
            logger.error(f"Final output generation failed: {str(e)}", exc_info=True)
            return None

    def _create_final_plot(self, data: pd.DataFrame, output_dir: str,
                           file_basename: str, condition: str,
                           columns: List[str], prefix: str,
                           title: str,
                           events: Optional[pd.DataFrame] = None) -> None:
        """Build one of the final overall plots, with auto-scaled axes and events remapped to line up with the trimmed data."""
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
        except Exception as e:
            plt.close(fig)
            logger.error(f"Failed to save figure {os.path.basename(path)}: {str(e)}")
            raise
