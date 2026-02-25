import os
import argparse
import logging
from typing import List, Optional, Tuple, Dict
from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.stats_collector import StatsCollector
from fnirs_PFC_2025.read.loaders import read_txt_file
import re

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Enhanced batch processor for multi-task fNIRS files with dual CV processing and post-walking trimming."""

    def __init__(self, fs: float = 50.0, sqi_threshold: float = 2.0,
                 post_walking_trim_seconds: float = 3.0):
        """
        Initialize batch processor.

        Args:
            fs: Sampling frequency in Hz
            sqi_threshold: SQI threshold value for quality assessment (1-5 scale, default 2.5)
            post_walking_trim_seconds: Seconds to trim after walking start event (default 3.0)
        """
        self.fs = fs
        self.sqi_threshold = sqi_threshold
        self.post_walking_trim_seconds = post_walking_trim_seconds

        # Create two processors: one with SQI filtering, one without - BOTH with trimming
        self.processor_no_sqi = FileProcessor(
            fs=fs,
            sqi_threshold=sqi_threshold,
            enable_sqi_filtering=False,
            post_walking_trim_seconds=post_walking_trim_seconds
        )
        self.processor_with_sqi = FileProcessor(
            fs=fs,
            sqi_threshold=sqi_threshold,
            enable_sqi_filtering=True,
            post_walking_trim_seconds=post_walking_trim_seconds
        )
        self.stats_collector = StatsCollector(fs=fs)

        # Track task-specific processing results for both batches
        self.task_results = {
            'no_sqi_filtering': {
                'processed': {},
                'skipped': {},
                'failed': {},
                'validation_failed': {}
            },
            'with_sqi_filtering': {
                'processed': {},
                'skipped': {},
                'failed': {},
                'validation_failed': {}
            }
        }

        logger.info(f"Initialized BatchProcessor with DUAL SQI processing "
                    f"(fs={fs}, SQI threshold={sqi_threshold}, post-walking trim={post_walking_trim_seconds}s)")

    def find_input_files(self, input_dir: str, task_filter: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Recursively find all .txt files in directory, organized by task type."""
        all_files: List[str] = []
        task_files: Dict[str, List[str]] = {}

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    all_files.append(os.path.join(root, file))

        for file_path in sorted(all_files):
            task_type = self._determine_task_type(file_path)
            if task_filter and task_type not in task_filter:
                continue
            task_files.setdefault(task_type, []).append(file_path)

        logger.info("Found files by task type:")
        for task_type, files in task_files.items():
            logger.info(f"  {task_type}: {len(files)} files")

        return task_files

    def _determine_task_type(self, file_path: str) -> str:
        """Determine task type from filename using boundary-aware detection."""
        import re
        basename = os.path.basename(file_path).upper()

        # Specific keywords first
        if "FTURN" in basename or "F_TURN" in basename or re.search(r'\bTURN\b', basename):
            return "fTurn"
        if "LSHAPE" in basename or "L_SHAPE" in basename:
            return "LShape"
        if "FIGURE8" in basename or "FIG8" in basename:
            return "Figure8"
        if "OBSTACLE" in basename:
            return "Obstacle"
        if "NAVIGATION" in basename or re.search(r'\bNAV\b', basename):
            return "Navigation"
        if "WALK" in basename:
            return "LongWalk"

        # Strict DT/ST tokens (avoid POST→ST and IDENT→DT)
        if re.search(r'(^|[^A-Z])DT([^A-Z]|$)', basename):
            return "DT"
        if re.search(r'(^|[^A-Z])ST([^A-Z]|$)', basename):
            return "ST"

        logger.warning(f"Could not determine task type from filename: {basename}")
        return "Unknown"

    def _process_files_batch(self,
                             task_files: Dict[str, List[str]],
                             input_dir: str,
                             output_dir: str,
                             processor: FileProcessor,
                             batch_name: str,
                             subject_y_limits: Optional[Dict] = None,
                             strict_validation: bool = True) -> Tuple[List[str], List[str]]:
        """Process files and return ORIGINAL INPUT file paths that were processed."""

        processed_input_files = []
        skipped_files = []

        total_tasks = len(task_files)
        current_task = 0

        for task_type, files in task_files.items():
            current_task += 1
            logger.info(f"Processing task {current_task}/{total_tasks}: {task_type} ({len(files)} files)")

            for file_idx, file_path in enumerate(files, 1):
                file_basename = os.path.basename(file_path)
                logger.info(f"  [{file_idx}/{len(files)}] Processing: {file_basename}")

                try:
                    # Determine subject for y-limits
                    subject = self._extract_subject_from_path(file_path)
                    y_limits = subject_y_limits.get(subject) if subject_y_limits else None

                    # Process the file
                    result = processor.process_file(
                        file_path=file_path,
                        output_base_dir=output_dir,
                        input_base_dir=input_dir,
                        subject_y_limits=y_limits,
                        read_file_func=read_txt_file
                    )

                    # Check if processing was successful
                    if result and result.get('success'):
                        processed_input_files.append(file_path)
                        self.task_results[batch_name]['processed'][task_type].append(file_path)
                        logger.info(f"    Successfully processed")
                    elif result and result.get('validation_failed'):
                        skipped_files.append(file_path)
                        self.task_results[batch_name]['validation_failed'][task_type].append(file_path)
                        logger.warning(f"    Validation failed: {result.get('error', 'Unknown reason')}")
                    else:
                        skipped_files.append(file_path)
                        self.task_results[batch_name]['failed'][task_type].append(file_path)
                        logger.warning(
                            f"    Processing failed: {result.get('error', 'Unknown reason') if result else 'No result returned'}")

                except Exception as e:
                    logger.error(f"    Error processing {file_basename}: {e}", exc_info=True)
                    skipped_files.append(file_path)
                    self.task_results[batch_name]['failed'][task_type].append(file_path)

            logger.info(
                f"  Task {task_type} complete: {len(self.task_results[batch_name]['processed'][task_type])} processed")

        logger.info(
            f"Batch '{batch_name}' complete: {len(processed_input_files)} successfully processed, {len(skipped_files)} skipped/failed")
        return processed_input_files, skipped_files

    def _extract_subject_from_path(self, file_path: str) -> Optional[str]:
        """Extract subject ID from file path."""
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if "Turn_" in part or "OHSU_Turn" in part:
                return part
        return None

    def _log_dual_processing_summary(self, total_files: int,
                                     batch1_processed: List[str], batch1_skipped: List[str],
                                     batch2_processed: List[str], batch2_skipped: List[str]) -> None:
        """Log comprehensive dual processing summary."""
        logger.info("\n" + "=" * 80)
        logger.info("DUAL BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total files: {total_files}")
        logger.info(f"Processing approach: Dual batch (each file processed twice)")
        logger.info(f"Post-walking trimming: {self.post_walking_trim_seconds}s applied to both batches")
        logger.info("")

        logger.info("BATCH 1 (No SQI Filtering - All Channels):")
        logger.info(f"  Successfully processed: {len(batch1_processed)}")
        logger.info(f"  Skipped/Failed: {len(batch1_skipped)}")
        if total_files > 0:
            success_rate1 = len(batch1_processed) / total_files * 100
            logger.info(f"  Success rate: {success_rate1:.1f}%")

        logger.info("")
        logger.info("BATCH 2 (With SQI Filtering - Poor Quality Channels Excluded):")
        logger.info(f"  Successfully processed: {len(batch2_processed)}")
        logger.info(f"  Skipped/Failed: {len(batch2_skipped)}")
        if total_files > 0:
            success_rate2 = len(batch2_processed) / total_files * 100
            logger.info(f"  Success rate: {success_rate2:.1f}%")

        # Task-specific summary for both batches
        logger.info("\nDetailed Breakdown by Task Type:")
        logger.info("-" * 60)

        for batch_name, batch_label in [('no_sqi_filtering', 'No SQI Filter'),
                                        ('with_sqi_filtering', 'With SQI Filter')]:
            logger.info(f"\n{batch_label}:")
            batch_results = self.task_results[batch_name]

            for task_type in sorted(batch_results['processed'].keys()):
                processed_count = len(batch_results['processed'][task_type])
                validation_failed = len(batch_results['validation_failed'][task_type])
                if processed_count + validation_failed > 0:
                    logger.info(f"  {task_type}: {processed_count} processed, {validation_failed} validation failed")

        logger.info("=" * 80)

    def run_two_pass_processing(self,
                                input_dir: str,
                                output_dir: str,
                                task_filter: Optional[List[str]] = None,
                                strict_validation: bool = True,
                                enable_dual_sqi: bool = True) -> dict:
        """
        Execute processing pipeline with optional dual SQI processing.

        Args:
            enable_dual_sqi: If True, runs dual batch processing. If False, runs original single batch.
        """
        if enable_dual_sqi:
            logger.info(" Running DUAL SQI processing mode")
            return self.run_dual_sqi_processing(input_dir, output_dir, task_filter, strict_validation)
        else:
            logger.info(" Running SINGLE batch processing mode (original behavior)")
            return self._run_original_processing(input_dir, output_dir, task_filter, strict_validation)

    def _run_original_processing(self,
                                 input_dir: str,
                                 output_dir: str,
                                 task_filter: Optional[List[str]] = None,
                                 strict_validation: bool = True) -> dict:
        """Original single-batch processing for backward compatibility."""
        os.makedirs(output_dir, exist_ok=True)

        # Determine which processor to use based on whether SQI filtering is requested
        # Check if sqi_filtering_only was requested by checking if the with_sqi processor
        # should be used instead. We detect this by checking if the no_sqi processor has
        # filtering disabled (it always does) - the caller controls which mode via
        # the sqi_filtering_only flag in main.py
        # For single batch, we need to know if user wants SQI filtering or not.
        # This is determined by checking the 'sqi_filtering_only' context.
        # Since _run_original_processing is called for single_batch mode,
        # we check if sqi_filtering_only was set by looking at the processor config.

        # Find files organized by task type
        task_files = self.find_input_files(input_dir, task_filter)
        total_files = sum(len(files) for files in task_files.values())

        logger.info(f"Found {total_files} input files across {len(task_files)} task types")

        if not task_files:
            raise ValueError("No .txt files found in input directory")

        # Initialize task results tracking for single batch
        for task_type in task_files.keys():
            self.task_results['no_sqi_filtering']['processed'][task_type] = []
            self.task_results['no_sqi_filtering']['skipped'][task_type] = []
            self.task_results['no_sqi_filtering']['failed'][task_type] = []
            self.task_results['no_sqi_filtering']['validation_failed'][task_type] = []

        # First pass - initial processing
        logger.info("--- First pass: Initial processing ---")
        processed_files, skipped_files = self._process_files_batch(
            task_files, input_dir, output_dir,
            processor=self.processor_no_sqi,
            batch_name="no_sqi_filtering",
            subject_y_limits=None,
            strict_validation=strict_validation
        )

        # Calculate consistent y-limits per subject
        logger.info("--- Calculating consistent y-limits ---")
        subject_y_limits = None
        if processed_files:
            subject_y_limits = self.stats_collector.calculate_subject_y_limits(
                processed_files, output_dir, input_dir)

        # Second pass - processing with consistent y-limits
        logger.info("--- Second pass: Processing with y-limits ---")
        final_processed, final_skipped = self._process_files_batch(
            task_files, input_dir, output_dir,
            processor=self.processor_no_sqi,
            batch_name="no_sqi_filtering",
            subject_y_limits=subject_y_limits,
            strict_validation=strict_validation
        )

        # Generate statistics - FIXED: handle RAW and ZSCORE separately with null checks
        logger.info("--- Generating statistics ---")
        stats_raw = None
        stats_zscore = None

        if final_processed:
            # RAW statistics
            logger.info("    Calculating RAW concentration statistics...")
            stats_raw = self.stats_collector.run_statistics(
                processed_files=final_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir,
                file_type="RAW"
            )

            if stats_raw is not None and not stats_raw.empty:
                logger.info("--- Creating RAW summary sheets ---")
                self.stats_collector.create_summary_sheets(stats_raw, output_dir, suffix="_RAW")
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_RAW.csv')
                stats_raw.to_csv(stats_path, index=False)
                logger.info(f"Saved RAW statistics to: {stats_path}")
            else:
                logger.warning("No RAW statistics were generated")

            # ZSCORE statistics
            logger.info("    Calculating Z-score statistics...")
            stats_zscore = self.stats_collector.run_statistics(
                processed_files=final_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir,
                file_type="ZSCORE"
            )

            if stats_zscore is not None and not stats_zscore.empty:
                logger.info("--- Creating ZSCORE summary sheets ---")
                self.stats_collector.create_summary_sheets(stats_zscore, output_dir, suffix="_ZSCORE")
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_ZSCORE.csv')
                stats_zscore.to_csv(stats_path, index=False)
                logger.info(f"Saved ZSCORE statistics to: {stats_path}")
            else:
                logger.warning("No ZSCORE statistics were generated")

        # Generate detailed task-specific report
        self._generate_task_report(output_dir, total_files)

        # Log final summary
        self._log_processing_summary(total_files, final_processed, final_skipped)

        return {
            'stats_raw': stats_raw,
            'stats_zscore': stats_zscore,
            'processed_files': final_processed,
            'skipped_files': final_skipped,
            'total_files': total_files,
            'task_results': self.task_results['no_sqi_filtering']
        }

    def run_dual_sqi_processing(self,
                                input_dir: str,
                                output_dir: str,
                                task_filter: Optional[List[str]] = None,
                                strict_validation: bool = True) -> dict:
        """
        Execute complete dual SQI processing pipeline with separate output directories.
        Processes all files twice: once with all channels, once with SQI filtering.
        """
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create separate output directories for each batch to prevent file conflicts
        output_dir_no_sqi = os.path.join(output_dir, "batch_no_SQI_filtering")
        output_dir_with_sqi = os.path.join(output_dir, "batch_with_SQI_filtering")

        os.makedirs(output_dir_no_sqi, exist_ok=True)
        os.makedirs(output_dir_with_sqi, exist_ok=True)

        # Find files organized by task type
        task_files = self.find_input_files(input_dir, task_filter)
        total_files = sum(len(files) for files in task_files.values())

        logger.info(f"Found {total_files} input files across {len(task_files)} task types")
        logger.info("Will process each file TWICE: once without SQI filtering, once with SQI filtering")
        logger.info(f"Post-walking trimming: {self.post_walking_trim_seconds}s applied to both batches")
        logger.info(f"Output directories:")
        logger.info(f"  Batch 1 (No SQI): {output_dir_no_sqi}")
        logger.info(f"  Batch 2 (With SQI): {output_dir_with_sqi}")

        if not task_files:
            raise ValueError("No .txt files found in input directory")

        # Initialize task results tracking
        for batch_type in ['no_sqi_filtering', 'with_sqi_filtering']:
            for task_type in task_files.keys():
                for result_type in ['processed', 'skipped', 'failed', 'validation_failed']:
                    self.task_results[batch_type][result_type][task_type] = []

        # BATCH 1: Process without SQI filtering
        logger.info("=" * 80)
        logger.info("BATCH 1: Processing WITHOUT SQI filtering (all channels included)")
        logger.info(f"Output directory: {output_dir_no_sqi}")
        logger.info("=" * 80)

        batch1_processed, batch1_skipped = self._process_files_batch(
            task_files=task_files,
            input_dir=input_dir,
            output_dir=output_dir_no_sqi,
            processor=self.processor_no_sqi,
            batch_name="no_sqi_filtering",
            subject_y_limits=None,
            strict_validation=strict_validation
        )

        # Calculate y-limits from batch 1 for reference
        logger.info("--- Calculating y-limits from batch 1 for reference ---")
        subject_y_limits = None
        if batch1_processed:
            subject_y_limits = self.stats_collector.calculate_subject_y_limits(
                batch1_processed, output_dir, input_dir)
            logger.info(f"Calculated y-limits for {len(subject_y_limits) if subject_y_limits else 0} subjects")

        # BATCH 2: Process with SQI filtering
        logger.info("=" * 80)
        logger.info("BATCH 2: Processing WITH SQI filtering (poor quality channels excluded)")
        logger.info(f"Output directory: {output_dir_with_sqi}")
        logger.info("=" * 80)

        batch2_processed, batch2_skipped = self._process_files_batch(
            task_files=task_files,
            input_dir=input_dir,
            output_dir=output_dir_with_sqi,
            processor=self.processor_with_sqi,
            batch_name="with_sqi_filtering",
            subject_y_limits=None,
            strict_validation=strict_validation
        )

        # Generate statistics for both batches
        logger.info("=" * 80)
        logger.info("Generating statistics for both batches (RAW and ZSCORE versions)")
        logger.info("=" * 80)

        stats_batch1_raw = None
        stats_batch1_zscore = None
        stats_batch2_raw = None
        stats_batch2_zscore = None

        if batch1_processed:
            logger.info("--- Generating statistics for batch 1 (no SQI filtering) ---")

            logger.info("    Calculating RAW concentration statistics...")
            stats_batch1_raw = self.stats_collector.run_statistics(
                processed_files=batch1_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir_no_sqi,
                file_type="RAW"
            )

            if stats_batch1_raw is not None and not stats_batch1_raw.empty:
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_RAW_no_SQI_filtering.csv')
                stats_batch1_raw.to_csv(stats_path, index=False)
                logger.info(f"    Saved RAW batch 1 statistics to: {stats_path}")

                batch_stats_path = os.path.join(output_dir_no_sqi, 'all_subjects_statistics_RAW_no_SQI_filtering.csv')
                stats_batch1_raw.to_csv(batch_stats_path, index=False)

                self.stats_collector.create_summary_sheets(stats_batch1_raw, output_dir_no_sqi,
                                                           suffix="_RAW_no_SQI_filtering")
            else:
                logger.warning("    No RAW statistics generated for batch 1")

            logger.info("    Calculating Z-score statistics...")
            stats_batch1_zscore = self.stats_collector.run_statistics(
                processed_files=batch1_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir_no_sqi,
                file_type="ZSCORE"
            )

            if stats_batch1_zscore is not None and not stats_batch1_zscore.empty:
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_ZSCORE_no_SQI_filtering.csv')
                stats_batch1_zscore.to_csv(stats_path, index=False)
                logger.info(f"    Saved ZSCORE batch 1 statistics to: {stats_path}")

                batch_stats_path = os.path.join(output_dir_no_sqi,
                                                'all_subjects_statistics_ZSCORE_no_SQI_filtering.csv')
                stats_batch1_zscore.to_csv(batch_stats_path, index=False)

                self.stats_collector.create_summary_sheets(stats_batch1_zscore, output_dir_no_sqi,
                                                           suffix="_ZSCORE_no_SQI_filtering")
            else:
                logger.warning("    No ZSCORE statistics generated for batch 1")

        if batch2_processed:
            logger.info("--- Generating statistics for batch 2 (with SQI filtering) ---")

            logger.info("    Calculating RAW concentration statistics...")
            stats_batch2_raw = self.stats_collector.run_statistics(
                processed_files=batch2_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir_with_sqi,
                file_type="RAW"
            )

            if stats_batch2_raw is not None and not stats_batch2_raw.empty:
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_RAW_with_SQI_filtering.csv')
                stats_batch2_raw.to_csv(stats_path, index=False)
                logger.info(f"    Saved RAW batch 2 statistics to: {stats_path}")

                batch_stats_path = os.path.join(output_dir_with_sqi,
                                                'all_subjects_statistics_RAW_with_SQI_filtering.csv')
                stats_batch2_raw.to_csv(batch_stats_path, index=False)

                self.stats_collector.create_summary_sheets(stats_batch2_raw, output_dir_with_sqi,
                                                           suffix="_RAW_with_SQI_filtering")
            else:
                logger.warning("    No RAW statistics generated for batch 2")

            logger.info("    Calculating Z-score statistics...")
            stats_batch2_zscore = self.stats_collector.run_statistics(
                processed_files=batch2_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir_with_sqi,
                file_type="ZSCORE"
            )

            if stats_batch2_zscore is not None and not stats_batch2_zscore.empty:
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_ZSCORE_with_SQI_filtering.csv')
                stats_batch2_zscore.to_csv(stats_path, index=False)
                logger.info(f"    Saved ZSCORE batch 2 statistics to: {stats_path}")

                batch_stats_path = os.path.join(output_dir_with_sqi,
                                                'all_subjects_statistics_ZSCORE_with_SQI_filtering.csv')
                stats_batch2_zscore.to_csv(batch_stats_path, index=False)

                self.stats_collector.create_summary_sheets(stats_batch2_zscore, output_dir_with_sqi,
                                                           suffix="_ZSCORE_with_SQI_filtering")
            else:
                logger.warning("    No ZSCORE statistics generated for batch 2")

        # Generate comprehensive reports in main output directory
        self._generate_dual_batch_report(output_dir, total_files)

        # Log final summary
        self._log_dual_processing_summary(total_files, batch1_processed, batch1_skipped,
                                          batch2_processed, batch2_skipped)

        return {
            'stats_batch1_raw': stats_batch1_raw,
            'stats_batch1_zscore': stats_batch1_zscore,
            'stats_batch2_raw': stats_batch2_raw,
            'stats_batch2_zscore': stats_batch2_zscore,
            'batch1_processed': batch1_processed,
            'batch1_skipped': batch1_skipped,
            'batch2_processed': batch2_processed,
            'batch2_skipped': batch2_skipped,
            'total_files': total_files,
            'task_results': self.task_results,
            'output_dir_no_sqi': output_dir_no_sqi,
            'output_dir_with_sqi': output_dir_with_sqi
        }

    def _generate_task_report(self, output_dir: str, total_files: int) -> None:
        """Generate detailed task-specific processing report (original single batch version)."""
        report_path = os.path.join(output_dir, 'task_processing_report.txt')
        batch_results = self.task_results['no_sqi_filtering']

        with open(report_path, 'w') as f:
            f.write("fNIRS Multi-Task Processing Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total files found: {total_files}\n")
            f.write(f"Sampling frequency: {self.fs} Hz\n")
            f.write(f"SQI threshold: {self.sqi_threshold}%\n")
            f.write(f"Post-walking trimming: {self.post_walking_trim_seconds}s\n")
            f.write("\n")

            # Task-by-task breakdown
            for task_type in sorted(batch_results['processed'].keys()):
                processed = len(batch_results['processed'][task_type])
                skipped = len(batch_results['skipped'][task_type])
                failed = len(batch_results['failed'][task_type])
                validation_failed = len(batch_results['validation_failed'][task_type])
                total_task = processed + skipped + failed + validation_failed

                if total_task == 0:
                    continue

                f.write(f"{task_type} Task Results:\n")
                f.write(f"  Total files: {total_task}\n")
                f.write(f"  Successfully processed: {processed}\n")
                f.write(f"  Validation failures: {validation_failed}\n")
                f.write(f"  Processing failures: {failed}\n")
                f.write(f"  Other skipped: {skipped}\n")
                f.write(f"  Success rate: {processed / total_task * 100:.1f}%\n")

                # List validation failures
                if validation_failed > 0:
                    f.write(f"  Validation failed files:\n")
                    for file_path in batch_results['validation_failed'][task_type]:
                        f.write(f"    - {os.path.basename(file_path)}\n")

                f.write("\n")

        logger.info(f"Saved detailed task report to: {report_path}")

    def _generate_dual_batch_report(self, output_dir: str, total_files: int) -> None:
        """Generate a summary report for dual SQI processing."""
        try:
            report_path = os.path.join(output_dir, "dual_sqi_processing_report.txt")

            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("DUAL SQI PROCESSING REPORT\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Total files processed: {total_files}\n")
                f.write(f"Output directory: {output_dir}\n\n")

                with_sqi_dir = os.path.join(output_dir, "batch_with_SQI_filtering")
                without_sqi_dir = os.path.join(output_dir, "batch_no_SQI_filtering")

                if os.path.exists(with_sqi_dir):
                    with_sqi_files = sum([len(files) for _, _, files in os.walk(with_sqi_dir)])
                    f.write(f"Files with SQI filtering: {with_sqi_files}\n")

                if os.path.exists(without_sqi_dir):
                    without_sqi_files = sum([len(files) for _, _, files in os.walk(without_sqi_dir)])
                    f.write(f"Files without SQI filtering: {without_sqi_files}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("Processing complete!\n")
                f.write("=" * 80 + "\n")

            logger.info(f" Generated dual SQI processing report: {report_path}")

        except Exception as e:
            logger.error(f"Error generating dual batch report: {str(e)}")

    def _log_processing_summary(self, total_files: int, processed: List[str], skipped: List[str]) -> None:
        """Log comprehensive processing summary (original single batch version)."""
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-TASK PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {len(processed)}")
        logger.info(f"Skipped/Failed: {len(skipped)}")
        logger.info(f"Post-walking trimming: {self.post_walking_trim_seconds}s")

        if total_files > 0:
            success_rate = len(processed) / total_files * 100
            logger.info(f"Overall success rate: {success_rate:.1f}%")

        # Task-specific summary
        logger.info("\nBy Task Type:")
        batch_results = self.task_results['no_sqi_filtering']
        for task_type in sorted(batch_results['processed'].keys()):
            processed_count = len(batch_results['processed'][task_type])
            validation_failed = len(batch_results['validation_failed'][task_type])
            if processed_count + validation_failed > 0:
                logger.info(f"  {task_type}: {processed_count} processed, {validation_failed} validation failed")

        logger.info("=" * 60)


def main():
    """Command line interface for batch processing with dual SQI option and post-walking trimming."""
    parser = argparse.ArgumentParser(
        description="fNIRS Multi-Task Batch Processing Pipeline with Dual SQI Processing and Post-Walking Trimming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing input .txt files")
    parser.add_argument("output_dir", help="Directory for processed outputs")
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Sampling frequency in Hz")
    parser.add_argument("--sqi_thresh", type=float, default=2.5,
                        help="SQI threshold value for quality assessment (1-5 scale)")
    parser.add_argument("--post_walking_trim", type=float, default=3.0,
                        help="Seconds to trim after walking start event for quality control (default: 3.0)")
    parser.add_argument("--task_filter", nargs='+',
                        help="Filter to specific task types (e.g., --task_filter DT ST fTurn)")
    parser.add_argument("--strict_validation", action='store_true',
                        help="Enable strict task validation (reject files without sufficient events)")
    parser.add_argument("--dual_sqi", action='store_true', default=False,
                        help="Enable dual SQI processing")
    parser.add_argument("--single_batch", action='store_true',
                        help="Use single batch processing (original behavior)")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Validate trimming parameter
    if args.post_walking_trim < 0:
        print(" Error: Post-walking trim cannot be negative")
        return 1

    # Configure logging
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Run pipeline with post-walking trimming parameter
        processor = BatchProcessor(
            fs=args.fs,
            sqi_threshold=args.sqi_thresh,
            post_walking_trim_seconds=args.post_walking_trim
        )

        # Determine processing mode
        enable_dual_sqi = args.dual_sqi and not args.single_batch

        results = processor.run_two_pass_processing(
            args.input_dir,
            args.output_dir,
            task_filter=args.task_filter,
            strict_validation=args.strict_validation,
            enable_dual_sqi=enable_dual_sqi
        )

        # Print summary
        print("\n" + "=" * 80)
        if enable_dual_sqi:
            print("DUAL BATCH PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Total files found: {results['total_files']}")
            print(f"Post-walking trimming: {args.post_walking_trim}s")
            print(f"SQI threshold: {args.sqi_thresh}")
            print(f"Batch 1 (No SQI filtering): {len(results['batch1_processed'])} processed")
            print(f"Batch 2 (With SQI filtering): {len(results['batch2_processed'])} processed")

            if results.get('stats_batch1_raw') is not None:
                print(
                    f"RAW Statistics (no SQI filtering) saved to: {os.path.join(args.output_dir, 'all_subjects_statistics_RAW_no_SQI_filtering.csv')}")
            if results.get('stats_batch2_raw') is not None:
                print(
                    f"RAW Statistics (with SQI filtering) saved to: {os.path.join(args.output_dir, 'all_subjects_statistics_RAW_with_SQI_filtering.csv')}")

            print(f"Detailed report saved to: {os.path.join(args.output_dir, 'dual_sqi_processing_report.txt')}")
        else:
            print("SINGLE BATCH PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Total files found: {results['total_files']}")
            print(f"Post-walking trimming: {args.post_walking_trim}s")
            print(f"SQI threshold: {args.sqi_thresh}")
            print(f"Successfully processed: {len(results['processed_files'])}")

            if results.get('stats_raw') is not None:
                print(f"RAW statistics saved to: {os.path.join(args.output_dir, 'all_subjects_statistics_RAW.csv')}")
            if results.get('stats_zscore') is not None:
                print(
                    f"ZSCORE statistics saved to: {os.path.join(args.output_dir, 'all_subjects_statistics_ZSCORE.csv')}")

        print("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n Pipeline failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
