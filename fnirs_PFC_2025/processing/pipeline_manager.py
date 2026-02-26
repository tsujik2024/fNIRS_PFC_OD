# pipeline_manager.py
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.stats_collector import StatsCollector
from fnirs_PFC_2025.read.loaders import read_txt_file
import logging
import re

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Enhanced pipeline manager for multi-task fNIRS processing with dual SQI processing support.
    """

    def __init__(self, fs=50.0, sqi_threshold=2.0, post_walking_trim_seconds=3.0):
        """
        Initialize with processing parameters.

        Args:
            fs: Sampling frequency (Hz)
            sqi_threshold: Threshold for SQI quality assessment (1-5 scale, default 2.0)
            post_walking_trim_seconds: Seconds to trim after walking start event (default 3.0)
        """
        self.fs = fs
        self.sqi_threshold = sqi_threshold
        self.post_walking_trim_seconds = post_walking_trim_seconds

        # Create processors for both SQI filtering modes
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

        # Track processing results by task type and SQI filtering mode
        self.task_processing_log = {
            'no_sqi_filtering': {
                'processed': {},
                'validation_failed': {},
                'processing_failed': {},
                'skipped': {}
            },
            'with_sqi_filtering': {
                'processed': {},
                'validation_failed': {},
                'processing_failed': {},
                'skipped': {}
            }
        }

        logger.info(
            f"Initialized PipelineManager with DUAL SQI processing support "
            f"(fs={fs}, SQI threshold={sqi_threshold}, post-walking trim={post_walking_trim_seconds}s)")

    def find_input_files(self, input_base_dir: str, task_filter: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Find all .txt files in directory tree, organized by task type.

        Args:
            input_base_dir: Root directory to search
            task_filter: Optional list of task types to include

        Returns:
            Dictionary mapping task types to lists of file paths
        """
        all_files = []
        task_files = {}

        # Find all .txt files
        for root, _, files in os.walk(input_base_dir):
            for file in files:
                if file.endswith('.txt'):
                    all_files.append(os.path.join(root, file))

        # Organize by task type
        for file_path in sorted(all_files):
            task_type = self._determine_task_type(file_path)

            # Apply task filter if provided
            if task_filter and task_type not in task_filter:
                continue

            if task_type not in task_files:
                task_files[task_type] = []
            task_files[task_type].append(file_path)

        # Initialize tracking for all found task types
        for task_type in task_files.keys():
            for sqi_mode in ['no_sqi_filtering', 'with_sqi_filtering']:
                for category in self.task_processing_log[sqi_mode].keys():
                    self.task_processing_log[sqi_mode][category][task_type] = []

        # Log distribution
        logger.info(f"File distribution by task type:")
        for task_type, files in task_files.items():
            logger.info(f"  {task_type}: {len(files)} files")

        return task_files

    def _determine_task_type(self, file_path: str) -> str:
        basename = os.path.basename(file_path).upper()

        # Specific tasks first
        if "FTURN" in basename or "F_TURN" in basename:
            return "fTurn"
        if "LSHAPE" in basename or "L_SHAPE" in basename:
            return "LShape"
        if "FIGURE8" in basename or "FIG8" in basename:
            return "Figure8"
        if "OBSTACLE" in basename:
            return "Obstacle"
        if "NAVIGATION" in basename or re.search(r'\bNAV\b', basename):
            return "Navigation"

        # Strict DT/ST tokens: match whole tokens separated by non-letters or ends
        if re.search(r'(^|[^A-Z])DT([^A-Z]|$)', basename):
            return "DT"
        if re.search(r'(^|[^A-Z])ST([^A-Z]|$)', basename):
            return "ST"

        if "WALK" in basename:
            return "LongWalk"

        logger.warning(f"Could not determine task type from filename: {basename}")
        return "Unknown"

    def run_dual_sqi_processing(self,
                                input_base_dir: str,
                                output_base_dir: str,
                                task_filter: Optional[List[str]] = None,
                                strict_validation: bool = True) -> Optional[Dict]:
        """
        Execute complete dual SQI processing pipeline.
        Processes all files twice: once with all channels, once with SQI filtering.

        Args:
            input_base_dir: Root input directory
            output_base_dir: Root output directory
            task_filter: Optional list of task types to process
            strict_validation: Whether to enforce strict task validation

        Returns:
            Dictionary with processing results and statistics for both SQI modes
        """
        # Setup output directory
        os.makedirs(output_base_dir, exist_ok=True)

        # Find all input files organized by task type
        task_files = self.find_input_files(input_base_dir, task_filter)
        total_files = sum(len(files) for files in task_files.values())

        logger.info(f"Found {total_files} .txt files across {len(task_files)} task types")
        logger.info("Will process each file TWICE: once without SQI filtering, once with SQI filtering")

        if not task_files:
            logger.warning("No .txt files found in input directory")
            return None

        # BATCH 1: Process without SQI filtering
        logger.info("=" * 80)
        logger.info("BATCH 1: Processing WITHOUT SQI filtering (all channels included)")
        logger.info("=" * 80)

        batch1_processed, batch1_skipped = self.run_processing_pass(
            task_files, output_base_dir, input_base_dir,
            processor=self.processor_no_sqi,
            sqi_mode="no_sqi_filtering",
            subject_y_limits=None,
            strict_validation=strict_validation
        )

        # Calculate y-limits from batch 1 for consistency
        logger.info("--- Calculating consistent y-limits from batch 1 ---")
        subject_y_limits = None
        if batch1_processed:
            subject_y_limits = self.stats_collector.calculate_subject_y_limits(
                batch1_processed, output_base_dir, input_base_dir)

        # BATCH 2: Process with SQI filtering
        logger.info("=" * 80)
        logger.info("BATCH 2: Processing WITH SQI filtering (poor quality channels excluded)")
        logger.info("=" * 80)

        batch2_processed, batch2_skipped = self.run_processing_pass(
            task_files, output_base_dir, input_base_dir,
            processor=self.processor_with_sqi,
            sqi_mode="with_sqi_filtering",
            subject_y_limits=subject_y_limits,
            strict_validation=strict_validation
        )

        # Generate statistics for both batches
        logger.info("=" * 80)
        logger.info("Generating statistics for both batches")
        logger.info("=" * 80)

        stats_batch1 = None
        stats_batch2 = None

        if batch1_processed:
            logger.info("--- Generating statistics for batch 1 (no SQI filtering) ---")
            stats_batch1 = self.stats_collector.run_statistics(
                batch1_processed, input_base_dir, output_base_dir)

            if stats_batch1 is not None:
                self.stats_collector.create_summary_sheets(stats_batch1, output_base_dir, suffix="_no_SQI_filtering")

                stats_path = os.path.join(output_base_dir, 'all_subjects_statistics_no_SQI_filtering.csv')
                stats_batch1.to_csv(stats_path, index=False)
                logger.info(f"Saved batch 1 statistics to: {stats_path}")

        if batch2_processed:
            logger.info("--- Generating statistics for batch 2 (with SQI filtering) ---")
            stats_batch2 = self.stats_collector.run_statistics(
                batch2_processed, input_base_dir, output_base_dir)

            if stats_batch2 is not None:
                self.stats_collector.create_summary_sheets(stats_batch2, output_base_dir, suffix="_with_SQI_filtering")

                stats_path = os.path.join(output_base_dir, 'all_subjects_statistics_with_SQI_filtering.csv')
                stats_batch2.to_csv(stats_path, index=False)
                logger.info(f"Saved batch 2 statistics to: {stats_path}")

        # Generate comprehensive reports
        self.generate_dual_sqi_reports(stats_batch1, stats_batch2, output_base_dir)

        # Log comprehensive summary
        self._log_dual_pipeline_summary(task_files, batch1_processed, batch1_skipped,
                                        batch2_processed, batch2_skipped)

        return {
            'stats_no_sqi': stats_batch1,
            'stats_with_sqi': stats_batch2,
            'batch1_processed': batch1_processed,
            'batch1_skipped': batch1_skipped,
            'batch2_processed': batch2_processed,
            'batch2_skipped': batch2_skipped,
            'total_files': total_files,
            'task_processing_log': self.task_processing_log
        }

    def run_processing_pass(self,
                            task_files: Dict[str, List[str]],
                            output_base_dir: str,
                            input_base_dir: str,
                            processor: FileProcessor,
                            sqi_mode: str,
                            subject_y_limits: Optional[dict] = None,
                            strict_validation: bool = True) -> Tuple[List[str], List[str]]:
        """
        Execute a processing pass on all files organized by task type.

        Args:
            task_files: Dictionary mapping task types to file lists
            output_base_dir: Root output directory
            input_base_dir: Root input directory
            processor: FileProcessor instance (with or without SQI filtering)
            sqi_mode: Either "no_sqi_filtering" or "with_sqi_filtering"
            subject_y_limits: Optional y-limits for consistent plotting
            strict_validation: Whether to enforce strict task validation

        Returns:
            Tuple of (processed_files, skipped_files)
        """
        all_processed = []
        all_skipped = []

        # Process each task type separately
        for task_type, file_paths in task_files.items():
            sqi_status = "WITH SQI filtering" if processor.enable_sqi_filtering else "WITHOUT SQI filtering"
            logger.info(f"\n--- Processing {task_type} files ({len(file_paths)} files) - {sqi_status} ---")

            task_processed = []
            task_skipped = []

            for file_path in file_paths:
                try:
                    logger.info(f" Processing {task_type}: {os.path.basename(file_path)} ({sqi_status})")

                    result = processor.process_file(
                        file_path=file_path,
                        output_base_dir=output_base_dir,
                        input_base_dir=input_base_dir,
                        subject_y_limits=subject_y_limits,
                        read_file_func=read_txt_file
                    )

                    if result is not None:
                        task_processed.append(file_path)
                        all_processed.append(file_path)
                        self.task_processing_log[sqi_mode]['processed'][task_type].append(file_path)
                        logger.info(f" Completed: {os.path.basename(file_path)} ({sqi_status})")
                    else:
                        task_skipped.append(file_path)
                        all_skipped.append(file_path)
                        self.task_processing_log[sqi_mode]['skipped'][task_type].append(file_path)
                        logger.warning(f" Skipped: {os.path.basename(file_path)} ({sqi_status})")

                except Exception as e:
                    task_skipped.append(file_path)
                    all_skipped.append(file_path)

                    # Categorize the type of failure
                    error_msg = str(e).lower()
                    if any(phrase in error_msg for phrase in
                           ["task requirements not met", "requires at least", "validation failed"]):
                        self.task_processing_log[sqi_mode]['validation_failed'][task_type].append(file_path)
                        logger.error(f" Validation failed for {os.path.basename(file_path)} ({sqi_status}): {str(e)}")
                    else:
                        self.task_processing_log[sqi_mode]['processing_failed'][task_type].append(file_path)
                        logger.error(f" Processing failed for {os.path.basename(file_path)} ({sqi_status}): {str(e)}")

            # Log task-specific summary
            if file_paths:
                success_rate = len(task_processed) / len(file_paths) * 100
                logger.info(
                    f"{task_type} pass completed ({sqi_status}): {len(task_processed)}/{len(file_paths)} processed ({success_rate:.1f}%)")

        logger.info(f"Overall pass completed ({sqi_status}): {len(all_processed)} processed, {len(all_skipped)} skipped")
        return all_processed, all_skipped

    def generate_dual_sqi_reports(self,
                                  stats_no_sqi: Optional[pd.DataFrame],
                                  stats_with_sqi: Optional[pd.DataFrame],
                                  output_dir: str) -> None:
        """
        Generate all output reports and summaries for dual SQI processing.

        Args:
            stats_no_sqi: Statistics DataFrame for no SQI filtering batch (can be None)
            stats_with_sqi: Statistics DataFrame for SQI filtering batch (can be None)
            output_dir: Output directory for reports
        """
        # Generate detailed dual SQI processing report
        self._generate_dual_sqi_processing_report(output_dir)

        # Create combined comparison report if both batches have results
        if stats_no_sqi is not None and stats_with_sqi is not None:
            self._generate_sqi_comparison_report(stats_no_sqi, stats_with_sqi, output_dir)

    def _generate_dual_sqi_processing_report(self, output_dir: str) -> None:
        """Generate detailed dual SQI processing report."""
        report_path = os.path.join(output_dir, 'dual_sqi_processing_detailed_report.txt')

        with open(report_path, 'w') as f:
            f.write("fNIRS Dual SQI Processing Detailed Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Processing Parameters:\n")
            f.write(f"  Sampling frequency: {self.fs} Hz\n")
            f.write(f"  SQI threshold: {self.sqi_threshold}\n")
            f.write(f"  Post-walking trim: {self.post_walking_trim_seconds}s\n")
            f.write(f"  Processing approach: Dual batch (with and without SQI filtering)\n\n")

            # Calculate totals for both batches
            for sqi_mode, sqi_label in [('no_sqi_filtering', 'WITHOUT SQI Filtering'),
                                        ('with_sqi_filtering', 'WITH SQI Filtering')]:
                batch_log = self.task_processing_log[sqi_mode]

                total_processed = sum(len(files) for files in batch_log['processed'].values())
                total_validation_failed = sum(len(files) for files in batch_log['validation_failed'].values())
                total_processing_failed = sum(len(files) for files in batch_log['processing_failed'].values())
                total_skipped = sum(len(files) for files in batch_log['skipped'].values())
                total_files = total_processed + total_validation_failed + total_processing_failed + total_skipped

                f.write(f"BATCH: {sqi_label}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Overall Summary:\n")
                f.write(f"  Total files: {total_files}\n")
                f.write(f"  Successfully processed: {total_processed}\n")
                f.write(f"  Validation failures: {total_validation_failed}\n")
                f.write(f"  Processing failures: {total_processing_failed}\n")
                f.write(f"  Other skipped: {total_skipped}\n")
                if total_files > 0:
                    f.write(f"  Success rate: {total_processed / total_files * 100:.1f}%\n")
                f.write("\n")

                # Task-by-task breakdown
                for task_type in sorted(batch_log['processed'].keys()):
                    processed = len(batch_log['processed'][task_type])
                    validation_failed = len(batch_log['validation_failed'][task_type])
                    processing_failed = len(batch_log['processing_failed'][task_type])
                    skipped = len(batch_log['skipped'][task_type])
                    total_task = processed + validation_failed + processing_failed + skipped

                    if total_task == 0:
                        continue

                    f.write(f"{task_type} Task Analysis:\n")
                    f.write(f"  Total files: {total_task}\n")
                    f.write(f"  Successfully processed: {processed}\n")
                    f.write(f"  Validation failures: {validation_failed}\n")
                    f.write(f"  Processing failures: {processing_failed}\n")
                    f.write(f"  Other skipped: {skipped}\n")
                    f.write(f"  Success rate: {processed / total_task * 100:.1f}%\n\n")

                    # List validation failures
                    if validation_failed > 0:
                        f.write(f"  Files that failed validation (insufficient events):\n")
                        for file_path in batch_log['validation_failed'][task_type]:
                            f.write(f"    - {os.path.basename(file_path)}\n")
                        f.write("\n")

                    # List processing failures
                    if processing_failed > 0:
                        f.write(f"  Files that failed processing:\n")
                        for file_path in batch_log['processing_failed'][task_type]:
                            f.write(f"    - {os.path.basename(file_path)}\n")
                        f.write("\n")

                f.write("\n" + "=" * 70 + "\n\n")

        logger.info(f"Saved detailed dual SQI processing report to: {report_path}")

    def _generate_sqi_comparison_report(self,
                                        stats_no_sqi: pd.DataFrame,
                                        stats_with_sqi: pd.DataFrame,
                                        output_dir: str) -> None:
        """Generate comparison report between SQI filtered and non-filtered results."""
        try:
            comparison_path = os.path.join(output_dir, 'sqi_filtering_impact_analysis.txt')

            with open(comparison_path, 'w') as f:
                f.write("SQI Filtering Impact Analysis\n")
                f.write("=" * 50 + "\n\n")

                # Overall statistics comparison
                f.write("Overall Statistics Comparison:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Files processed without SQI filtering: {len(stats_no_sqi)}\n")
                f.write(f"Files processed with SQI filtering: {len(stats_with_sqi)}\n\n")

                # Task type comparison
                task_types_no_sqi = set(stats_no_sqi['TaskType'].unique())
                task_types_with_sqi = set(stats_with_sqi['TaskType'].unique())
                common_task_types = task_types_no_sqi & task_types_with_sqi

                if common_task_types:
                    f.write("Task Type Analysis:\n")
                    f.write("-" * 20 + "\n")

                    for task_type in sorted(common_task_types):
                        no_sqi_data = stats_no_sqi[stats_no_sqi['TaskType'] == task_type]
                        with_sqi_data = stats_with_sqi[stats_with_sqi['TaskType'] == task_type]

                        f.write(f"\n{task_type} Task:\n")
                        f.write(f"  Without SQI filtering: {len(no_sqi_data)} files\n")
                        f.write(f"  With SQI filtering: {len(with_sqi_data)} files\n")

                        if len(no_sqi_data) > 0 and len(with_sqi_data) > 0:
                            # Calculate mean differences
                            no_sqi_hbo_mean = no_sqi_data['Overall grand oxy Mean'].mean()
                            with_sqi_hbo_mean = with_sqi_data['Overall grand oxy Mean'].mean()
                            no_sqi_hbr_mean = no_sqi_data['Overall grand deoxy Mean'].mean()
                            with_sqi_hbr_mean = with_sqi_data['Overall grand deoxy Mean'].mean()

                            f.write(f"  HbO mean (no SQI): {no_sqi_hbo_mean:.6f}\n")
                            f.write(f"  HbO mean (with SQI): {with_sqi_hbo_mean:.6f}\n")
                            f.write(f"  HbO difference: {with_sqi_hbo_mean - no_sqi_hbo_mean:.6f}\n")
                            f.write(f"  HbR mean (no SQI): {no_sqi_hbr_mean:.6f}\n")
                            f.write(f"  HbR mean (with SQI): {with_sqi_hbr_mean:.6f}\n")
                            f.write(f"  HbR difference: {with_sqi_hbr_mean - no_sqi_hbr_mean:.6f}\n")

                f.write("\n" + "=" * 50 + "\n")
                f.write("Note: Positive differences indicate higher values with SQI filtering.\n")
                f.write("Large differences may indicate that poor quality channels were\n")
                f.write("significantly affecting the analysis results.\n")

            logger.info(f"Saved SQI filtering impact analysis to: {comparison_path}")

        except Exception as e:
            logger.error(f"Error generating SQI comparison report: {str(e)}")

    def _log_dual_pipeline_summary(self,
                                   task_files: Dict[str, List[str]],
                                   batch1_processed: List[str],
                                   batch1_skipped: List[str],
                                   batch2_processed: List[str],
                                   batch2_skipped: List[str]) -> None:
        """
        Log comprehensive dual pipeline summary with task-specific breakdowns.
        """
        total_files = sum(len(files) for files in task_files.values())

        logger.info("\n" + "=" * 80)
        logger.info("DUAL SQI PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total files found: {total_files}")
        logger.info(f"Processing approach: Dual batch (each file processed twice)")
        logger.info("")

        logger.info("BATCH 1 (No SQI Filtering - All Channels):")
        logger.info(f"  Successfully processed: {len(batch1_processed)}")
        logger.info(f"  Failed/Skipped: {len(batch1_skipped)}")
        if total_files > 0:
            success_rate1 = len(batch1_processed) / total_files * 100
            logger.info(f"  Success rate: {success_rate1:.1f}%")

        logger.info("")
        logger.info("BATCH 2 (With SQI Filtering - Poor Quality Channels Excluded):")
        logger.info(f"  Successfully processed: {len(batch2_processed)}")
        logger.info(f"  Failed/Skipped: {len(batch2_skipped)}")
        if total_files > 0:
            success_rate2 = len(batch2_processed) / total_files * 100
            logger.info(f"  Success rate: {success_rate2:.1f}%")

        logger.info("\nDetailed Breakdown by Task Type:")
        logger.info("-" * 60)

        for sqi_mode, sqi_label in [('no_sqi_filtering', 'No SQI Filter'), ('with_sqi_filtering', 'With SQI Filter')]:
            logger.info(f"\n{sqi_label}:")
            batch_log = self.task_processing_log[sqi_mode]

            for task_type in sorted(batch_log['processed'].keys()):
                processed_count = len(batch_log['processed'][task_type])
                validation_failed_count = len(batch_log['validation_failed'][task_type])
                processing_failed_count = len(batch_log['processing_failed'][task_type])

                if processed_count + validation_failed_count + processing_failed_count > 0:
                    logger.info(
                        f"  {task_type:12} | {processed_count:3} processed | "
                        f"Val.Fail: {validation_failed_count:2} | Proc.Fail: {processing_failed_count:2}")

        # Highlight validation issues for event-dependent tasks
        event_dependent_tasks = ['fTurn', 'LShape', 'Figure8', 'Obstacle', 'Navigation']
        validation_issues = []

        for sqi_mode in ['no_sqi_filtering', 'with_sqi_filtering']:
            for task_type in event_dependent_tasks:
                if task_type in self.task_processing_log[sqi_mode]['validation_failed']:
                    failed_count = len(self.task_processing_log[sqi_mode]['validation_failed'][task_type])
                    if failed_count > 0:
                        sqi_label = "No SQI" if sqi_mode == 'no_sqi_filtering' else "With SQI"
                        validation_issues.append(f"{task_type} ({sqi_label}): {failed_count}")

        if validation_issues:
            logger.info("\n  Event-dependent tasks with validation failures:")
            for issue in validation_issues:
                logger.info(f"   {issue}")
            logger.info("   (These files lack sufficient event markers)")

        logger.info("=" * 80)

    def run_pipeline(self,
                     input_base_dir: str,
                     output_base_dir: str,
                     task_filter: Optional[List[str]] = None,
                     strict_validation: bool = True,
                     enable_dual_sqi: bool = True) -> Optional[Dict]:
        """
        Execute complete processing pipeline with optional dual SQI processing.

        Args:
            input_base_dir: Root input directory
            output_base_dir: Root output directory
            task_filter: Optional list of task types to process
            strict_validation: Whether to enforce strict task validation
            enable_dual_sqi: If True, runs dual SQI processing. If False, runs single batch.

        Returns:
            Processing results dictionary (None if no files processed successfully)
        """
        if enable_dual_sqi:
            return self.run_dual_sqi_processing(input_base_dir, output_base_dir, task_filter, strict_validation)
        else:
            return self.run_single_sqi_processing(input_base_dir, output_base_dir, task_filter, strict_validation)

    def run_single_sqi_processing(self,
                                  input_base_dir: str,
                                  output_base_dir: str,
                                  task_filter: Optional[List[str]] = None,
                                  strict_validation: bool = True,
                                  enable_sqi_filtering: bool = False) -> Optional[Dict]:
        """
        Execute single batch processing with optional SQI filtering.

        Args:
            input_base_dir: Root input directory
            output_base_dir: Root output directory
            task_filter: Optional list of task types to process
            strict_validation: Whether to enforce strict task validation
            enable_sqi_filtering: If True, uses SQI filtering. If False, includes all channels.

        Returns:
            Single batch processing results dictionary
        """
        # Setup output directory
        os.makedirs(output_base_dir, exist_ok=True)

        # Find all input files organized by task type
        task_files = self.find_input_files(input_base_dir, task_filter)
        total_files = sum(len(files) for files in task_files.values())

        logger.info(f"Found {total_files} .txt files across {len(task_files)} task types")

        if not task_files:
            logger.warning("No .txt files found in input directory")
            return None

        # Choose processor based on SQI filtering setting
        processor = self.processor_with_sqi if enable_sqi_filtering else self.processor_no_sqi
        sqi_mode = "with_sqi_filtering" if enable_sqi_filtering else "no_sqi_filtering"

        # First pass - initial processing
        logger.info("--- First pass: Initial processing ---")
        processed_files, skipped_files = self.run_processing_pass(
            task_files, output_base_dir, input_base_dir,
            processor=processor,
            sqi_mode=sqi_mode,
            subject_y_limits=None,
            strict_validation=strict_validation
        )

        if not processed_files:
            logger.error("No files were successfully processed in first pass")
            return None

        # Calculate consistent y-limits per subject
        logger.info("--- Calculating consistent y-limits ---")
        subject_y_limits = self.stats_collector.calculate_subject_y_limits(
            processed_files, output_base_dir, input_base_dir)

        # Second pass - processing with consistent y-limits
        logger.info("--- Second pass: Processing with y-limits ---")
        final_processed, final_skipped = self.run_processing_pass(
            task_files, output_base_dir, input_base_dir,
            processor=processor,
            sqi_mode=sqi_mode,
            subject_y_limits=subject_y_limits,
            strict_validation=strict_validation
        )

        if not final_processed:
            logger.error("No files were successfully processed in second pass")
            return None

        # Generate statistics and reports
        logger.info("--- Generating reports ---")
        stats_df = self.stats_collector.run_statistics(
            final_processed, input_base_dir, output_base_dir)

        if stats_df is not None:
            # Create summary sheets
            suffix = "_with_SQI_filtering" if enable_sqi_filtering else "_no_SQI_filtering"
            self.stats_collector.create_summary_sheets(stats_df, output_base_dir, suffix=suffix)

            # Save stats
            stats_filename = f'all_subjects_statistics{suffix}.csv'
            stats_path = os.path.join(output_base_dir, stats_filename)
            stats_df.to_csv(stats_path, index=False)
            logger.info(f"Saved statistics to: {stats_path}")

        # Generate single batch report
        self._generate_single_processing_report(output_base_dir, total_files, sqi_mode, enable_sqi_filtering)

        # Log summary
        self._log_single_pipeline_summary(task_files, final_processed, final_skipped, enable_sqi_filtering)

        return {
            'stats': stats_df,
            'processed_files': final_processed,
            'skipped_files': final_skipped,
            'total_files': total_files,
            'task_results': self.task_processing_log[sqi_mode]
        }

    def _generate_single_processing_report(self, output_dir: str, total_files: int, sqi_mode: str,
                                           enable_sqi_filtering: bool) -> None:
        """Generate single batch processing report."""
        report_path = os.path.join(output_dir, 'single_batch_processing_report.txt')
        batch_log = self.task_processing_log[sqi_mode]

        with open(report_path, 'w') as f:
            f.write("fNIRS Single Batch Processing Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total files found: {total_files}\n")
            f.write(f"Sampling frequency: {self.fs} Hz\n")
            f.write(f"SQI threshold: {self.sqi_threshold}\n")
            f.write(f"Post-walking trim: {self.post_walking_trim_seconds}s\n")
            f.write(f"SQI filtering enabled: {'Yes' if enable_sqi_filtering else 'No'}\n\n")

            # Task-by-task breakdown
            for task_type in sorted(batch_log['processed'].keys()):
                processed = len(batch_log['processed'][task_type])
                skipped = len(batch_log['skipped'][task_type])
                processing_failed = len(batch_log['processing_failed'][task_type])
                validation_failed = len(batch_log['validation_failed'][task_type])
                total_task = processed + skipped + processing_failed + validation_failed

                if total_task == 0:
                    continue

                f.write(f"{task_type} Task Results:\n")
                f.write(f"  Total files: {total_task}\n")
                f.write(f"  Successfully processed: {processed}\n")
                f.write(f"  Validation failures: {validation_failed}\n")
                f.write(f"  Processing failures: {processing_failed}\n")
                f.write(f"  Other skipped: {skipped}\n")
                f.write(f"  Success rate: {processed / total_task * 100:.1f}%\n")

                # List validation failures
                if validation_failed > 0:
                    f.write(f"  Validation failed files:\n")
                    for file_path in batch_log['validation_failed'][task_type]:
                        f.write(f"    - {os.path.basename(file_path)}\n")

                f.write("\n")

        logger.info(f"Saved single batch processing report to: {report_path}")

    def _log_single_pipeline_summary(self, task_files: Dict[str, List[str]], processed: List[str], skipped: List[str],
                                     enable_sqi_filtering: bool) -> None:
        """Log single pipeline summary."""
        total_files = sum(len(files) for files in task_files.values())
        sqi_status = "WITH SQI filtering" if enable_sqi_filtering else "WITHOUT SQI filtering"

        logger.info("\n" + "=" * 70)
        logger.info(f"SINGLE BATCH PIPELINE SUMMARY ({sqi_status})")
        logger.info("=" * 70)
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {len(processed)}")
        logger.info(f"Failed/Skipped: {len(skipped)}")

        if total_files > 0:
            success_rate = len(processed) / total_files * 100
            logger.info(f"Overall success rate: {success_rate:.1f}%")

        logger.info("=" * 70)

    def get_task_summary(self, sqi_mode: str = 'no_sqi_filtering') -> Dict[str, Dict[str, int]]:
        """
        Get summary of processing results by task type for specified SQI mode.

        Args:
            sqi_mode: Either 'no_sqi_filtering' or 'with_sqi_filtering'

        Returns:
            Dictionary with task types and their processing statistics
        """
        summary = {}
        batch_log = self.task_processing_log[sqi_mode]

        for task_type in batch_log['processed'].keys():
            summary[task_type] = {
                'processed': len(batch_log['processed'][task_type]),
                'validation_failed': len(batch_log['validation_failed'][task_type]),
                'processing_failed': len(batch_log['processing_failed'][task_type]),
                'skipped': len(batch_log['skipped'][task_type])
            }

            total = sum(summary[task_type].values())
            if total > 0:
                summary[task_type]['success_rate'] = summary[task_type]['processed'] / total * 100
            else:
                summary[task_type]['success_rate'] = 0.0

        return summary

    def get_validation_failures(self, sqi_mode: str = 'no_sqi_filtering') -> Dict[str, List[str]]:
        """
        Get files that failed validation by task type for specified SQI mode.

        Args:
            sqi_mode: Either 'no_sqi_filtering' or 'with_sqi_filtering'

        Returns:
            Dictionary mapping task types to lists of files that failed validation
        """
        return {task_type: files.copy()
                for task_type, files in self.task_processing_log[sqi_mode]['validation_failed'].items()
                if files}
