import os
import argparse
import logging
from typing import List, Optional, Tuple, Dict
from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.stats_collector import StatsCollector
from fnirs_PFC_2025.read.loaders import read_txt_file

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Enhanced batch processor for multi-task fNIRS files with dual CV processing and post-walking trimming."""

    def __init__(self, fs: float = 50.0, cv_threshold: float = 50.0,
                 post_walking_trim_seconds: float = 3.0):
        """
        Initialize batch processor.

        Args:
            fs: Sampling frequency in Hz
            cv_threshold: CV threshold value for quality assessment (default 50%)
            post_walking_trim_seconds: Seconds to trim after walking start event (default 3.0)
        """
        self.fs = fs
        self.cv_threshold = cv_threshold
        self.post_walking_trim_seconds = post_walking_trim_seconds

        # Create two processors: one with CV filtering, one without - BOTH with trimming
        self.processor_no_cv = FileProcessor(
            fs=fs,
            cv_threshold=cv_threshold,
            enable_cv_filtering=False,
            post_walking_trim_seconds=post_walking_trim_seconds
        )
        self.processor_with_cv = FileProcessor(
            fs=fs,
            cv_threshold=cv_threshold,
            enable_cv_filtering=True,
            post_walking_trim_seconds=post_walking_trim_seconds
        )
        self.stats_collector = StatsCollector(fs=fs)

        # Track task-specific processing results for both batches
        self.task_results = {
            'no_cv_filtering': {
                'processed': {},
                'skipped': {},
                'failed': {},
                'validation_failed': {}
            },
            'with_cv_filtering': {
                'processed': {},
                'skipped': {},
                'failed': {},
                'validation_failed': {}
            }
        }

        logger.info(f"Initialized BatchProcessor with DUAL CV processing "
                    f"(fs={fs}, CV threshold={cv_threshold}%, post-walking trim={post_walking_trim_seconds}s)")

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

    def run_dual_cv_processing(self,
                               input_dir: str,
                               output_dir: str,
                               task_filter: Optional[List[str]] = None,
                               strict_validation: bool = True) -> dict:
        """
        Execute complete dual CV processing pipeline with separate output directories.
        Processes all files twice: once with all channels, once with CV filtering.
        """
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create separate output directories for each batch to prevent file conflicts
        output_dir_no_cv = os.path.join(output_dir, "batch_no_CV_filtering")
        output_dir_with_cv = os.path.join(output_dir, "batch_with_CV_filtering")

        os.makedirs(output_dir_no_cv, exist_ok=True)
        os.makedirs(output_dir_with_cv, exist_ok=True)

        # Find files organized by task type
        task_files = self.find_input_files(input_dir, task_filter)
        total_files = sum(len(files) for files in task_files.values())

        logger.info(f"Found {total_files} input files across {len(task_files)} task types")
        logger.info("Will process each file TWICE: once without CV filtering, once with CV filtering")
        logger.info(f"Post-walking trimming: {self.post_walking_trim_seconds}s applied to both batches")
        logger.info(f"Output directories:")
        logger.info(f"  Batch 1 (No CV): {output_dir_no_cv}")
        logger.info(f"  Batch 2 (With CV): {output_dir_with_cv}")

        if not task_files:
            raise ValueError("No .txt files found in input directory")

        # Initialize task results tracking
        for batch_type in ['no_cv_filtering', 'with_cv_filtering']:
            for task_type in task_files.keys():
                for result_type in ['processed', 'skipped', 'failed', 'validation_failed']:
                    self.task_results[batch_type][result_type][task_type] = []

        # BATCH 1: Process without CV filtering
        logger.info("=" * 80)
        logger.info("BATCH 1: Processing WITHOUT CV filtering (all channels included)")
        logger.info(f"Output directory: {output_dir_no_cv}")
        logger.info("=" * 80)

        batch1_processed, batch1_skipped = self._process_files_batch(
            task_files, input_dir, output_dir_no_cv,  # Use separate directory
            processor=self.processor_no_cv,
            batch_name="no_cv_filtering",
            subject_y_limits=None,
            strict_validation=strict_validation
        )

        # Calculate y-limits from batch 1 for consistency (but don't apply to batch 2 to avoid conflicts)
        logger.info("--- Calculating y-limits from batch 1 for reference ---")
        subject_y_limits = None
        if batch1_processed:
            # Calculate y-limits but don't use them for batch 2 to ensure independence
            subject_y_limits = self.stats_collector.calculate_subject_y_limits(
                batch1_processed, output_dir_no_cv, input_dir)
            logger.info(f"Calculated y-limits for {len(subject_y_limits) if subject_y_limits else 0} subjects")

        # BATCH 2: Process with CV filtering (independent processing)
        logger.info("=" * 80)
        logger.info("BATCH 2: Processing WITH CV filtering (poor quality channels excluded)")
        logger.info(f"Output directory: {output_dir_with_cv}")
        logger.info("=" * 80)

        batch2_processed, batch2_skipped = self._process_files_batch(
            task_files, input_dir, output_dir_with_cv,  # Use separate directory
            processor=self.processor_with_cv,
            batch_name="with_cv_filtering",
            subject_y_limits=None,  # Use None to ensure independent processing
            strict_validation=strict_validation
        )

        # Generate statistics for both batches
        logger.info("=" * 80)
        logger.info("Generating statistics for both batches")
        logger.info("=" * 80)

        stats_batch1 = None
        stats_batch2 = None

        if batch1_processed:
            logger.info("--- Generating statistics for batch 1 (no CV filtering) ---")
            stats_batch1 = self.stats_collector.run_statistics(
                processed_files=batch1_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir_no_cv  # Use batch-specific directory
            )

            if stats_batch1 is not None:
                # Create summary sheets for batch 1 in its own directory
                self.stats_collector.create_summary_sheets(stats_batch1, output_dir_no_cv, suffix="_no_CV_filtering")

                # Save batch 1 stats in main output directory for easy comparison
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_no_CV_filtering.csv')
                stats_batch1.to_csv(stats_path, index=False)
                logger.info(f"Saved batch 1 statistics to: {stats_path}")

                # Also save in batch-specific directory
                batch_stats_path = os.path.join(output_dir_no_cv, 'all_subjects_statistics_no_CV_filtering.csv')
                stats_batch1.to_csv(batch_stats_path, index=False)

        if batch2_processed:
            logger.info("--- Generating statistics for batch 2 (with CV filtering) ---")
            stats_batch2 = self.stats_collector.run_statistics(
                processed_files=batch2_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir_with_cv  # Use batch-specific directory
            )

            if stats_batch2 is not None:
                # Create summary sheets for batch 2 in its own directory
                self.stats_collector.create_summary_sheets(stats_batch2, output_dir_with_cv,
                                                           suffix="_with_CV_filtering")

                # Save batch 2 stats in main output directory for easy comparison
                stats_path = os.path.join(output_dir, 'all_subjects_statistics_with_CV_filtering.csv')
                stats_batch2.to_csv(stats_path, index=False)
                logger.info(f"Saved batch 2 statistics to: {stats_path}")

                # Also save in batch-specific directory
                batch_stats_path = os.path.join(output_dir_with_cv, 'all_subjects_statistics_with_CV_filtering.csv')
                stats_batch2.to_csv(batch_stats_path, index=False)

        # Generate comprehensive reports in main output directory
        self._generate_dual_batch_report(output_dir, total_files)

        # Log final summary
        self._log_dual_processing_summary(total_files, batch1_processed, batch1_skipped,
                                          batch2_processed, batch2_skipped)

        return {
            'stats_no_cv': stats_batch1,
            'stats_with_cv': stats_batch2,
            'batch1_processed': batch1_processed,
            'batch1_skipped': batch1_skipped,
            'batch2_processed': batch2_processed,
            'batch2_skipped': batch2_skipped,
            'total_files': total_files,
            'task_results': self.task_results,
            'output_dir_no_cv': output_dir_no_cv,
            'output_dir_with_cv': output_dir_with_cv
        }

    def _process_files_batch(self,
                             task_files: Dict[str, List[str]],
                             input_dir: str,
                             output_dir: str,
                             processor: FileProcessor,
                             batch_name: str,
                             subject_y_limits: Optional[dict] = None,
                             strict_validation: bool = True) -> Tuple[List[str], List[str]]:
        """Process files for a single batch (either with or without CV filtering)."""
        all_processed = []
        all_skipped = []

        for task_type, file_paths in task_files.items():
            logger.info(f"\n--- Processing {task_type} files ({len(file_paths)} files) - {batch_name} ---")

            task_processed = []
            task_skipped = []

            for file_path in file_paths:
                try:
                    cv_status = "WITH CV filtering" if processor.enable_cv_filtering else "WITHOUT CV filtering"
                    logger.info(f" Processing {task_type}: {os.path.basename(file_path)} ({cv_status})")

                    # Process the file
                    result = processor.process_file(
                        file_path=file_path,
                        output_base_dir=output_dir,
                        input_base_dir=input_dir,
                        subject_y_limits=subject_y_limits,
                        read_file_func=read_txt_file,
                        baseline_duration=20.0
                    )

                    if result is not None:
                        task_processed.append(file_path)
                        all_processed.append(file_path)
                        self.task_results[batch_name]['processed'][task_type].append(file_path)
                        logger.info(f" Successfully processed: {os.path.basename(file_path)} ({cv_status})")
                    else:
                        task_skipped.append(file_path)
                        all_skipped.append(file_path)
                        self.task_results[batch_name]['skipped'][task_type].append(file_path)
                        logger.warning(f" Skipped file: {os.path.basename(file_path)} ({cv_status})")

                except Exception as e:
                    task_skipped.append(file_path)
                    all_skipped.append(file_path)

                    # Categorize the failure type
                    if "Task requirements not met" in str(e) or "requires at least" in str(e):
                        self.task_results[batch_name]['validation_failed'][task_type].append(file_path)
                        logger.error(f" Validation failed for {os.path.basename(file_path)} ({cv_status}): {str(e)}")
                    else:
                        self.task_results[batch_name]['failed'][task_type].append(file_path)
                        logger.error(f" Processing failed for {os.path.basename(file_path)} ({cv_status}): {str(e)}")

            # Log task summary
            success_rate = len(task_processed) / len(file_paths) * 100 if file_paths else 0
            logger.info(
                f"{task_type} summary ({batch_name}): {len(task_processed)}/{len(file_paths)} processed ({success_rate:.1f}%)")

        logger.info(f"Overall batch ({batch_name}): {len(all_processed)} processed, {len(all_skipped)} skipped")
        return all_processed, all_skipped

    def _generate_dual_batch_report(self, output_dir: str, total_files: int) -> None:
        """Generate detailed dual batch processing report."""
        report_path = os.path.join(output_dir, 'dual_batch_processing_report.txt')

        with open(report_path, 'w') as f:
            f.write("fNIRS Dual Batch Processing Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total files found: {total_files}\n")
            f.write(f"Sampling frequency: {self.fs} Hz\n")
            f.write(f"CV threshold: {self.cv_threshold}%\n")
            f.write(f"Post-walking trimming: {self.post_walking_trim_seconds}s\n")
            f.write("Processing approach: Dual batch (with and without CV filtering)\n\n")

            # Summary for both batches
            for batch_name, batch_label in [('no_cv_filtering', 'WITHOUT CV Filtering'),
                                            ('with_cv_filtering', 'WITH CV Filtering')]:
                f.write(f"BATCH: {batch_label}\n")
                f.write("-" * 40 + "\n")

                batch_results = self.task_results[batch_name]

                # Overall batch statistics
                total_processed = sum(len(files) for files in batch_results['processed'].values())
                total_validation_failed = sum(len(files) for files in batch_results['validation_failed'].values())
                total_processing_failed = sum(len(files) for files in batch_results['failed'].values())
                total_skipped = sum(len(files) for files in batch_results['skipped'].values())

                f.write(f"Overall Results:\n")
                f.write(f"  Successfully processed: {total_processed}\n")
                f.write(f"  Validation failures: {total_validation_failed}\n")
                f.write(f"  Processing failures: {total_processing_failed}\n")
                f.write(f"  Other skipped: {total_skipped}\n")
                if total_files > 0:
                    success_rate = total_processed / total_files * 100
                    f.write(f"  Success rate: {success_rate:.1f}%\n")
                f.write("\n")

                # Task-by-task breakdown
                for task_type in sorted(batch_results['processed'].keys()):
                    processed = len(batch_results['processed'][task_type])
                    validation_failed = len(batch_results['validation_failed'][task_type])
                    processing_failed = len(batch_results['failed'][task_type])
                    skipped = len(batch_results['skipped'][task_type])
                    total_task = processed + validation_failed + processing_failed + skipped

                    if total_task == 0:
                        continue

                    f.write(f"{task_type} Task Results:\n")
                    f.write(f"  Total files: {total_task}\n")
                    f.write(f"  Successfully processed: {processed}\n")
                    f.write(f"  Validation failures: {validation_failed}\n")
                    f.write(f"  Processing failures: {processing_failed}\n")
                    f.write(f"  Other skipped: {skipped}\n")
                    f.write(f"  Success rate: {processed / total_task * 100:.1f}%\n")

                    # List validation failures (these are the important ones)
                    if validation_failed > 0:
                        f.write(f"  Validation failed files:\n")
                        for file_path in batch_results['validation_failed'][task_type]:
                            f.write(f"    - {os.path.basename(file_path)}\n")

                    f.write("\n")

                f.write("\n" + "=" * 60 + "\n\n")

        logger.info(f"Saved detailed dual batch report to: {report_path}")

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

        logger.info("BATCH 1 (No CV Filtering - All Channels):")
        logger.info(f"  Successfully processed: {len(batch1_processed)}")
        logger.info(f"  Skipped/Failed: {len(batch1_skipped)}")
        if total_files > 0:
            success_rate1 = len(batch1_processed) / total_files * 100
            logger.info(f"  Success rate: {success_rate1:.1f}%")

        logger.info("")
        logger.info("BATCH 2 (With CV Filtering - Poor Quality Channels Excluded):")
        logger.info(f"  Successfully processed: {len(batch2_processed)}")
        logger.info(f"  Skipped/Failed: {len(batch2_skipped)}")
        if total_files > 0:
            success_rate2 = len(batch2_processed) / total_files * 100
            logger.info(f"  Success rate: {success_rate2:.1f}%")

        # Task-specific summary for both batches
        logger.info("\nDetailed Breakdown by Task Type:")
        logger.info("-" * 60)

        for batch_name, batch_label in [('no_cv_filtering', 'No CV Filter'), ('with_cv_filtering', 'With CV Filter')]:
            logger.info(f"\n{batch_label}:")
            batch_results = self.task_results[batch_name]

            for task_type in sorted(batch_results['processed'].keys()):
                processed_count = len(batch_results['processed'][task_type])
                validation_failed = len(batch_results['validation_failed'][task_type])
                if processed_count + validation_failed > 0:
                    logger.info(f"  {task_type}: {processed_count} processed, {validation_failed} validation failed")

        logger.info("=" * 80)

    # Keep existing run_two_pass_processing method for backward compatibility
    def run_two_pass_processing(self,
                                input_dir: str,
                                output_dir: str,
                                task_filter: Optional[List[str]] = None,
                                strict_validation: bool = True,
                                enable_dual_cv: bool = True) -> dict:
        """
        Execute processing pipeline with optional dual CV processing.

        Args:
            enable_dual_cv: If True, runs dual batch processing. If False, runs original single batch.
        """
        if enable_dual_cv:
            logger.info(" Running DUAL CV processing mode")
            return self.run_dual_cv_processing(input_dir, output_dir, task_filter, strict_validation)
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

        # Find files organized by task type
        task_files = self.find_input_files(input_dir, task_filter)
        total_files = sum(len(files) for files in task_files.values())

        logger.info(f"Found {total_files} input files across {len(task_files)} task types")

        if not task_files:
            raise ValueError("No .txt files found in input directory")

        # Initialize task results tracking for single batch
        for task_type in task_files.keys():
            self.task_results['no_cv_filtering']['processed'][task_type] = []
            self.task_results['no_cv_filtering']['skipped'][task_type] = []
            self.task_results['no_cv_filtering']['failed'][task_type] = []
            self.task_results['no_cv_filtering']['validation_failed'][task_type] = []

        # First pass - initial processing
        logger.info("--- First pass: Initial processing ---")
        processed_files, skipped_files = self._process_files_batch(
            task_files, input_dir, output_dir,
            processor=self.processor_no_cv,
            batch_name="no_cv_filtering",
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
            processor=self.processor_no_cv,
            batch_name="no_cv_filtering",
            subject_y_limits=subject_y_limits,
            strict_validation=strict_validation
        )

        # Generate statistics
        logger.info("--- Generating statistics ---")
        stats = None
        if final_processed:
            stats = self.stats_collector.run_statistics(
                processed_files=final_processed,
                input_base_dir=input_dir,
                output_base_dir=output_dir
            )

            # Create summary sheets
            logger.info("--- Creating summary sheets ---")
            self.stats_collector.create_summary_sheets(stats, output_dir)

            # Save complete stats
            stats_path = os.path.join(output_dir, 'all_subjects_statistics.csv')
            stats.to_csv(stats_path, index=False)
            logger.info(f"Saved combined statistics to: {stats_path}")

        # Generate detailed task-specific report
        self._generate_task_report(output_dir, total_files)

        # Log final summary
        self._log_processing_summary(total_files, final_processed, final_skipped)

        return {
            'stats': stats,
            'processed_files': final_processed,
            'skipped_files': final_skipped,
            'total_files': total_files,
            'task_results': self.task_results['no_cv_filtering']
        }

    def _generate_task_report(self, output_dir: str, total_files: int) -> None:
        """Generate detailed task-specific processing report (original single batch version)."""
        report_path = os.path.join(output_dir, 'task_processing_report.txt')
        batch_results = self.task_results['no_cv_filtering']

        with open(report_path, 'w') as f:
            f.write("fNIRS Multi-Task Processing Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total files found: {total_files}\n")
            f.write(f"Sampling frequency: {self.fs} Hz\n")
            f.write(f"CV threshold: {self.cv_threshold}%\n")
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
        batch_results = self.task_results['no_cv_filtering']
        for task_type in sorted(batch_results['processed'].keys()):
            processed_count = len(batch_results['processed'][task_type])
            validation_failed = len(batch_results['validation_failed'][task_type])
            if processed_count + validation_failed > 0:
                logger.info(f"  {task_type}: {processed_count} processed, {validation_failed} validation failed")

        logger.info("=" * 60)


def main():
    """Command line interface for batch processing with dual CV option and post-walking trimming."""
    parser = argparse.ArgumentParser(
        description="fNIRS Multi-Task Batch Processing Pipeline with Dual CV Processing and Post-Walking Trimming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing input .txt files")
    parser.add_argument("output_dir", help="Directory for processed outputs")
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Sampling frequency in Hz")
    parser.add_argument("--cv_thresh", type=float, default=50.0,
                        help="CV threshold value for quality assessment")
    parser.add_argument("--post_walking_trim", type=float, default=3.0,
                        help="Seconds to trim after walking start event for quality control (default: 3.0)")
    parser.add_argument("--task_filter", nargs='+',
                        help="Filter to specific task types (e.g., --task_filter DT ST fTurn)")
    parser.add_argument("--strict_validation", action='store_true',
                        help="Enable strict task validation (reject files without sufficient events)")
    parser.add_argument("--dual_cv", action='store_true', default=True,
                        help="Enable dual CV processing (default: True)")
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
            cv_threshold=args.cv_thresh,
            post_walking_trim_seconds=args.post_walking_trim
        )

        # Determine processing mode
        enable_dual_cv = args.dual_cv and not args.single_batch

        results = processor.run_two_pass_processing(
            args.input_dir,
            args.output_dir,
            task_filter=args.task_filter,
            strict_validation=args.strict_validation,
            enable_dual_cv=enable_dual_cv
        )

        # Print summary
        print("\n" + "=" * 80)
        if enable_dual_cv:
            print("DUAL BATCH PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Total files found: {results['total_files']}")
            print(f"Post-walking trimming: {args.post_walking_trim}s")
            print(f"Batch 1 (No CV filtering): {len(results['batch1_processed'])} processed")
            print(f"Batch 2 (With CV filtering): {len(results['batch2_processed'])} processed")

            if results['stats_no_cv'] is not None:
                print(
                    f"Statistics (no CV filtering) saved to: {os.path.join(args.output_dir, 'all_subjects_statistics_no_CV_filtering.csv')}")
            if results['stats_with_cv'] is not None:
                print(
                    f"Statistics (with CV filtering) saved to: {os.path.join(args.output_dir, 'all_subjects_statistics_with_CV_filtering.csv')}")

            print(f"Detailed report saved to: {os.path.join(args.output_dir, 'dual_batch_processing_report.txt')}")
        else:
            print("SINGLE BATCH PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Total files found: {results['total_files']}")
            print(f"Post-walking trimming: {args.post_walking_trim}s")
            print(f"Successfully processed: {len(results['processed_files'])}")

            if results['stats'] is not None:
                print(f"Statistics saved to: {os.path.join(args.output_dir, 'all_subjects_statistics.csv')}")

        print("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n Pipeline failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())