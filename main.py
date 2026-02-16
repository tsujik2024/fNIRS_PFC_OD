"""
fNIRS Multi-Task Processing Pipeline with Dual CV Processing and Post-Walking Trimming - Command Line Interface
"""

import argparse
import logging
import os
import sys
from fnirs_PFC_2025.processing.batch_processor import BatchProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process fNIRS data through the complete multi-task pipeline with dual CV processing and post-walking trimming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Process with dual CV batches and default 3s post-walking trimming
  python main.py data/ results/ --dual_cv

  # Process with custom post-walking trimming
  python main.py data/ results/ --dual_cv --post_walking_trim 5.0

  # Process with no post-walking trimming
  python main.py data/ results/ --dual_cv --post_walking_trim 0

  # Process only with CV filtering enabled and custom trimming
  python main.py data/ results/ --single_batch --cv_filtering_only --post_walking_trim 2.0

  # Process only specific task types with dual CV and trimming
  python main.py data/ results/ --task_filter DT ST fTurn LShape --dual_cv --post_walking_trim 3.0

  # Process with custom parameters and dual CV
  python main.py data/ results/ --fs 25.0 --cv_thresh 40.0 --dual_cv

  # Original single batch processing (backward compatibility)
  python main.py data/ results/ --single_batch
        """
    )

    # Required arguments
    parser.add_argument("input_dir", help="Directory containing .txt fNIRS files")
    parser.add_argument("output_dir", help="Directory for processed outputs")

    # Processing parameters
    parser.add_argument("--fs", type=float, default=50.0, help="Sampling rate in Hz")
    parser.add_argument("--cv_thresh", type=float, default=50.0,
                        help="Coefficient of Variation threshold for quality assessment (%%)")

    # NEW: Post-walking trimming parameter
    parser.add_argument("--post_walking_trim", type=float, default=3.0,
                        help="Seconds to trim after walking start event for quality control (default: 3.0)")

    # Multi-task options
    parser.add_argument("--task_filter", nargs='+', metavar='TASK',
                        help="Filter to specific task types (e.g., DT ST fTurn LShape Figure8 Obstacle Navigation)")
    parser.add_argument("--strict_validation", action='store_true',
                        help="Enable strict validation (reject files without sufficient event markers)")
    parser.add_argument("--list_tasks", action='store_true',
                        help="List all detected task types and exit")

    # CV Processing options
    parser.add_argument("--dual_cv", action='store_true', default=True,
                        help="Enable dual CV processing (process each file twice: with and without CV filtering)")
    parser.add_argument("--single_batch", action='store_true',
                        help="Use single batch processing instead of dual CV processing")
    parser.add_argument("--cv_filtering_only", action='store_true',
                        help="When using single batch, only process with CV filtering enabled")

    # Logging options
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity level")
    parser.add_argument("--quiet", "-q", action='store_true',
                        help="Suppress console output (log to file only)")
    parser.add_argument("--verbose", "-v", action='store_true',
                        help="Enable verbose output (show detailed processing steps)")

    args = parser.parse_args()

    # Validate arguments
    if args.dual_cv and args.single_batch:
        print(" Error: Cannot use both --dual_cv and --single_batch options")
        return 1

    if args.cv_filtering_only and not args.single_batch:
        print(" Error: --cv_filtering_only can only be used with --single_batch")
        return 1

    # Validate trimming parameter
    if args.post_walking_trim < 0:
        print(" Error: Post-walking trim cannot be negative")
        return 1

    # Set dual_cv to False if single_batch is requested
    if args.single_batch:
        args.dual_cv = False

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f" Error: Input directory '{args.input_dir}' does not exist")
        return 1

    if not os.path.isdir(args.input_dir):
        print(f" Error: '{args.input_dir}' is not a directory")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(args.output_dir, "fnirs_processing.log")
    handlers = [logging.FileHandler(log_file)]
    if not args.quiet:
        handlers.append(logging.StreamHandler())

    log_level = "DEBUG" if args.verbose else args.log_level
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger(__name__)

    # Initialize processor with trimming parameter
    try:
        processor = BatchProcessor(
            fs=args.fs,
            cv_threshold=args.cv_thresh,
            post_walking_trim_seconds=args.post_walking_trim  # NEW PARAMETER
        )
    except Exception as e:
        print(f" Error initializing processor: {str(e)}")
        return 1

    # Handle --list_tasks option (with trimming info)
    if args.list_tasks:
        print("🔍 Scanning for task types...")
        try:
            task_files = processor.find_input_files(args.input_dir)
            if task_files:
                print("\n Detected task types:")
                total_files = 0
                for task_type, files in sorted(task_files.items()):
                    print(f"  {task_type}: {len(files)} files")
                    total_files += len(files)
                print(f"\nTotal: {total_files} files")
                print(f"Post-walking trimming: {args.post_walking_trim}s")  # NEW LINE

                if args.dual_cv:
                    print(f"\n With dual CV processing, each file will be processed twice:")
                    print(f"  - Total processing operations: {total_files * 2}")
                    print(f"  - Batch 1: All channels included")
                    print(f"  - Batch 2: Poor quality channels (CV > {args.cv_thresh}%) excluded")
                    print(f"  - Both batches: {args.post_walking_trim}s trimmed after walking start")
            else:
                print(" No .txt files found in input directory")
            return 0
        except Exception as e:
            print(f" Error scanning files: {str(e)}")
            return 1

    # Determine processing method
    processing_method = "DUAL CV PROCESSING" if args.dual_cv else "SINGLE BATCH PROCESSING"
    cv_filter_status = ""
    if args.single_batch:
        if args.cv_filtering_only:
            cv_filter_status = " (CV filtering ENABLED)"
        else:
            cv_filter_status = " (CV filtering DISABLED)"

    # Log startup information (enhanced with trimming info)
    logger.info("=" * 80)
    logger.info(f"fNIRS Multi-Task Processing Pipeline Started - {processing_method}")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Processing method: {processing_method}{cv_filter_status}")
    logger.info(f"Sampling rate: {args.fs} Hz")
    logger.info(f"CV threshold: {args.cv_thresh}%")
    logger.info(f"Post-walking trimming: {args.post_walking_trim}s")  # NEW LINE
    logger.info(f"Task filter: {args.task_filter if args.task_filter else 'All tasks'}")
    logger.info(f"Strict validation: {args.strict_validation}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    if not args.quiet:
        print(" Starting fNIRS Multi-Task Processing Pipeline")
        print(f" Input: {args.input_dir}")
        print(f" Output: {args.output_dir}")
        print(f"  Method: {processing_method}{cv_filter_status}")

        if args.dual_cv:
            print(" Dual CV Processing:")
            print("   - Batch 1: All channels included in analysis")
            print("   - Batch 2: Poor quality channels excluded from analysis")
            print(f"   - CV threshold: {args.cv_thresh}%")

        print(f"🔧 Post-walking trimming: {args.post_walking_trim}s after walking start event")

        if args.task_filter:
            print(f" Processing only: {', '.join(args.task_filter)}")
        if args.strict_validation:
            print("  Strict validation enabled (event-dependent tasks require sufficient markers)")
        print(f" Logging to: {log_file}")

    try:
        # Run the pipeline
        if args.dual_cv:
            # Dual CV processing
            results = processor.run_dual_cv_processing(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                task_filter=args.task_filter,
                strict_validation=args.strict_validation
            )

            # Display dual batch results
            if not args.quiet:
                print("\n" + "=" * 80)
                print(" DUAL CV PROCESSING COMPLETE!")
                print("=" * 80)
                print(f" Input directory: {args.input_dir}")
                print(f" Output directory: {args.output_dir}")
                print(f" Total files found: {results['total_files']}")
                print(f" Post-walking trimming: {args.post_walking_trim}s applied to both batches")

                print(f"\n BATCH 1 RESULTS (All Channels):")
                print(f"  Files processed: {len(results['batch1_processed'])}/{results['total_files']}")
                if results['total_files'] > 0:
                    success_rate1 = len(results['batch1_processed']) / results['total_files'] * 100
                    print(f"  Success rate: {success_rate1:.1f}%")

                print(f"\n BATCH 2 RESULTS (CV Filtered):")
                print(f"  Files processed: {len(results['batch2_processed'])}/{results['total_files']}")
                if results['total_files'] > 0:
                    success_rate2 = len(results['batch2_processed']) / results['total_files'] * 100
                    print(f"  Success rate: {success_rate2:.1f}%")

                # Show task-specific results for both batches
                print(f"\n Results by Task Type:")
                for task_type in sorted(results['task_results']['no_cv_filtering']['processed'].keys()):
                    batch1_processed = len(results['task_results']['no_cv_filtering']['processed'][task_type])
                    batch2_processed = len(results['task_results']['with_cv_filtering']['processed'][task_type])
                    batch1_validation_failed = len(
                        results['task_results']['no_cv_filtering']['validation_failed'][task_type])
                    batch2_validation_failed = len(
                        results['task_results']['with_cv_filtering']['validation_failed'][task_type])

                    if batch1_processed + batch2_processed + batch1_validation_failed + batch2_validation_failed > 0:
                        print(f"  {task_type}:")
                        print(
                            f"    All channels: {batch1_processed} processed, {batch1_validation_failed} validation failed")
                        print(
                            f"    CV filtered:  {batch2_processed} processed, {batch2_validation_failed} validation failed")

                # Output file information
                print(f"\n Output Files Generated:")
                if results['stats_no_cv'] is not None:
                    print(f"  Statistics (all channels): all_subjects_statistics_no_CV_filtering.csv")
                if results['stats_with_cv'] is not None:
                    print(f"  Statistics (CV filtered): all_subjects_statistics_with_CV_filtering.csv")
                print(f"  Detailed report: dual_batch_processing_report.txt")
                print(f"  Log file: {os.path.basename(log_file)}")
                print("=" * 80)

        else:
            # Single batch processing (original behavior or CV filtering only)
            enable_cv_filtering = args.cv_filtering_only

            # Temporarily modify processor for CV filtering only mode
            if enable_cv_filtering:
                # Use the CV filtering processor
                results = processor._process_files_batch(
                    processor.find_input_files(args.input_dir, args.task_filter),
                    args.input_dir, args.output_dir,
                    processor.processor_with_cv, "with_cv_filtering",
                    None, args.strict_validation
                )
                # Convert to expected format
                results = {
                    'processed_files': results[0],
                    'skipped_files': results[1],
                    'total_files': len(results[0]) + len(results[1]),
                    'stats': None
                }
            else:
                # Use original processing
                results = processor._run_original_processing(
                    args.input_dir, args.output_dir, args.task_filter, args.strict_validation
                )

            # Display single batch results
            if not args.quiet:
                print("\n" + "=" * 80)
                print(" PROCESSING COMPLETE!")
                print("=" * 80)
                print(f" Input directory: {args.input_dir}")
                print(f" Output directory: {args.output_dir}")
                cv_status = " (CV Filtered)" if enable_cv_filtering else " (All Channels)"
                print(f" Files processed{cv_status}: {len(results['processed_files'])}/{results['total_files']}")
                print(f" Post-walking trimming: {args.post_walking_trim}s applied to all files")

                if results['total_files'] > 0:
                    success_rate = len(results['processed_files']) / results['total_files'] * 100
                    print(f" Overall success rate: {success_rate:.1f}%")

                if results.get('skipped_files'):
                    print(f"\n  Total skipped/failed: {len(results['skipped_files'])}")

                if results.get('stats') is not None:
                    stats_filename = 'all_subjects_statistics_with_CV_filtering.csv' if enable_cv_filtering else 'all_subjects_statistics.csv'
                    print(f"📈 Statistics saved to: {stats_filename}")

                report_filename = 'task_processing_report.txt'
                print(f" Task report saved to: {report_filename}")
                print(f" Log file saved to: {os.path.basename(log_file)}")
                print("=" * 80)

        # Log completion
        logger.info("Pipeline completed successfully")
        if args.dual_cv:
            logger.info(
                f"Dual CV processing: Batch 1: {len(results['batch1_processed'])}, Batch 2: {len(results['batch2_processed'])}")
        else:
            logger.info(f"Single batch processing: {len(results['processed_files'])} files processed")

        logger.info(f"Post-walking trimming: {args.post_walking_trim}s applied to all processed files")

        # Return appropriate exit code
        if args.dual_cv:
            return 0 if (results['batch1_processed'] or results['batch2_processed']) else 2
        else:
            return 0 if results['processed_files'] else 2

    except KeyboardInterrupt:
        error_msg = "Pipeline interrupted by user"
        logger.warning(error_msg)
        if not args.quiet:
            print(f"\n️  {error_msg}")
        return 130  # Standard exit code for SIGINT

    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        logger.critical(error_msg)
        if not args.quiet:
            print(f" Error: {error_msg}")
        return 1

    except PermissionError as e:
        error_msg = f"Permission denied: {str(e)}"
        logger.critical(error_msg)
        if not args.quiet:
            print(f" Error: {error_msg}")
        return 1

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        if not args.quiet:
            print(f" Error: {error_msg}")
            print(f" Check log file for details: {log_file}")
        return 1


if __name__ == "__main__":
    exit(main())